module LearningAbstractions

using TOML
using BSON
using ProgressMeter

using GaussianProcesses
using NearestNeighbors
using StatsBase

using Meshes
using LinearAlgebra: norm, I, nullspace
using Plots
using SparseArrays
using StaticArrays
using ConvexBodyProximityQueries

include("data.jl")
include("gpwrapper.jl")
include("GPBounding/GPBounding.jl")
include("rkhs.jl")
include("discretization.jl")
include("transitions.jl")
include("imdptools.jl")
include("plotting.jl")

function learn_abstraction(config_file::String)
	f = open(config_file)
	config = TOML.parse(f)
	close(f)
	
	L = SA_F64[config["workspace"]["lower"]...]
	U = SA_F64[config["workspace"]["upper"]...]
	X_extent = [[l u] for (l, u) in zip(L,U)]
	diameter_domain = sqrt(sum((L-U).^2))
	grid_spacing = SA_F64[config["discretization"]["grid_spacing"]...]
	grid = LearningAbstractions.grid_generator(L, U, grid_spacing)
	
	lipschitz_bound = config["system"]["lipschitz_bound"] 
	# TODO: Get this from the dataset
	σ_noise = config["system"]["measurement_noise_sigma"]

	data_filename = config["system"]["datafile"]
	res = BSON.load(data_filename)
	data_dict = res[:dataset_dict]
	input_data = data_dict[:input]
	output_data = data_dict[:output]

	results_dir = config["results_directory"]
	state_filename = "$results_dir/states.bson"
	imdp_filename = "$results_dir/imdp.bson"
	reloaded_states_flag = false
	reloaded_results_flag = false
	
	if config["reuse_results"] && isfile(imdp_filename)
		@info "Reloading all state information and IMDP transitions from $results_dir"
		state_dict = BSON.load(state_filename)
		all_states_SA = state_dict[:states]
		all_state_images = state_dict[:images]
		all_state_σ_bounds = state_dict[:bounds]
		all_state_means = state_dict[:state_means]
		all_image_means = state_dict[:image_means]
		
		imdp_dict = BSON.load(imdp_filename)
		P̌ = imdp_dict[:Pcheck]
		P̂ = imdp_dict[:Phat]
		reloaded_results_flag = true
	elseif config["reuse_states"] && isfile(state_filename)
		@info "Reloading state and image definitions from $state_filename"
		state_dict = BSON.load(state_filename)
		all_states_SA = state_dict[:states]
		all_state_images = state_dict[:images]
		all_state_σ_bounds = state_dict[:bounds]
		all_state_means = state_dict[:state_means]
		all_image_means = state_dict[:image_means]
		reloaded_states_flag = true

		gps = LearningAbstractions.condition_gps(input_data, output_data)
		diameter_domain = sqrt(sum((L-U).^2))
		sup_f = maximum(U) + lipschitz_bound*diameter_domain
		gp_info = LearningAbstractions.create_gp_info(gps, σ_noise, diameter_domain, sup_f)

		P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_SA, all_state_images, all_state_means, all_image_means, LearningAbstractions.extent_to_SA(X_extent), gp_rkhs_info=gp_info, σ_bounds_all=all_state_σ_bounds)
	else
		gps = LearningAbstractions.condition_gps(input_data, output_data)
		diameter_domain = sqrt(sum((L-U).^2))
		sup_f = maximum(U) + lipschitz_bound*diameter_domain
		gp_info = LearningAbstractions.create_gp_info(gps, σ_noise, diameter_domain, sup_f)

		all_states_SA, 
		all_state_images, 
		all_state_σ_bounds, 
		all_state_means, 
		all_image_means = LearningAbstractions.find_state_images(grid, gps, grid_spacing)
		P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_SA, all_state_images, all_state_means, all_image_means, LearningAbstractions.extent_to_SA(X_extent), gp_rkhs_info=gp_info, σ_bounds_all=all_state_σ_bounds)
	end
	
	if config["save_results"] && !reloaded_results_flag
		@info "Saving abstraction info to $results_dir"
		mkpath(results_dir)

		if !reloaded_states_flag && !reloaded_results_flag
			bson(state_filename, Dict(:states => all_states_SA,
								:images => all_state_images,
								:bounds => all_state_σ_bounds,
								:state_means => all_state_means,
								:image_means => all_image_means)
			)
		end

		if !reloaded_results_flag
			bson(imdp_filename, Dict(:Pcheck => P̌, :Phat => P̂))
		end
	end

	return P̌, P̂, all_states_SA, all_state_means, results_dir
end

function find_state_images(grid, gps, grid_spacing; local_gps_flag=false, local_gps_nns=200, local_gps_data=nothing)
	n_states = length(grid)
	all_states_SA = Vector{SMatrix}(undef, n_states)
	all_state_images = Vector{Any}(undef, n_states)
	all_state_σ_bounds = Vector{Any}(undef, n_states) 
	all_state_means = Vector{SVector}(undef, n_states) 
	all_image_means = Vector{Any}(undef, n_states) 

	p = Progress(n_states, desc="Computing image bounds...", dt=0.01)

	neg_gps = []
	for gp in gps
		neg_gp = deepcopy(gp)
		neg_gp.alpha *= -1
		push!(neg_gps, neg_gp)
	end

	if local_gps_flag
		kdtree = KDTree(local_gps_data[1])
	end

	Threads.@threads for (i, grid_lower) in collect(enumerate(grid))     # Implicit ordering of the states remains the same
		state = LearningAbstractions.lower_to_SA(grid_lower, grid_spacing)
		all_states_SA[i] = state
		all_state_means[i] = (state[:,1] + state[:,end-1])/2

		if local_gps_flag
			local_gps = create_local_gps(local_gps_data[1], local_gps_data[2], all_state_means[i], num_neighbors=local_gps_nns, kdtree=kdtree)
			local_neg_gps = []
			for gp in local_gps
				neg_gp = deepcopy(gp)
				neg_gp.alpha *= -1
				push!(local_neg_gps, neg_gp)
			end
			image, all_state_σ_bounds[i] = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], local_gps, local_neg_gps) 
		else
			image, all_state_σ_bounds[i] = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], gps, neg_gps) 
		end

		all_state_images[i] = extent_to_SA(image)
		all_image_means[i] = (all_state_images[i][:,1] + all_state_images[i][:,end-1])/2
		next!(p)
	end

	return all_states_SA, all_state_images, all_state_σ_bounds, all_state_means, all_image_means 
end

end
