module LearningAbstractions

using TOML
using BSON
using Serialization
using ProgressMeter

using GaussianProcesses
using NearestNeighbors
using Distances
import Distances: evaluate
using StatsBase
using Distributions

using LinearAlgebra
using LinearAlgebra: norm, I, nullspace
using LinearAlgebra: BlasReal, BlasFloat
using PDMats

using SparseArrays
using StaticArrays

using PosteriorBounds

using StatsBase
using Random

global status_bar_period = 30.0

include("gpwrapper.jl")
include("rkhs.jl")
include("discretization.jl")
include("images.jl")
include("refinement.jl")
include("transitions.jl")
include("imdptools.jl")
include("merging.jl")
include("output.jl")

function config_entry_try(dict, key, default_value)
	res = default_value
	try
		res = dict[key]
	catch
		@warn "Key $key not found in configuration dictionary. Using default value $default_value"
	end
	return res
end

function learn_abstraction(config_filename::String;)
	f = open(config_filename)
	config = TOML.parse(f)
	close(f)
	
	# System config parsing
	L = SA_F64[config["workspace"]["lower"]...]
	U = SA_F64[config["workspace"]["upper"]...]
	angle_dims = config_entry_try(config["workspace"], "angle_dims", [])
	norm_weights = config_entry_try(config["workspace"], "norm_weights", ones(length(L)))
	distance_metric = GeneralMetric(angle_dims, norm_weights)
	X_extent = [[l u] for (l, u) in zip(L,U)]
	diameter_domain = sqrt(sum((L-U).^2))
	desired_spacing = SA_F64[config["discretization"]["grid_spacing"]...]
	grid, grid_spacing = LearningAbstractions.grid_generator(L, U, desired_spacing)
	lipschitz_bound = config["system"]["lipschitz_bound"] 

	# Local GP setup
	local_gps_flag = config["local"]["use_local_gps"]
	local_gps_nns = config["local"]["local_gp_neighbors"]
	full_gp_subset = config["local"]["full_gp_subset"]
	approximate_σ_flag = config_entry_try(config["system"], "approximate_sigma", false)
	local_gp_metadata = nothing

	# Datafile parsing
	data_filename = config["system"]["datafile"]
	res = BSON.load(data_filename)
	data_dict = res[:dataset_dict]
	input_data = data_dict[:input]
	output_data = data_dict[:output]
	if data_dict[:noise]["measurement_std"] > 0.0 && data_dict[:noise]["process_std"] > 0.0
		@error "Only one of either measurement or process noise is supported."	
	end
	σ_noise = max(data_dict[:noise]["measurement_std"], data_dict[:noise]["process_std"]) # One of these is zero, so take which one is not
	process_noise_flag = data_dict[:noise]["process_std"] > 0.0 

	if process_noise_flag
		noise_config = data_dict[:noise] 
		process_noise_dist = create_noise_dist(noise_config)
		@info "System has process noise"
	else
		process_noise_dist = nothing
		@info "System has measurement noise"
	end

	delta_input_flag = config_entry_try(config["system"], "delta_input_flag", false) # flag to indicate training on the delta from each point, i.e.
	if delta_input_flag
		output_data -= input_data[1:size(output_data,1),:]
	end

	results_dir = config["results_directory"]
	base_results_dir = "$results_dir/base"
	mkpath(base_results_dir)
	state_filename = "$base_results_dir/states.bson"
	imdp_filename = "$base_results_dir/imdp.bson"
	gps_filename = "$base_results_dir/gps.bson"
	
	all_states_SA, all_state_images, all_state_σ_bounds, P̌, P̂ = load_results(state_filename, imdp_filename, reuse_states=config["reuse_states"], reuse_results=config["reuse_results"])
	reloaded_states_flag = !isnothing(all_states_SA)
	reloaded_results_flag = !isnothing(P̌)
	
	if !reloaded_results_flag
		if false && isfile(gps_filename)
			@info "Loading existing GPs from $gps_filename"
			# TODO: Any benefit to doing this?
		end

		@info "Performing GP regression and saving to $gps_filename"
		gps = LearningAbstractions.condition_gps(input_data, output_data, data_subset=full_gp_subset)
		diameter_domain = sqrt(sum((L-U).^2))
		sup_f = U + lipschitz_bound*diameter_domain
		gp_info = LearningAbstractions.create_gp_info(gps, σ_noise, diameter_domain, sup_f, process_noise=process_noise_flag)
		gp_info_dict = create_gp_info_dict(gp_info)
		save_gps(Dict(:gps => gps, :info => gp_info_dict), gps_filename)
	end

	if !reloaded_states_flag
		@info "Determing the state images under GP dynamics"
		n_states = length(grid)
		all_states_SA = Vector{SMatrix}(undef, n_states)
		[all_states_SA[i] = LearningAbstractions.lower_to_SA(grid_lower, grid_spacing) for (i,grid_lower) in enumerate(grid)]
		all_state_images, all_state_σ_bounds = state_bounds(all_states_SA, gps; local_gps_flag=local_gps_flag, local_gps_data=(input_data, output_data), local_gps_nns=local_gps_nns, metric=distance_metric, delta_input_flag=delta_input_flag, approximate_σ_flag=approximate_σ_flag)
	end

	if local_gps_flag
		@info "Performing local GP regression with $local_gps_nns-nearest neighbors"
		# TODO: This needs to be put somewhere else!!
		local_gp_metadata = [ones(length(lipschitz_bound)), 0.65*ones(length(lipschitz_bound)), local_gps_nns]
	end

	if !reloaded_results_flag	
		@info "Calculating the transition probability intervals"
		P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_SA, all_state_images, LearningAbstractions.extent_to_SA(X_extent), process_noise_dist=process_noise_dist, gp_rkhs_info=gp_info, σ_bounds_all=all_state_σ_bounds, local_gp_metadata=local_gp_metadata)
	end
	
	if config["save_results"] && !reloaded_results_flag
		@info "Saving abstraction info to $base_results_dir"
		mkpath(base_results_dir)

		if !reloaded_states_flag && !reloaded_results_flag
			bson(state_filename, Dict(:states => all_states_SA,
								:images => all_state_images,
								:bounds => all_state_σ_bounds,
								)
			)
		end

		if !reloaded_results_flag
			bson(imdp_filename, Dict(:Pcheck => P̌, :Phat => P̂))
		end
	end

	return P̌, P̂, all_states_SA, base_results_dir, all_state_images, all_state_σ_bounds
end

function generate_abstraction(config_filename::String, f_sys; linear_map_flag=false)	# System function f

	# > 0. Get the deets from the config files
	f = open(config_filename)
	config = TOML.parse(f)
	close(f)
	
	# System config parsing
	L = SA_F64[config["workspace"]["lower"]...]
	U = SA_F64[config["workspace"]["upper"]...]
	X_extent = [[l u] for (l, u) in zip(L,U)]
	desired_spacing = SA_F64[config["discretization"]["grid_spacing"]...]
	grid, grid_spacing = LearningAbstractions.grid_generator(L, U, desired_spacing)

	# Datafile parsing
	data_filename = config["system"]["datafile"]
	res = BSON.load(data_filename)
	data_dict = res[:dataset_dict]
	if data_dict[:noise]["measurement_std"] > 0.0 && data_dict[:noise]["process_std"] > 0.0
		@error "Only one of either measurement or process noise is supported."	
	end
	process_noise_flag = data_dict[:noise]["process_std"] > 0.0 

	if process_noise_flag
		noise_config = data_dict[:noise] 
		process_noise_dist = create_noise_dist(noise_config)
		@info "System has process noise"
	else
		process_noise_dist = nothing
		@info "System has measurement noise"
	end

	results_dir = config["results_directory"]
	base_results_dir = "$results_dir/known_system"
	mkpath(base_results_dir)
	state_filename = "$base_results_dir/states.bson"
	imdp_filename = "$base_results_dir/imdp.bson"

	all_states_SA, all_state_images, all_state_σ_bounds, P̌, P̂ = load_results(state_filename, imdp_filename, reuse_states=config["reuse_states"], reuse_results=config["reuse_results"])
	reloaded_states_flag = !isnothing(all_states_SA)
	reloaded_results_flag = !isnothing(P̌)

	# > 1. Using sampling based method, just get an estimate of the image
	if !reloaded_states_flag

		n_states = length(grid)
		all_states_SA = Vector{SMatrix}(undef, n_states)
		[all_states_SA[i] = LearningAbstractions.lower_to_SA(grid_lower, grid_spacing) for (i,grid_lower) in enumerate(grid)]
		all_state_images = []

		if linear_map_flag
			@info "Determing the state images with linear mapping"
			for state in all_states_SA
				lower = f_sys(state[:,1])
				upper = f_sys(state[:,end-1])
				extent = [[lower[i], upper[i]] for i=1:length(lower)]
				image = extent_to_SA(extent)
				push!(all_state_images, image)
			end
		else
			N = 1000
			@info "Determing the state images via sampling with N=$N samples"
			mt = MersenneTwister(11)
			for state in all_states_SA	
				x_samp = vcat([rand(mt, Uniform(state[i,1], state[i,end-1]), 1, N) for i=1:size(state,1)]...)
				f_samp = hcat([f_sys(c) for c in eachcol(x_samp)]...)
				lower = [minimum(v) for v in eachrow(f_samp)]
				upper = [maximum(v) for v in eachrow(f_samp)]
				extent = [[lower[i], upper[i]] for i=1:length(lower)]
				image = extent_to_SA(extent)
				push!(all_state_images, image)
			end
		end

		
		all_state_σ_bounds = nothing
	end

	# > 2. Generate transitions 
	if !reloaded_results_flag	
		@info "Calculating the transition probability intervals"
		P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_SA, all_state_images, LearningAbstractions.extent_to_SA(X_extent), process_noise_dist=process_noise_dist)
	end

	# > 3. Save and move on!
	all_state_σ_bounds = nothing
	if config["save_results"] && !reloaded_results_flag
		@info "Saving abstraction info to $base_results_dir"
		mkpath(base_results_dir)

		if !reloaded_states_flag && !reloaded_results_flag
			bson(state_filename, Dict(:states => all_states_SA,
								:images => all_state_images,
								:bounds => all_state_σ_bounds,
								)
			)
		end

		if !reloaded_results_flag
			bson(imdp_filename, Dict(:Pcheck => P̌, :Phat => P̂))
		end
	end

	return P̌, P̂, all_states_SA, base_results_dir, all_state_images, all_state_σ_bounds
end

"""
Create an explicit distribution from a config dictionary.
"""
function create_noise_dist(config)
	# TODO: Change data config struct to use general terms
	noise_dist = nothing
	if config["process_distribution"] == "Gaussian"
		noise_dist = Normal(config["process_mean"], config["process_std"])
	else
		# TODO: Add bounded Gaussian, etc;
	end
	return noise_dist
end

end
