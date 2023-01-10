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

using LinearAlgebra: norm, I, nullspace
using SparseArrays
using StaticArrays
using ConvexBodyProximityQueries

global status_bar_period = 30.0

include("gpwrapper.jl")
include("GPBounding/GPBounding.jl")
include("rkhs.jl")
include("discretization.jl")
include("images.jl")
include("refinement.jl")
include("transitions.jl")
include("imdptools.jl")
include("merging.jl")

function config_entry_try(dict, key, default_value)
	res = default_value
	try
		res = dict[key]
	catch
		@warn "Key $key not found in configuration dictionary. Using default value $default_value"
	end
	return res
end

function learn_abstraction(config_file::String)
	f = open(config_file)
	config = TOML.parse(f)
	close(f)
	
	L = SA_F64[config["workspace"]["lower"]...]
	U = SA_F64[config["workspace"]["upper"]...]
	domain_type = config["workspace"]["domain_type"]
	X_extent = [[l u] for (l, u) in zip(L,U)]
	diameter_domain = sqrt(sum((L-U).^2))
	grid_spacing = SA_F64[config["discretization"]["grid_spacing"]...]
	grid = LearningAbstractions.grid_generator(L, U, grid_spacing)
	lipschitz_bound = config["system"]["lipschitz_bound"] 

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
		@info "System has process noise."
	else
		process_noise_dist = nothing
		@info "System has measurement noise."
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
	reloaded_states_flag = false
	reloaded_results_flag = false

	# Local GP setup
	local_gps_flag = config["local"]["use_local_gps"]
	local_gps_nns = config["local"]["local_gp_neighbors"]
	full_gp_subset = config["local"]["full_gp_subset"]
	local_gp_metadata = nothing

	if local_gps_flag
		@info "Performing local GP regression with $local_gps_nns-nearest neighbors"
		# TODO: This needs to be put somewhere else!!
		local_gp_metadata = [ones(length(lipschitz_bound)), 0.65*ones(length(lipschitz_bound)), local_gps_nns]
	end

	if config["reuse_results"] && isfile(imdp_filename)
		@info "Reloading all state information and IMDP transitions from $base_results_dir"
		state_dict = BSON.load(state_filename)
		all_states_SA = state_dict[:states]
		all_state_images = state_dict[:images]
		all_state_σ_bounds = state_dict[:bounds]
		
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
		reloaded_states_flag = true

		gps = LearningAbstractions.condition_gps(input_data, output_data, data_subset=full_gp_subset)
		diameter_domain = sqrt(sum((L-U).^2))
		sup_f = U + lipschitz_bound*diameter_domain
		gp_info = LearningAbstractions.create_gp_info(gps, σ_noise, diameter_domain, sup_f, process_noise=process_noise_flag)
		gp_info_dict = create_gp_info_dict(gp_info)
		save_gps(Dict(:gps => gps, :info => gp_info_dict), gps_filename)
		P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_SA, all_state_images, LearningAbstractions.extent_to_SA(X_extent), process_noise_dist=process_noise_dist, gp_rkhs_info=gp_info, σ_bounds_all=all_state_σ_bounds, local_gp_metadata=local_gp_metadata)
	else
		gps = LearningAbstractions.condition_gps(input_data, output_data, data_subset=full_gp_subset)
		diameter_domain = sqrt(sum((L-U).^2))
		sup_f = U + lipschitz_bound*diameter_domain
		gp_info = LearningAbstractions.create_gp_info(gps, σ_noise, diameter_domain, sup_f, process_noise=process_noise_flag)
		gp_info_dict = create_gp_info_dict(gp_info)
		save_gps(Dict(:gps => gps, :info => gp_info_dict), gps_filename)

		n_states = length(grid)
		all_states_SA = Vector{SMatrix}(undef, n_states)
		[all_states_SA[i] = LearningAbstractions.lower_to_SA(grid_lower, grid_spacing) for (i,grid_lower) in enumerate(grid)]
		all_state_images, all_state_σ_bounds = state_bounds(all_states_SA, gps; local_gps_flag=local_gps_flag, local_gps_data=(input_data, output_data), local_gps_nns=local_gps_nns, domain_type=domain_type, delta_input_flag=delta_input_flag)

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
