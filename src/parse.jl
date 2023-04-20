# Functions used for parsing

function parse_discretization_params(discretization_dict::Dict)
	L = SA_F64[discretization_dict["lower"]...]
	U = SA_F64[discretization_dict["upper"]...]	
	desired_spacing = SA_F64[discretization_dict["grid_spacing"]...]
	refinement_procedure = config_entry_try(discretization_dict, "refinement_procedure", "uniform") 
	# TODO: add other discretization params here!	

	X_extent = [[l u] for (l, u) in zip(L,U)]
	return L, U, X_extent, desired_spacing, refinement_procedure
end

function parse_data_params(data_config_dict::Dict; load_data=false)
	data_filename = data_config_dict["datafile"]
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

	if load_data
		input_data = data_dict[:input]
		output_data = data_dict[:output]

		σ_noise = max(data_dict[:noise]["measurement_std"], data_dict[:noise]["process_std"]) # One of these is zero, so take which one is not

		delta_input_flag = config_entry_try(data_config_dict, "delta_input_flag", false) # flag to indicate training on the delta from each point, i.e.
		if delta_input_flag
			output_data -= input_data[1:size(output_data,1),:]
		end
		return input_data, output_data, σ_noise, delta_input_flag, process_noise_flag, process_noise_dist
	else
		return process_noise_dist
	end
end

function parse_results_params(config::Dict)
	results_dir = config["results_directory"]
	base_results_dir = "$results_dir/known_system"
	mkpath(base_results_dir)
	state_filename = "$base_results_dir/states.bson"
	imdp_filename = "$base_results_dir/imdp.bson"
	gps_filename = "$base_results_dir/gps.bson"
	return base_results_dir, state_filename, imdp_filename, gps_filename
end

function parse_system_params(system_dict::Dict)
	lipschitz_bound = system_dict["lipschitz_bound"] 
	angle_dims = config_entry_try(system_dict, "angle_dims", [])
	norm_weights = config_entry_try(system_dict, "norm_weights", ones(length(lipschitz_bound)))
	distance_metric = GeneralMetric(angle_dims, norm_weights)
	return distance_metric, lipschitz_bound
end