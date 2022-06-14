"""
    uniform_refinement

Refine a discrete state uniformly and update the region info dict.
"""
function uniform_refinement(state)
    deltas = (state[:,end-1] - state[:,1])/2
    refined_grid = grid_generator(state[:,1], state[:,end-1], deltas)
    new_states = [lower_to_SA(gl, deltas) for gl in refined_grid]
    return new_states
end

"""
    refine_abstraction

Refine the abstraction.
"""
# TODO: There is a lot of redundant code here with the find images function. Consolidate when I have more time. 
function refine_abstraction(config_filename, all_states_SA, all_state_images, all_σ_bounds, states_to_refine; refinement_dirname=nothing)
    f = open(config_filename)
	config = TOML.parse(f)
	close(f)
	
    results_dir = config["results_directory"]
	# state_filename = "$results_dir/states.bson"
	# imdp_filename = "$results_dir/imdp.bson"
	gps_filename = "$results_dir/gps.bson"

    refinement_dir = isnothing(refinement_dirname) ? "$results_dir/refined" : "$results_dir/$refinement_dirname"
    !isdir(refinement_dir) && mkpath(refinement_dir)
    state_refined_filename = "$refinement_dir/states.bson"
	imdp_refined_filename = "$refinement_dir/imdp.bson"

    if config["reuse_results"] && isfile(state_refined_filename) && isfile(imdp_refined_filename)
        @info "Reloading all state information and IMDP transitions from $results_dir"
		state_dict = BSON.load(state_refined_filename)

		all_states_refined = state_dict[:states]
		all_state_images_refined = state_dict[:images]
		all_σ_bounds_refined = state_dict[:bounds]
		all_state_means = state_dict[:state_means]
		all_image_means = state_dict[:image_means]
		
		imdp_dict = BSON.load(imdp_refined_filename)
		P̌ = imdp_dict[:Pcheck]
		P̂ = imdp_dict[:Phat]
    else
        # Local GP setup
        local_gps_flag = config["local"]["use_local_gps"]
        local_gps_nns = config["local"]["local_gp_neighbors"]
        if local_gps_flag
            @info "Performing local GP regression with $local_gps_nns-nearest neighbors"
            # Reload the data here
            data_filename = config["system"]["datafile"]
            res = BSON.load(data_filename)
            data_dict = res[:dataset_dict]
            input_data = data_dict[:input]
            output_data = data_dict[:output]
            local_gps_data = (input_data, output_data)
        end

        L = SA_F64[config["workspace"]["lower"]...]
        U = SA_F64[config["workspace"]["upper"]...]
        X_extent = [[l u] for (l, u) in zip(L,U)]

        # Load the GPs // or // construct the local GPs 
        gps, gp_info = load_gps(gps_filename)
        neg_gps = []
        for gp in gps
            neg_gp = deepcopy(gp)
            neg_gp.alpha *= -1
            push!(neg_gps, neg_gp)
        end

        # Generate new discretization
        new_states_list = []
        for state_to_refine in states_to_refine
            new_states = uniform_refinement(all_states_SA[state_to_refine])
            push!(new_states_list, new_states...)
        end

        # delete the old states
        all_states_refined = copy(all_states_SA)
        all_state_images_refined = copy(all_state_images)
        all_σ_bounds_refined = copy(all_σ_bounds)

        deleteat!(all_states_refined, states_to_refine)
        push!(all_states_refined, new_states_list...)

        # delete old image bounds
        deleteat!(all_state_images_refined, states_to_refine)
        deleteat!(all_σ_bounds_refined, states_to_refine)

        # Setup for bounding refined states with local GP regression
        if local_gps_flag
            kdtree = KDTree(local_gps_data[1])
        end

        # Generate new posterior bounds
        p = Progress(length(new_states_list), desc="Computing image bounds...", dt=0.01)
        Threads.@threads for new_state in new_states_list 

            # If local GP, create local GP here
            if local_gps_flag
                state_mean = 0.5*(new_state[:,1] + new_state[:,end-1])
                local_gps = create_local_gps(local_gps_data[1], local_gps_data[2], state_mean, num_neighbors=local_gps_nns, kdtree=kdtree)
                local_neg_gps = []
                for gp in local_gps
                    neg_gp = deepcopy(gp)
                    neg_gp.alpha *= -1
                    push!(local_neg_gps, neg_gp)
                end
                image, σ_bounds = LearningAbstractions.GPBounding.bound_image([new_state[:,1], new_state[:,end-1]], local_gps, local_neg_gps) 
            
            else
                image, σ_bounds = LearningAbstractions.GPBounding.bound_image([new_state[:,1], new_state[:,end-1]], gps, neg_gps)
            end
            push!(all_state_images_refined, extent_to_SA(image))
            push!(all_σ_bounds_refined, σ_bounds)
            next!(p)
        end

        all_state_means = [0.5*(s[:,1] + s[:,end-1]) for s in all_states_refined]
        all_image_means = [0.5*(s[:,1] + s[:,end-1]) for s in all_state_images_refined] 

        P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_refined, all_state_images_refined, all_state_means, all_image_means, LearningAbstractions.extent_to_SA(X_extent), gp_rkhs_info=gp_info, σ_bounds_all=all_σ_bounds_refined)

        @info "Saving abstraction info to $refinement_dir"
        
        bson(state_refined_filename, Dict(:states => all_states_refined,
                            :images => all_state_images_refined,
                            :bounds => all_σ_bounds_refined,
                            :state_means => all_state_means,
                            :image_means => all_image_means)
        )

        bson(imdp_refined_filename, Dict(:Pcheck => P̌, :Phat => P̂))
    end
    return P̌, P̂, all_states_refined, all_state_means, refinement_dir, all_state_images_refined, all_σ_bounds_refined
end