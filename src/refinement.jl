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

    num_states = length(all_states_SA)
    num_refine_states = length(states_to_refine)
    frac = length(states_to_refine)/num_states
    @info "Refining $num_refine_states of $num_states ($frac) states"
	
    results_dir = config["results_directory"]
	# state_filename = "$results_dir/states.bson"
	# imdp_filename = "$results_dir/imdp.bson"
	gps_filename = "$results_dir/gps.bson"

    refinement_dir = isnothing(refinement_dirname) ? "$results_dir/refined" : "$results_dir/$refinement_dirname"
    !isdir(refinement_dir) && mkpath(refinement_dir)
    state_refined_filename = "$refinement_dir/states.bson"
	imdp_refined_filename = "$refinement_dir/imdp.bson"

    if config["reuse_results"] && isfile(state_refined_filename) && isfile(imdp_refined_filename)
        # TODO: Function for reloading 
        @info "Reloading all state information and IMDP transitions from $results_dir"
		state_dict = BSON.load(state_refined_filename)

		all_states_refined = state_dict[:states]
		all_state_images_refined = state_dict[:images]
		all_σ_bounds_refined = state_dict[:bounds]
		
		imdp_dict = BSON.load(imdp_refined_filename)
		P̌ = imdp_dict[:Pcheck]
		P̂ = imdp_dict[:Phat]
    else
        # TODO: This is all the same!
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
        else
            local_gps_data = nothing
        end

        L = SA_F64[config["workspace"]["lower"]...]
        U = SA_F64[config["workspace"]["upper"]...]
        X_extent = [[l u] for (l, u) in zip(L,U)]

        # Load the existing global GPs
        gps, gp_info = load_gps(gps_filename)

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

        new_images, new_σ_bounds = state_bounds(new_states_list, gps; local_gps_flag=local_gps_flag, local_gps_data=local_gps_data, local_gps_nns=local_gps_nns)
        all_state_images_refined = vcat(all_state_images_refined, new_images)
        all_σ_bounds_refined = vcat(all_σ_bounds_refined, new_σ_bounds)

        P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_refined, all_state_images_refined, LearningAbstractions.extent_to_SA(X_extent), gp_rkhs_info=gp_info, σ_bounds_all=all_σ_bounds_refined)

        @info "Saving abstraction info to $refinement_dir"
        
        bson(state_refined_filename, Dict(:states => all_states_refined,
                            :images => all_state_images_refined,
                            :bounds => all_σ_bounds_refined,
                            )
        )

        bson(imdp_refined_filename, Dict(:Pcheck => P̌, :Phat => P̂))
    end
    return P̌, P̂, all_states_refined, refinement_dir, all_state_images_refined, all_σ_bounds_refined
end

function find_states_to_refine(P̂, res_mat; p_threshold=0.95)

    n_yes = findall(x -> x>=p_threshold, res_mat[:,3])
    n_no = findall(x -> x<p_threshold, res_mat[:,4])

    if isempty(n_yes)
        # if nothing is yes, such as in safety, return all states for refinement
        return res_mat[:,1]
    end

    # now n_yes consists of states that satisfy the spec
    # the idea is to only refine states that may possibly satisfy and can transition to this set
    states_to_refine = []
    
    for n in n_yes
        poss_states = findall(x -> x>0., P̂[:,n]) 
        setdiff!(poss_states, n_yes)
        setdiff!(poss_states, n_no)
        setdiff!(poss_states, states_to_refine)
        union!(states_to_refine, poss_states) 
    end

    return sort(states_to_refine)
end