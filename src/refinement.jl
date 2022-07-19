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
function refine_abstraction(config_filename, all_states_SA, all_state_images, all_σ_bounds, states_to_refine, P̌_old, P̂_old; refinement_dirname=nothing)
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
        new_state_dict = Dict() # Mapping to help update transition probability intervals
        old_state_dict = Dict()
        new_state_idx = num_states - num_refine_states + 1 # We now have a dictionary that can be used to go from new state to old state!! 
        for state_to_refine in states_to_refine
            new_states = uniform_refinement(all_states_SA[state_to_refine])
            push!(new_states_list, new_states...)
            old_state_dict[state_to_refine] = []
            for state in new_states
                new_state_dict[new_state_idx] = state_to_refine
                push!(old_state_dict[state_to_refine], new_state_idx) 
                new_state_idx += 1
            end
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


        #! This is a hot mess, but it works.

        # For each new state, 
        target_idxs_dict = Dict()
        for new_state_idx in sort(collect(keys(new_state_dict)))
            # target idxs to compute transition interval - zero otherwise!!
            target_idxs = []
            old_idx = new_state_dict[new_state_idx] 
            succ_states = findall(x -> x > 0., P̂_old[old_idx, :])
            for succ_state in succ_states 
                # Is the successor state a target of refinement? If so, add all those new state idxs to the transition targets
                if succ_state in states_to_refine
                    push!(target_idxs, old_state_dict[succ_state]...) 
                # Otherwise, it is an unrefined state that may have a new index. Calculate this value here.
                else
                    # Calculate the new index of the unrefined state
                    num_states_prior = sum(states_to_refine .< succ_state)
                    new_unrefined_idx = succ_state - num_states_prior
                    # @info "Old idx: $succ_state, New idx: $new_unrefined_idx, sum: $num_states_prior"
                    push!(target_idxs, new_unrefined_idx)
                end
            end
            if isempty(target_idxs)
                throw("Target index set is empty for state $new_state_idx. This should not happen.")
            end
            @assert new_state_idx ∉ keys(target_idxs_dict)
            target_idxs_dict[new_state_idx] = sort(unique(target_idxs))
        end

        # For all other states, only focus on transitions to states that were refined
        # Iterate over only the hot idxs
        hot_idxs = setdiff(1:num_states+1, states_to_refine)

        # Error is here!!!
        j_dummy = 1 # Need a dummy index to correctly assign target idx dict
        for i in hot_idxs[1:end-1]
            target_idxs = []

            # Find all possible transitions according to old matrix 
            succ_states = findall(x -> x>0., P̂_old[i, :]) # yes, got all the successor states

            for state in succ_states 
                if state in states_to_refine
                    push!(target_idxs, old_state_dict[state]...) 
                end
                # Ignore the unrefined states, because we set them later with hot indeces
            end
            @assert j_dummy ∉ keys(target_idxs_dict)
            target_idxs_dict[j_dummy] = sort(unique(target_idxs)) 
            j_dummy += 1
        end

        @assert length(keys(target_idxs_dict)) == length(all_states_refined)
        # Hot includes the unsafe state in the final rows and cols
        P̌_hot = P̌_old[hot_idxs, hot_idxs] 
        P̂_hot = P̂_old[hot_idxs, hot_idxs] 

        P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_refined, all_state_images_refined, LearningAbstractions.extent_to_SA(X_extent), gp_rkhs_info=gp_info, σ_bounds_all=all_σ_bounds_refined, P̌_hot=P̌_hot, P̂_hot=P̂_hot, target_idxs_dict=target_idxs_dict)

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

function find_states_to_refine(P̂, res_mat; p_threshold=0.95, refine_targets=false, refine_unsafe=false)

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

    if refine_unsafe
        for n in n_no
            poss_states = findall(x -> x>0., P̂[:,n]) 
            setdiff!(poss_states, n_yes)
            setdiff!(poss_states, n_no)
            setdiff!(poss_states, states_to_refine)
            union!(states_to_refine, poss_states) 
        end
    end

    if refine_targets
        for poss_state in states_to_refine 
            poss_targets = findall(x -> x > 0.0, P̂[poss_state, :])
            setdiff!(poss_targets, n_yes)
            setdiff!(poss_targets, n_no)
            setdiff!(poss_targets, states_to_refine)
            union!(states_to_refine, poss_targets)
        end

    end

    return sort(states_to_refine)
end