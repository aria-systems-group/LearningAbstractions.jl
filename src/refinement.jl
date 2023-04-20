"""
    uniform_refinement

Refine a discrete state uniformly and update the region info dict.
"""
function uniform_refinement(state)
    deltas = (state[:,end-1] - state[:,1])/2
    refined_grid, refined_deltas = grid_generator(state[:,1], state[:,end-1], deltas)
    new_states = [lower_to_SA(gl, refined_deltas) for gl in refined_grid]
    return new_states
end

"""
    dimension_refinement

Refine a discrete state along the specified dimension.
"""
function dimension_refinement(state, idx)
    # target a certain dimension for refinement
    deltas = Vector((state[:,end-1] - state[:,1]))
    deltas[idx] *= 0.5
    refined_grid, refined_deltas = grid_generator(state[:,1], state[:,end-1], deltas)
    new_states = [lower_to_SA(gl, refined_deltas) for gl in refined_grid]
    return new_states
end

function sek(x, y, σ2=1.0, ℓ2=1.0)
    return σ2*exp(-(x-y)'*(x-y)/(2*ℓ2))
end

function dμdx(x, gp)
    x_in = gp.x
    α = gp.alpha
    d = sum([[(x - xp)*sek(x,xp)*a] for (xp, a) in zip(eachcol(x_in), α)])
    return abs.(d...)
    # return d
end

function max_derivative_dim(x, gp)
    partials = dμdx(x,gp)
    max_val, idx = findmax(partials)
    return max_val, idx
end

"""
    gp_derivative_refinement

Refinement based on the GP derivative.
"""
function gp_derivative_refinement(state, image, gps)
    state_mean = state_means([state])[1]
    # @info image[:,end-1] - image[:,1] 
    res, max_image_idx = findmax(image[:,end-1] - image[:,1])
    # @info res, max_image_idx

    state_dels = state[:,end-1] - state[:,1]

    #! Create Local GP Here?
    #! Or just base on global GPs eh?
    # _, max_idx = max_derivative_dim(state_mean, gps[max_image_idx])
    partials = dμdx(state_mean, gps[max_image_idx])
    # @info partials
    weighted = (state_dels .* partials) / sum(state_dels)
    # @info state_dels
    # @info weighted
    _, max_idx = findmax(weighted)
    new_states = dimension_refinement(state, max_idx)
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
	# System config parsing
	distance_metric, lipschitz_bound = parse_system_params(config["system"])

    # TODO: Put this stuff in the directory generator?
    results_dir = config["results_directory"]
	base_results_dir = "$results_dir/base"
	gps_filename = "$base_results_dir/gps.bson"
    refinement_dir = isnothing(refinement_dirname) ? "$results_dir/refined" : "$results_dir/$refinement_dirname"
    !isdir(refinement_dir) && mkpath(refinement_dir)
    state_refined_filename = "$refinement_dir/states.bson"
	imdp_refined_filename = "$refinement_dir/imdp.bson"

    all_states_refined, all_state_images_refined, all_σ_bounds_refined, P̌, P̂ = load_results(state_refined_filename, imdp_refined_filename, reuse_states=config["reuse_states"], reuse_results=config["reuse_results"])
	reloaded_states_flag = !isnothing(all_states_refined)
	reloaded_results_flag = !isnothing(P̌)

    refinement_procedure = config_entry_try(config["workspace"], "refinement_procedure", "uniform") 

    if !reloaded_results_flag
		# Load the existing global GPs
        @info "Loading the GPs for refinement"
        gps, gp_info_dict = load_gps(gps_filename)
        gp_info = GPRelatedInformation(gp_info_dict["γ_bounds"],
                        gp_info_dict["RKHS_norm_bounds"],
                        gp_info_dict["logNoise"],
                        gp_info_dict["post_scale_factors"],
                        gp_info_dict["Kinv"],
                        gp_info_dict["f_sup"],
                        gp_info_dict["measurement_noise"],
                        gp_info_dict["process_noise"])
	end

    # Local GP setup
    local_gps_flag = config["local"]["use_local_gps"]
    local_gps_nns = config["local"]["local_gp_neighbors"]
    delta_input_flag = config_entry_try(config["system"], "delta_input_flag", false) # flag to indicate training on the delta from each point, i.e.

    if local_gps_flag
        @info "Performing local GP regression with $local_gps_nns-nearest neighbors"
        # Reload the data here
        input_data, output_data, _, delta_input_flag, _, _ = parse_data_params(config["system"], load_data=true)
        local_gps_data = (input_data, output_data)
        # TODO: Generalize this!
        local_gp_metadata = [ones(length(lipschitz_bound)), 0.65*ones(length(lipschitz_bound)), local_gps_nns]
    else
        local_gps_data = nothing
        local_gp_metadata = nothing
    end

    if !reloaded_results_flag 
        @info "Parsing metadata and identifying indexes for refinement"
        # Parse the config for necessary info
        _, _, X_extent, _, _ = parse_discretization_params(config["workspace"])	

        num_refine_states = length(states_to_refine)
        frac = length(states_to_refine)/num_states
        @info "Refining $num_refine_states of $num_states ($frac) states with $refinement_procedure procedure"

        # Generate new discretization
        new_states_list = []
        new_state_dict = Dict() # Mapping to help update transition probability intervals
        old_state_dict = Dict()
        new_state_idx = num_states - num_refine_states + 1 # We now have a dictionary that can be used to go from new state to old state!! 
        # refinement_procedure = "uniform"
        for state_to_refine in states_to_refine
            if refinement_procedure == "uniform"
                new_states = uniform_refinement(all_states_SA[state_to_refine])
            elseif refinement_procedure == "gp_derivative"
                new_states = gp_derivative_refinement(all_states_SA[state_to_refine], all_state_images[state_to_refine], gps)
            end
            push!(new_states_list, new_states...)
            old_state_dict[state_to_refine] = []
            for state in new_states
                new_state_dict[new_state_idx] = state_to_refine
                push!(old_state_dict[state_to_refine], new_state_idx) 
                new_state_idx += 1
            end
        end

        target_idxs_dict = Dict()
        # > Fcn: Smart refinement, e.g. start with only possible
        smart_refine_flag = false
        # ! TODO THIS DOES NOT WORK
        # if smart_refine_flag
        #     for new_state_idx in keys(new_state_dict)
        #         # target idxs to compute transition interval - zero otherwise!!
        #         target_idxs = [] # Good
        #         old_idx = new_state_dict[new_state_idx] # Good
        #         succ_states = findall(x -> x>0., P̂_old[old_idx, 1:end-1]) ∩ findall(x -> x>0., P̂_old[old_idx, 1:end-1] - P̌_old[old_idx, 1:end-1]) #! Added a check for interval width, should not make a difference

        #         for succ_state in succ_states 
        #             # Is the successor state a target of refinement? If so, add all those new state idxs to the transition targets
        #             if succ_state in states_to_refine
        #                 push!(target_idxs, old_state_dict[succ_state]...) # Good 
        #             # Otherwise, it is an unrefined state that may have a new index. Calculate this value here.
        #             else
        #                 # Calculate the new index of the unrefined state
        #                 num_states_prior = sum(states_to_refine .< succ_state) # Good
        #                 new_unrefined_idx = succ_state - num_states_prior # Good
        #                 # @info "Old idx: $succ_state, New idx: $new_unrefined_idx, sum: $num_states_prior"
        #                 push!(target_idxs, new_unrefined_idx) # Good
        #             end
        #         end
        #         if isempty(target_idxs)
        #             throw("Target index set is empty for state $new_state_idx. This should not happen.")
        #         end
        #         @assert new_state_idx ∉ keys(target_idxs_dict)
        #         target_idxs_dict[new_state_idx] = sort(unique(target_idxs))
        #     end
        # end

        # For all other states, only focus on transitions to states that were refined
        # Iterate over only the hot idxs
        hot_idxs = get_hot_idxs(num_states+1, states_to_refine)

        j_dummy = 1 # Need a dummy index to correctly assign target idx dict
        for i in hot_idxs[1:end-1]
            target_idxs = []

            # Find all possible transitions according to old matrix 
            succ_states = post(i, P̂_old) # yes, got all the successor states

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
	end
        
    if !reloaded_states_flag
        @info "Finding the images of the refined states"
        # delete the old states
        all_states_refined = copy(all_states_SA)
        all_state_images_refined = copy(all_state_images)
        all_σ_bounds_refined = copy(all_σ_bounds)

        deleteat!(all_states_refined, states_to_refine)
        push!(all_states_refined, new_states_list...)

        # delete old image bounds
        deleteat!(all_state_images_refined, states_to_refine)
        deleteat!(all_σ_bounds_refined, states_to_refine)

        new_images, new_σ_bounds = state_bounds(new_states_list, gps; local_gps_flag=local_gps_flag, local_gps_data=local_gps_data, local_gps_nns=local_gps_nns, metric=distance_metric, delta_input_flag=delta_input_flag)
        all_state_images_refined = vcat(all_state_images_refined, new_images)
        all_σ_bounds_refined = vcat(all_σ_bounds_refined, new_σ_bounds)
    end

    if !reloaded_results_flag
        # Hot includes the unsafe state in the final rows and cols
        P̌_hot = P̌_old[hot_idxs, hot_idxs] 
        P̂_hot = P̂_old[hot_idxs, hot_idxs] 

        # TODO: Add process noise here!
        P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_refined, all_state_images_refined, LearningAbstractions.extent_to_SA(X_extent), gp_rkhs_info=gp_info, σ_bounds_all=all_σ_bounds_refined, P̌_hot=P̌_hot, P̂_hot=P̂_hot, target_idxs_dict=target_idxs_dict, local_gp_metadata=local_gp_metadata)
    end

    if config["save_results"] && !reloaded_results_flag
        @info "Saving abstraction info to $refinement_dir"
            if !reloaded_states_flag && !reloaded_results_flag
            bson(state_refined_filename, Dict(:states => all_states_refined,
                                :images => all_state_images_refined,
                                :bounds => all_σ_bounds_refined,
                                )
            )
        end
        
        if !reloaded_results_flag
            bson(imdp_refined_filename, Dict(:Pcheck => P̌, :Phat => P̂))
        end
    end
    return P̌, P̂, all_states_refined, refinement_dir, all_state_images_refined, all_σ_bounds_refined
end

function get_hot_idxs(num_states, states_to_refine)
    return setdiff(1:num_states, states_to_refine)
end

function find_states_to_refine(P̂, res_mat, all_states; p_threshold=0.95, refine_targets=false, refine_unsafe=false, diameter_threshold=0.0)

    n_yes = findall(x -> x>=p_threshold, res_mat[:,3])
    n_no = findall(x -> x<p_threshold, res_mat[1:end,4])

    if isempty(n_yes)
        # if nothing is yes, such as in safety, return all states for refinement
        return setdiff(Int.(res_mat[1:end-1,1]), n_no)
    end

    # now n_yes consists of states that satisfy the spec
    # the idea is to only refine states that may possibly satisfy and can transition to this set
    states_to_refine = []
    
    for n in n_yes
        poss_states = pre(n, P̂)
        setdiff!(poss_states, n_yes)
        setdiff!(poss_states, n_no)
        setdiff!(poss_states, states_to_refine)
        union!(states_to_refine, poss_states) 
    end

    if refine_unsafe
        for n in n_no
            poss_states = pre(n, P̂)
            setdiff!(poss_states, n_yes)
            setdiff!(poss_states, n_no)
            setdiff!(poss_states, states_to_refine)
            union!(states_to_refine, poss_states) 
        end
    end

    if refine_targets
        for poss_state in states_to_refine 
            poss_targets = pre(poss_state, P̂)
            setdiff!(poss_targets, n_yes)
            setdiff!(poss_targets, n_no)
            setdiff!(poss_targets, states_to_refine)
            union!(states_to_refine, poss_targets)
        end
    end

    if diameter_threshold > 0.0
        state_diameters = [sqrt(sum((all_states[i][:,1]-all_states[i][:,end-1]).^2)) for i in states_to_refine]
        filter_idx = findall(x -> x < diameter_threshold, state_diameters)
        nf = length(filter_idx)
        @info "Skipping the refinement of $nf states" 
        deleteat!(states_to_refine, filter_idx)
    end

    return sort(states_to_refine)
end