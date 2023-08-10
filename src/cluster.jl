# Tool related to clustering and implicit merging

"""
    find_intersecting_states
"""
function find_intersecting_states(image, states)
    intersecting_states = []
    image_ex = LearningAbstractions.SA_to_extent(image)
    for (idx, state) in enumerate(states)
        if LearningAbstractions.intersects(image_ex, LearningAbstractions.SA_to_extent(state)) 
            push!(intersecting_states, idx)
        end
    end
    return intersecting_states
end

"""
    build_super_state
"""
function build_super_state(states)
    # find min extents
    dims = size(states[1], 1)
    exs = []
    for dim = 1:dims
        minv = Inf
        maxv = -Inf
        for state in states
            row = state[dim, :];
            minv = min(minimum(row), minv)
            maxv = max(maximum(row), maxv)
        end
        push!(exs, [minv, maxv])
    end
    return LearningAbstractions.extent_to_SA(exs) 
end

"""
    calculate_implicit_plow

Calculate the new L.B. probability of satisfaction for a certain state with implicit clustering of successor states 
"""
function calculate_implicit_plow(P̌_row, P̂_row, image, states, prior_results::AbstractMatrix, gp_rkhs_info, σ_bounds, local_RKHS_bound; process_noise_dist=nothing, η_manual=0.0)

    # Create Q̃ and compute the transition interval from q
    intersecting_state_idxs = find_intersecting_states(image, states) 
    cluster_state = build_super_state(states[intersecting_state_idxs])
    P̌_new, P̂_new = LearningAbstractions.transition_inverval(image, cluster_state, nothing, process_noise_dist=process_noise_dist, gp_rkhs_info=gp_rkhs_info, σ_bounds=σ_bounds, local_RKHS_bound=local_RKHS_bound, local_gp_metadata=nothing, η_manual=η_manual)

    # Find all the states in Q^* 
    all_succ_states_star = setdiff(findall(x->x>0, P̂_row), intersecting_state_idxs)
    all_P̌ = Array(P̌_row[all_succ_states_star])
    all_P̂ = Array(P̂_row[all_succ_states_star])
    all_succ_res = prior_results[all_succ_states_star, 3]
    push!(all_P̌, P̌_new)

    # trim from P̂, then add the new one! 
    for i in eachindex(all_P̂)
        if all_P̂[i] > 1 - P̌_new 
            all_P̂[i] = 1 - P̌_new
        end
    end
    push!(all_P̂, P̂_new) 

    # Calculate the new results
    p̌_cluster = minimum(prior_results[intersecting_state_idxs, 3])
    all_succ_res = [all_succ_res;  p̌_cluster]
    idx_perm = sortperm(all_succ_res)
    p_true = LearningAbstractions.true_transition_propabilities(all_P̌, all_P̂, idx_perm)
    p̌_new = sum(p_true .* all_succ_res[idx_perm]) 
    return p̌_new
end