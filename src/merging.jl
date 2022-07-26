# Methods related to merging states and reducing the abstraction

"""
    merge_check

Checks whether the two hyperrectangles can be merged by checking for a shared face.
"""
function merge_check(s1, s2)
    n = size(s1, 1)
    f1, f2 = 0, 0
    for i=1:n
        axis = zeros(n,1)
        axis[i] = 1
        c1 = s1'*axis
        c2 = s2'*axis
        if 0.0 ∈ c1 - c2[end:-1:1]
            f1 += 1
        elseif iszero(c1-c2)
            f2 += 1
        end
    end

    if (f1 == 1 && f2 == n-1) 
        @debug s1, s2
        @debug "Shapes can be merged!" 
    end
    return (f1 == 1 && f2 == n-1) 
end

"""
    merge_regions

Merges two hyperrectangles.
"""
function merge_regions(s1, s2)
    n = size(s1, 1)
    lower = vcat([minimum([s1[i,1], s2[i,1]]) for i=1:n]...)
    upper = vcat([maximum([s1[i,2], s2[i,2]]) for i=1:n]...)
    return [lower upper]
end

"""
    iterative_merge

Iteratively try merging to maximize the merging.
"""
function iterative_merge(merge_idxs, state_array)
    num_merges = Inf

    merge_dict = Dict()
    merged_states = []
    while num_merges > 0
        idxs_merged, idxs_merge_pairs = simple_merge(merge_idxs, state_array) 
        num_merges = length(idxs_merge_pairs)
        for pair in idxs_merge_pairs
            s1 = [state_array[pair[1]][:,1] state_array[pair[1]][:,end-1]]
            s2 = [state_array[pair[2]][:,1] state_array[pair[2]][:,end-1]]
            new_state_ex = merge_regions(s1, s2)
            # TODO: Not saving all of the state def, only extrema 
            new_state = zeros(size(s1))
            new_state[:,1] = new_state_ex[:,1]
            new_state[:,end-1] = new_state_ex[:,2]

            # The merged state will always take the minimum idx!
            key_idx = minimum(pair)
            part_idx = maximum(pair)

            # Save the new state
            state_array[key_idx] = new_state
            if key_idx ∈ keys(merge_dict) 
                if part_idx ∈ keys(merge_dict)
                    push!(merge_dict[key_idx], [part_idx, merge_dict[part_idx]...])
                else
                    push!(merge_dict[key_idx], part_idx)
                end
            else
                if part_idx ∈ keys(merge_dict)
                    merge_dict[key_idx] = [part_idx, merge_dict[part_idx]...] 
                    delete!(merge_dict, part_idx)
                else
                    merge_dict[key_idx] = [part_idx] 
                end
            end
            push!(merged_states, part_idx) # This index is now gone! We don't need to consider it anymore. 
            deleteat!(merge_idxs, merge_idxs .== maximum_pair) # Delete this index from consideration
            # Keep the merged state in the pool to do more refinement.
        end
    end

    new_state_array = deleteat(state_array, merged_states)
    return merge_dict, new_state_array 
end

function simple_merge(merge_idxs, state_array)

    candidate_merge_idxs = copy(merge_idxs)

    # get state means
    state_means = LearningAbstractions.state_means(state_array)

    # build a KD tree
    mean_tree = KDTree(state_means)
    nns = 2*size(state_array[1], 1) # max number of neighbors is fcn of dimension

    idxs_merged = [] 
    idxs_merge_pairs = []
    for i in candidate_merge_idxs
        if i ∈ idxs_merged
            continue
        end
        # possible neighbors
        candidate_idxs, _ = knn(mean_tree, state_means[i], nns, false)
        candidate_idxs = candidate_idxs ∩ candidate_merge_idxs # remove non-target states

        for idx in candidate_idxs
            if idx == i || idx ∈ idxs_merged || i ∈ idxs_merged
                continue 
            end
            s1 = [state_array[i][:,1] state_array[i][:,end-1]]
            s2 = [state_array[idx][:,1] state_array[idx][:,end-1]]
            merge_flag = merge_check(s1, s2)
            if merge_flag
                # idxs_merge_dict[i] = idx
                push!(idxs_merge_pairs, (i, idx))
                push!(idxs_merged, i)
                push!(idxs_merged, idx)
            end
        end
    end

    sort!(idxs_merged)

    return idxs_merged, idxs_merge_pairs
end

"""
    merge_transitions

Reduces the transition interval matrices based on states that have been merged.
"""
function merge_transitions(merged_idxs_dict, P̌, P̂)
    # merged idxs is list of all idxs merged together

    num_merge_clusters = merged_idxs_dict.count
    merged_idxs = []
    key_offset_dict = Dict()
    for k in keys(merged_idxs_dict)
        idxs = merged_idxs_dict[k]
        num_idxs += length(idxs)
        merged_idxs = merged_idxs ∪ idxs

        # calculate offset
        num_lt = 0
        for l in keys(merged_idxs_dict)
            idxs2 = merged_idxs_dict[l]
            num_lt += sum(idxs2 .< k)
        end
        @assert 0 <= num_lt < k
        key_offset_dict[k] = k - num_lt
    end

    new_state_num = size(P̌, 2) - num_idxs + num_merge_clusters # all states merged into few 

    @info "$num_idxs states merged into $num_merge_clusters clusters!"

    P̌_merge = spzeros(new_state_num, new_state_num) 
    P̂_merge = spzeros(new_state_num, new_state_num)

    hot_idxs = collect(1:size(P̌, 2))
    deleteat!(hot_idxs, merged_idxs)
    P̌_merge[1:end, 1:end] = P̌[hot_idxs, hot_idxs]
    P̂_merge[1:end, 1:end] = P̂[hot_idxs, hot_idxs]

    # ! TODO This can be done not in a for loop I believe
    for key_idx in keys(merged_idxs_dict)
        offset_key = key_offset_dict[key_idx]
        target_idxs = merged_idxs_dict[key_idx]
        for i=1:new_state_num-1 # ! ignore the last state
            P̌_merge[i, offset_key] = minimum(P̌[hot_idxs[i], target_idxs ∪ [key_idx]])
            P̂_merge[i, offset_key] = maximum(P̂[hot_idxs[i], target_idxs ∪ [key_idx]])
        end
    end
    # TODO: Double check that the merged index is acting correctly

    return P̌_merge, P̂_merge 
end