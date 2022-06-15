# Functions to find the images of discrete states

function calculate_state_bounds(states_vec, gps; local_gps_flag=false, local_gps_data=nothing, local_gps_nns=nothing)
 
    image_vec = Vector{typeof(states_vec[1])}(undef, length(states_vec))
    σ_bounds_vec = Vector{Vector{Real}}(undef, length(states_vec))

    # Setup all the GP stuff
    neg_gps = []
    for gp in gps
        neg_gp = deepcopy(gp)
        neg_gp.alpha *= -1
        push!(neg_gps, neg_gp)
    end

    # Setup for bounding refined states with local GP regression
    if local_gps_flag
        kdtree = KDTree(local_gps_data[1])
    end

    # Generate new posterior bounds
    p = Progress(length(states_vec), desc="Computing image bounds...", dt=30)
    Threads.@threads for (idx, state) in collect(enumerate(states_vec)) 

        # current_idx = new_idx_start + i
        # If local GP, create local GP here
        if local_gps_flag
            state_mean = 0.5*(state[:,1] + state[:,end-1])
            local_gps = create_local_gps(local_gps_data[1], local_gps_data[2], state_mean, num_neighbors=local_gps_nns, kdtree=kdtree)
            local_neg_gps = []
            for gp in local_gps
                neg_gp = deepcopy(gp)
                neg_gp.alpha *= -1
                push!(local_neg_gps, neg_gp)
            end
            image, σ_bounds = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], local_gps, local_neg_gps) 
        
        else
            image, σ_bounds = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], gps, neg_gps)
        end
        # push!(all_state_images_refined, extent_to_SA(image))
        image_vec[idx] = extent_to_SA(image)
        # push!(all_σ_bounds_refined, σ_bounds)
        σ_bounds_vec[idx] = σ_bounds
        next!(p)
    end

    return image_vec, σ_bounds_vec
end

"""
    state_means

    Calculate the mean point for all states in a vec
"""
function state_means(state_vec)
    return [0.5*(s[:,1] + s[:,end-1]) for s in state_vec]
end