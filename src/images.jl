# Functions to find the images of discrete states

function state_bounds(states_vec, gps; local_gps_flag=false, local_gps_data=nothing, local_gps_nns=nothing, domain_type="", delta_input_flag=false)
 
    odims = length(gps)
    type = SMatrix{odims, 2^odims, Float64, odims*2^odims}
    image_vec = Vector{type}(undef, length(states_vec))
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
        tree = create_data_tree(local_gps_data[1], domain_type) 
    end

    # Generate new posterior bounds
    p = Progress(length(states_vec), desc="Computing image bounds...", dt=status_bar_period)
    Threads.@threads for (idx, state) in collect(enumerate(states_vec)) 
        # If local GP, create local GP here
        if local_gps_flag
            state_mean = 0.5*(state[:,1] + state[:,end-1])
            local_gps = create_local_gps(local_gps_data[1], local_gps_data[2], state_mean, num_neighbors=local_gps_nns, tree=tree)
            local_neg_gps = []
            for gp in local_gps
                neg_gp = deepcopy(gp)
                neg_gp.alpha *= -1
                push!(local_neg_gps, neg_gp)
            end
            image, σ_bounds = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], local_gps, local_neg_gps, delta_input_flag=delta_input_flag) 
        
        else
            image, σ_bounds = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], gps, neg_gps, delta_input_flag=delta_input_flag)
        end
        image_vec[idx] = extent_to_SA(image)
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