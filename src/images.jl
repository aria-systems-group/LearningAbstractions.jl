# Functions to find the images of discrete states

# Contains preallocated arrays that are used a lot in intermediate calculations in image bounding
struct ImageBoundPreallocation
    dx_L::Vector{Float64} 
    dx_U::Vector{Float64} 
    H::Vector{Float64} 
    f::Matrix{Float64} 
    x_star_h::Vector{Float64} 
    vec_h::Vector{Float64} 
    b_i_vec::Vector{Float64} 
    bi_x_h::Matrix{Float64} 
    α_h::Vector{Float64} 
    K_h::Matrix{Float64} 
    mu_h::Matrix{Float64} 
    Kcross::Matrix{Float64}
    Kpred::Matrix{Float64}
end

function state_bounds(states_vec, gps; local_gps_flag=false, local_gps_data=nothing, local_gps_nns=nothing, domain_type="", delta_input_flag=false)
 
    odims = length(gps)
    type = SMatrix{odims, 2^odims, Float64, odims*2^odims}
    image_vec = Vector{type}(undef, length(states_vec))
    σ_bounds_vec = Vector{Vector{Real}}(undef, length(states_vec))

    # Setup for bounding refined states with local GP regression
    if local_gps_flag
        tree = create_data_tree(local_gps_data[1], domain_type) 
    else
        # Setup all the global GP stuff
        neg_gps = []
        for gp in gps
            neg_gp = deepcopy(gp)
            neg_gp.alpha *= -1
            push!(neg_gps, neg_gp)
        end
    end

    # Generate new posterior bounds
    p = Progress(length(states_vec), desc="Computing image bounds...", dt=status_bar_period)
    nthreads = Threads.nthreads();

    gpdim = gps[1].dim
    # All GP components have same input
    if local_gps_flag
        gpnobs = local_gps_nns
        gp_x = local_gps_data[1]
    else
        gpnobs = gps[1].nobs
        gp_x = gps[1].x
    end

    @timeit "Preallocations" begin
    # Preallocated arrays for memory savings 
    image_preallocs = [
        ImageBoundPreallocation(
            Array{Float64}(undef, gpdim),
            Array{Float64}(undef, gpdim),
            Array{Float64}(undef, gpdim),
            Array{Float64}(undef, (1, gpdim)),
            Array{Float64}(undef, gpdim),
            Array{Float64}(undef, 2),
            Array{Float64}(undef, gpnobs),
            Array{Float64}(undef, (1,gpdim)),
            Array{Float64}(undef, gpnobs),
            Array{Float64}(undef, (gpnobs,1)),
            Array{Float64}(undef, (1,1)),
            Array{Float64}(undef, (gpnobs, 1)), 
            Array{Float64}(undef, (1, 1))
        )
        for _ in 1:nthreads
    ]

    # image_preallocs=[nothing for _ in 1:nthreads] 

    gp_x_sq = gp_x.^2 
    theta_vec_train_squared = zeros(size(gp_x_sq, 2))
    # ! TODO HANDLE PARAMS
    theta_vec = (@SVector ones(size(gp_x_sq, 1))) * 1 ./ (2*gps[1].kernel.ℓ2) # All have same params, assume for now
    theta_vec_T = transpose(theta_vec)
    for i in axes(gp_x_sq, 2)
        @views theta_vec_train_squared[i] = (theta_vec_T * (gp_x_sq[:, i]))[1]
    end   
    end
    Threads.@threads for (idx, state) in collect(enumerate(states_vec)) 
        tid = Threads.threadid()
        image_prealloc = image_preallocs[tid]

        # If local GP, create local GP here
        if local_gps_flag
            state_mean = 0.5*(state[:,1] + state[:,end-1])
            local_gps, sub_idx = create_local_gps(local_gps_data[1], local_gps_data[2], state_mean, num_neighbors=local_gps_nns, tree=tree)
            @views theta_vec_train_sq_sub = theta_vec_train_squared[sub_idx] 
            local_neg_gps = []
            for gp in local_gps
                neg_gp = deepcopy(gp)
                neg_gp.alpha *= -1
                push!(local_neg_gps, neg_gp)
            end
            image, σ_bounds = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], local_gps, local_neg_gps, theta_vec_train_sq_sub, theta_vec, image_prealloc, delta_input_flag=delta_input_flag) 
        
        else
            image, σ_bounds = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], gps, neg_gps, theta_vec_train_squared, theta_vec, image_prealloc, delta_input_flag=delta_input_flag)
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