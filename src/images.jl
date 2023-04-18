# Functions to find the images of discrete states

# Contains preallocated arrays that are used a lot in intermediate calculations in image bounding

# Structure for local GP components
mutable struct LocalGP
    sub_idxs::Vector{Int}
    knn_dists::Vector{Float64}
    gps::Vector{PosteriorGP}
end

"""
Compute bounds on the image and covariance of a vector of states through a GP map.
"""
function state_bounds(states_vec, gps; local_gps_flag=false, local_gps_data=nothing, local_gps_nns=nothing, metric=Euclidean(), delta_input_flag=false, approximate_σ_flag=false)
 
    odims = length(gps)
    type = SMatrix{odims, 2^odims, Float64, odims*2^odims}
    image_vec = Vector{type}(undef, length(states_vec))
    σ_bounds_vec = Vector{Vector{Real}}(undef, length(states_vec))

    # Setup for bounding refined states with local GP regression
    if local_gps_flag
        tree = create_data_tree(local_gps_data[1], metric) 
        local_neg_gps = [nothing for _ in 1:length(gps)]
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

        # Preallocate nthreads*ngps local GP objects
        local_gps = [LocalGP(
                            Vector{Int}(undef, gpnobs),
                            Vector{Float64}(undef, gpnobs),
                            [PosteriorBounds.PosteriorGP(
                                gpdim,
                                gpnobs,
                                gp_x[:,1:gpnobs], 
                                Matrix{Float64}(undef, gpnobs, gpnobs),
                                Matrix{Float64}(undef, gpnobs, gpnobs),
                                UpperTriangular(zeros(gpnobs, gpnobs)),
                                Symmetric(Matrix{Float64}(undef, gpnobs, gpnobs)),
                                Vector{Float64}(undef, gpnobs), 
                                SEKernel(gps[i].kernel.σ2, gps[i].kernel.ℓ2),   # TODO: Generalize for any kernel
                            ) for i in eachindex(gps)]
                        ) for _ in 1:nthreads]       
    else
        gpnobs = gps[1].nobs
        gp_x = gps[1].x
        
        # create Posterior GPs from GaussianProcesses GPs
        post_gps = [PosteriorBounds.PosteriorGP(
            gpdim,
            gpnobs,
            gp_x, 
            gps[i].cK,
            Matrix{Float64}(undef, gpnobs, gpnobs),
            UpperTriangular(zeros(gpnobs, gpnobs)),
            Matrix(inv(deepcopy(gps[i].cK))),
            gps[i].alpha, 
            SEKernel(gps[i].kernel.σ2, gps[i].kernel.ℓ2),   # TODO: Generalize for any kernel
        ) for i in eachindex(gps)]
        [PosteriorBounds.compute_factors!(gp) for gp in post_gps]
    end

    # Preallocated arrays for memory savings 
    image_preallocs = [PosteriorBounds.preallocate_matrices(gpdim, gpnobs) for _ in 1:nthreads]

    # TODO: Create using the function in PB
    gp_x_sq = gp_x.^2 
    theta_vec_train_squared_all = [zeros(size(gp_x_sq, 2)) for _ in 1:odims] 
    theta_vec_all = [(@SVector ones(size(gp_x_sq, 1))) * 1 ./ (2*gps[i].kernel.ℓ2) for i=1:odims] # All have same params, assume for now
    
    for i = 1:odims
        theta_vec_T = transpose(theta_vec_all[i])
        for j in axes(gp_x_sq, 2)
            @views theta_vec_train_squared_all[i][j] = (theta_vec_T * (gp_x_sq[:, j]))[1]
        end
    end

    all_state_means = state_means(states_vec)

    Threads.@threads for (idx, state) in collect(enumerate(states_vec)) 
        tid = Threads.threadid()
        image_prealloc = image_preallocs[tid]

        # If local GP, create local GP here
        if local_gps_flag
            create_local_gps!(local_gps[tid], gps, all_state_means[idx], tree, gp_x, local_gps_data[2])
            @views theta_vec_train_sq_sub = [theta_vec_train_squared_all[i][local_gps[tid].sub_idxs] for i=1:odims]
            image, σ_bounds = bound_image([state[:,1], state[:,end-1]], local_gps[tid].gps, local_neg_gps, theta_vec_train_sq_sub, theta_vec_all, image_prealloc, delta_input_flag=delta_input_flag, approximate_σ_flag=approximate_σ_flag) 
        else
            image, σ_bounds = bound_image([state[:,1], state[:,end-1]], post_gps, neg_gps, theta_vec_train_squared_all, theta_vec_all, image_prealloc, delta_input_flag=delta_input_flag, approximate_σ_flag=approximate_σ_flag)
        end
        image_vec[idx] = extent_to_SA(image)
        σ_bounds_vec[idx] = σ_bounds
        next!(p)
    end

    return image_vec, σ_bounds_vec
end

""" 
    bound_image

Generate overapproximations of posterior mean and covariance functions using one of several methods.
# Arguments
- `extent` - Discrete state extent 
- `gps` - Vector of GPs and associated metadata
- `neg_gps` - Vector of GPs with -1*α vector
- `delta_input_flag` - True uses `x` as the known component 
"""
function bound_image(extent, gps, neg_gps, theta_vec_train_squared, theta_vec, image_prealloc; delta_input_flag=false, data_deps=nothing, known_component=nothing, σ_ubs=nothing, approximate_σ_flag=false)
    # TODO: mod keyword handling and document
    ndims = length(gps) 
    image_extent = Vector{Vector{Float64}}(undef, ndims)
    σ_bounds = zeros(ndims) 
    for i=1:ndims 
        image_extent[i], σ_bounds[i] = bound_extent_dim(gps[i], neg_gps[i], extent[1], extent[2], theta_vec_train_squared[i], theta_vec[i], image_prealloc, approximate_σ_flag=approximate_σ_flag)
        if delta_input_flag
            image_extent[i][1] += extent[1][i]
            image_extent[i][2] += extent[2][i]
        end
    end
    return image_extent, σ_bounds
end

function bound_extent_dim(gp, neg_gp, lbf, ubf, theta_vec_train_squared, theta_vec, image_prealloc; approximate_μ_flag=false, approximate_σ_flag=true)

    if approximate_μ_flag
        μ_L_lb, μ_U_ub = PosteriorBounds.compute_μ_bounds_approx(gp, lbf, ubf) 
    else
        _, μ_L_lb, _ = PosteriorBounds.compute_μ_bounds_bnb(gp, lbf, ubf, theta_vec_train_squared, theta_vec; prealloc=image_prealloc) 
        if typeof(gp) == LocalGP
            _, _, μ_U_ub = PosteriorBounds.compute_μ_bounds_bnb(gp, lbf, ubf, theta_vec_train_squared, theta_vec; prealloc=image_prealloc, max_flag=true)
        else
            _, _, μ_U_ub = PosteriorBounds.compute_μ_bounds_bnb(gp, lbf, ubf, theta_vec_train_squared, theta_vec; prealloc=image_prealloc, max_flag=true)
        end
    end

    if approximate_σ_flag
        _, σ_U_lb, σ_U_ub = PosteriorBounds.compute_σ_ub_bounds_approx(gp, lbf, ubf) 
    else
        _, σ_U_lb, σ_U_ub = PosteriorBounds.compute_σ_bounds(gp, lbf, ubf, theta_vec_train_squared, theta_vec, gp.K_inv, prealloc=image_prealloc)
    end

    if !(μ_L_lb <= μ_U_ub)
        @error "μ LB is greater than μ UB! ", μ_L_lb, μ_U_ub
        throw("aborting") 
        @assert μ_L_lb <= μ_U_ub
    end
    if !(σ_U_lb <= σ_U_ub)
        @error "Sigma LB is greater than sigma UB! ", σ_U_lb, σ_U_ub
        throw("aborting")
        @assert σ_U_lb <= σ_U_ub
    end 
    image_extent = SA_F64[μ_L_lb, μ_U_ub]
    return image_extent, σ_U_ub 
end

"""
    state_means

    Calculate the mean point for all states in a vec
"""
function state_means(state_vec)
    return [0.5*(s[:,1] + s[:,end-1]) for s in state_vec]
end