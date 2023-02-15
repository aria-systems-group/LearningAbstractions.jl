module GPBounding

using GaussianProcesses
using LinearAlgebra
using LinearAlgebra: BlasReal, BlasFloat
using Random
using Distributions
using StaticArrays
using Tullio
include("squared_exponential.jl")

export bound_image, bound_images

using PDMats

# Structure for local GP components
mutable struct LocalGP
    sub_idxs::Vector{Int}
    x
    alpha::Vector{Float64}
    cK::Matrix{Float64}
    cKchol::Matrix{Float64} # Prealloc for cholesky factors
    cKcholut::UpperTriangular{Float64, Matrix{Float64}}
    cKinv::Symmetric{<:BlasReal} # Prealloc for cK inverse
    kernel
    knn_dists
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
function bound_image(extent, gps, neg_gps, theta_vec_train_squared, theta_vec, image_prealloc; delta_input_flag=false, data_deps=nothing, known_component=nothing, σ_ubs=nothing, σ_approx_flag=false)
    # TODO: mod keyword handling and document
    ndims = length(gps) 
    image_extent = Vector{Vector{Float64}}(undef, ndims)
    σ_bounds = zeros(ndims) 
    for i=1:ndims 
        image_extent[i], σ_bounds[i] = bound_extent_dim(gps[i], neg_gps[i], extent[1], extent[2], theta_vec_train_squared[i], theta_vec[i], image_prealloc)
        if delta_input_flag
            image_extent[i][1] += extent[1][i]
            image_extent[i][2] += extent[2][i]
        end
    end
    return image_extent, σ_bounds
end

function bound_extent_dim(gp, neg_gp, lbf, ubf, theta_vec_train_squared, theta_vec, image_prealloc; approximate_μ_flag=false, approximate_σ_flag=true)

    if approximate_μ_flag
        μ_L_lb, μ_U_ub = compute_μ_bounds_approx(gp, lbf, ubf) 
    else
        _, μ_L_lb, _ = compute_μ_bounds_bnb(gp, lbf, ubf, theta_vec_train_squared, theta_vec; image_prealloc=image_prealloc) 
        if typeof(gp) == LocalGP
            gp.alpha[:] .*= -1
            _, _, μ_U_ub = compute_μ_bounds_bnb(gp, lbf, ubf, theta_vec_train_squared, theta_vec; image_prealloc=image_prealloc, max_flag=true)
            gp.alpha[:] .*= -1
        else
            _, _, μ_U_ub = compute_μ_bounds_bnb(neg_gp, lbf, ubf, theta_vec_train_squared, theta_vec; image_prealloc=image_prealloc, max_flag=true)
        end
    end

    if approximate_σ_flag
        _, σ_U_lb, σ_U_ub = compute_σ_ub_bounds_approx(gp, lbf, ubf, preallocs=image_prealloc) 
    else
        # Kinv = inv(gp.cK.mat + exp(gp.logNoise.value)^2*I)
        _, σ_U_lb, σ_U_ub = compute_σ_ub_bounds(gp, gp.cKinv, lbf, ubf, theta_vec_train_squared, theta_vec, image_prealloc=image_prealloc)
    end
    # elseif !isnothing(σ_ubs)
    # _, σ_U_lb, σ_U_ub = compute_σ_ub_bounds_from_gp(gp, lbf, ubf)
    # else
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

function compute_μ_bounds_bnb(gp, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=100, bound_epsilon=1e-2, max_flag=false, image_prealloc=nothing)

    # If no preallocation object is provided, preallocate
    # This could be done more elegantly, but leave it for now...
    if isnothing(image_prealloc)      
        dx_L = zeros(gp.dim)
        dx_U = zeros(gp.dim)
        H = zeros(gp.dim)
        f = zeros(1, gp.dim)
        x_star_h = zeros(gp.dim)
        vec_h = zeros(2)
        bi_x_h = zeros(1,gp.dim)
        b_i_vec = Array{Float64}(undef, gp.nobs)
        α_h = zeros(gp.nobs)
        K_h = zeros(gp.nobs,1)
        mu_h = zeros(1,1)
    else
        dx_L = image_prealloc.dx_L 
        dx_U = image_prealloc.dx_U 
        H = image_prealloc.H 
        f = image_prealloc.f 
        x_star_h = image_prealloc.x_star_h 
        vec_h = image_prealloc.vec_h 
        bi_x_h = image_prealloc.bi_x_h 
        b_i_vec = image_prealloc.b_i_vec 
        α_h = image_prealloc.α_h 
        K_h = image_prealloc.K_h 
        mu_h = image_prealloc.mu_h 
    end
    
    x_best, lbest, ubest = compute_μ_lower_bound(gp, x_L, x_U, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, vec_h, bi_x_h, α_h, K_h, mu_h, upper_flag=max_flag)
    if max_flag
        temp = lbest
        lbest = -ubest
        ubest = -temp
    end
    
    candidates = [(x_L, x_U)]
    iterations = 0

    split_regions = nothing
    x_avg = zeros(length(x_L))

    while !isempty(candidates) && iterations < max_iterations
        new_candidates = []
        for extent in candidates
            
            if isnothing(split_regions)
                split_regions = split_region!(extent[1], extent[2], x_avg) 
            else
                split_regions = split_region!(extent[1], extent[2], x_avg, new_regions=split_regions)
            end  

            for pair in split_regions
                x_lb1, lb1, ub1 = compute_μ_lower_bound(gp, pair[1], pair[2], theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, vec_h, bi_x_h, α_h, K_h, mu_h, upper_flag=max_flag)
                if max_flag
                    temp = lb1
                    lb1 = -ub1
                    ub1 = -temp
                end
                
                if ub1 <= ubest
                    ubest = ub1
                    lbest = lb1
                    x_best = x_lb1
                    push!(new_candidates, pair)
                elseif lb1 < ubest   
                    push!(new_candidates, pair)
                end
                
            end
            
        end

        if norm(ubest - lbest) < bound_epsilon
            break
        end
        candidates = new_candidates
        iterations += 1
    end
    if max_flag
        temp = lbest
        lbest = -ubest
        ubest = -temp
    end

    return x_best, lbest, ubest 
end

function compute_σ_ub_bounds(gp, K_inv, x_L, x_U, theta_vec_train_squared, theta_vec; max_iterations=10, bound_epsilon=1e-4, image_prealloc=nothing)
    
    if isnothing(image_prealloc)      # Make the assumption that if one is nothing, they all are.
        dx_L = zeros(gp.dim)
        dx_U = zeros(gp.dim)
        H = zeros(gp.dim)
        f = zeros(1, gp.dim)
        x_star_h = zeros(gp.dim)
        vec_h = zeros(2)
        bi_x_h = zeros(1,gp.dim)
        b_i_vec = Array{Float64}(undef, gp.nobs)
        α_h = zeros(gp.nobs)
        K_h = zeros(gp.nobs,1)
        mu_h = zeros(1,1)
    else
        dx_L = image_prealloc.dx_L 
        dx_U = image_prealloc.dx_U 
        H = image_prealloc.H 
        f = image_prealloc.f 
        x_star_h = image_prealloc.x_star_h 
        vec_h = image_prealloc.vec_h 
        bi_x_h = image_prealloc.bi_x_h 
        b_i_vec = image_prealloc.b_i_vec 
        α_h = image_prealloc.α_h 
        K_h = image_prealloc.K_h 
        mu_h = image_prealloc.mu_h 
    end

    lbest = -Inf
    ubest = 1.
    x_best = nothing
    
    candidates = [[(x_L, x_U), lbest, ubest]]
    iterations = 0

    split_regions = nothing
    x_avg = zeros(gp.dim)
   
    while !isempty(candidates) && iterations < max_iterations
        new_candidates = []
        for candidate in candidates
            
            extent = candidate[1]
            lb_can = candidate[2]
            ub_can = candidate[3]

            if ub_can < lbest
                continue
            end
            
            round_lbest = lbest
            round_ubest = ubest

            if isnothing(split_regions)
                split_regions = split_region!(extent[1], extent[2], x_avg) 
            else
                split_regions = split_region!(extent[1], extent[2], x_avg, new_regions=split_regions)
            end  
            

            for pair in split_regions
                x_ub1, lb1, ub1 = compute_σ_upper_bound(gp, pair[1], pair[2], K_inv, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, vec_h, bi_x_h, α_h, K_h, mu_h,)
                
                if lb1 > lbest
                    lbest = lb1
                    ubest = ub1
                    x_best = x_ub1
                    push!(new_candidates, hcat([pair, lb1, ub1]))
                elseif ub1 < ubest && ub1 > lbest
                    ubest = ub1
                    push!(new_candidates, hcat([pair, lb1, ub1]))
                elseif lb1 > lb_can && ub1 < ub_can && ub1 > lbest
                    push!(new_candidates, hcat([pair, lb1, ub1]))
                end
                
                if norm(ubest - lbest) < bound_epsilon
                    @debug ubest, lbest
                    return x_best, lbest, ubest
                end
            end

        end
        
        if (length(new_candidates) == 0 && norm(ubest - lbest) > bound_epsilon)
            delta_x = 10. ^(-iterations)
            if delta_x < 1e-6
                break
            end
            x_L_n = [x_best[i] - delta_x for i=1:length(x_best)]
            x_U_n = [x_best[i] + delta_x for i=1:length(x_best)]
            new_candidates = [[(x_L_n, x_U_n), lbest, ubest]]
        end
        candidates = new_candidates
        iterations += 1
    end
    
    while norm(ubest - lbest) > bound_epsilon
        delta_x = 10. ^(-iterations)
        if delta_x < 1e-6
            break
        end
        x_L_n = [x_best[i] - delta_x for i=1:length(x_best)]
        x_U_n = [x_best[i] + delta_x for i=1:length(x_best)]
        x_best, lbest, ubest = compute_σ_upper_bound(gp, x_L_n, x_U_n, K_inv, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, vec_h, bi_x_h, α_h, K_h, mu_h,)
        iterations += 1
    end
    
    return x_best, lbest, ubest
    
end

function compute_μ_bounds_approx(gp, x_L, x_U; N=100)
    mt = MersenneTwister(11)
    # Get N samples uniformly dist.
    x_samp = vcat([rand(mt, Uniform(x_L[i], x_U[i]), 1, N) for i=1:length(x_L)]...)
    μ, _ = predict_f(gp, x_samp)
    μ_lb = minimum(μ) 
    μ_ub = maximum(μ)
    return μ_lb, μ_ub 
end

function compute_σ_ub_bounds_approx(gp, x_L, x_U; 
    preallocs=SigmaPreallocs(Matrix{Float64}(undef, gp.nobs, 1), Matrix{Float64}(undef, 1, 1)), 
    N=100, 
    twister = MersenneTwister(11), 
    x_samp_alloc = Matrix{Float64}(undef, length(x_L), N))

    max_sigma = -Inf
    for i in eachindex(x_L) 
        @views x_samp_alloc[i,:] = rand(twister, Uniform(x_L[i], x_U[i]), 1, N)
    end

    for x_col in eachcol(x_samp_alloc)
        σ2 = compute_σ_single(gp, hcat(x_col), preallocs)
        max_sigma = max(max_sigma, σ2)
    end
    return 0.0, 0.0, sqrt(max_sigma)
end

function create_x_matrix(xL, xU, N)
    dim = length(xL)
    xds = [(xU[i]-xL[i])/N for i=1:dim]
    # Assume 2D for now
    xd_prod = collect(Iterators.product(xL[1]:xds[1]:xU[1], xL[2]:xds[2]:xU[2]))
    return reshape(collect(Iterators.flatten(xd_prod)), dim, length(xd_prod))
end

function prepare_σ_gp(gp, x_L, x_U, ub; N=10, ll=-2.0)
    xn = create_x_matrix(x_L, x_U, N)
    yn = sqrt.(predict_f(gp, xn)[2])
    
    meanfcn = MeanConst(ub)
    kernel = SE(ll, 0.0)
    
    gp_σ = GP(xn, yn, meanfcn, kernel, 0.0)
    return gp_σ
end

function compute_σ_ub_bounds_from_gp(gp, x_L, x_U; ub=1.0)
    σ_gp = prepare_σ_gp(gp, x_L, x_U, ub)
    res_test = GPBounding.compute_μ_bounds_bnb(σ_gp, x_L, x_U, max_flag=true, max_iterations=4)
    return res_test[1], 0., res_test[3][1]+ub
end

function split_region!(x_L, x_U, x_avg; new_regions=nothing)
    n = length(x_L)
    x_avg .= (x_L .+ x_U)/2

    lowers = [[x_L[i], x_avg[i]] for i=1:n]
    uppers = [[x_avg[i], x_U[i]] for i=1:n]

    if isnothing(new_regions)
        new_regions = [[[lower...], [upper...]] for (lower, upper) in zip(Base.product(lowers...), Base.product(uppers...))] 
    else
        new_regions .= [[[lower...], [upper...]] for (lower, upper) in zip(Base.product(lowers...), Base.product(uppers...))]  
    end

    return new_regions
end

"""
Calculate μ using GP components assuming zero-mean prior.
"""
function predict_μ(gp, xpred, K_h, mu_h)
    xtrain = gp.x
    kernel = gp.kernel
    alpha = gp.alpha

    GaussianProcesses.cov!(K_h, kernel, xtrain, xpred)
    mul!(mu_h, K_h', alpha)
return mu_h  
end

"""
Compute a single value of σ using GP components in a memory efficient way
"""
function compute_σ_single(gp, x_pred, preallocs)
    GaussianProcesses.cov!(preallocs.Kcross, gp.kernel, gp.x, x_pred)
    GaussianProcesses.cov!(preallocs.Kpred, gp.kernel, x_pred, x_pred)
    Lck = whiten_component(gp, preallocs.Kcross) 
    GaussianProcesses.subtract_Lck!(preallocs.Kpred, Lck)
    return preallocs.Kpred[1]
end

function whiten_component(gp::LocalGP, Kcross)
    return LinearAlgebra.ldiv!(transpose(gp.cKcholut), Kcross)
end

function whiten_component(gp::GPE, Kcross)
    return PDMats.whiten!(gp.cK, Kcross)
end

end # module
