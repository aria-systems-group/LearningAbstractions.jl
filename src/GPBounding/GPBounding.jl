module GPBounding

using GaussianProcesses
using LinearAlgebra
using Random
using Distributions
using StaticArrays
using Tullio
include("squared_exponential.jl")

export bound_image, bound_images

# """ 
#     bound_images

# Generate overapproximations of posterior mean and covariance functions for a collection of extents.
# """
# function bound_images(extents, gps)
#     image_bounds = Vector{Any}(undef, length(extents))
#     for i=1:length(extents)
#         image_bounds[i] = bound_image(extents[i], gps)
#     end
#     return image_bounds
# end

""" 
    bound_image

Generate overapproximations of posterior mean and covariance functions using one of several methods.
# Arguments
- `extent` - Discrete state extent 
- `gps` - Vector of GPs and associated metadata
- `neg_gps` - Vector of GPs with -1*α vector
- `delta_input_flag` - True uses `x` as the known component 
"""
function bound_image(extent, gps::Vector{Any}, neg_gps::Vector{Any}; delta_input_flag=false, data_deps=nothing, known_component=nothing, σ_ubs=nothing, σ_approx_flag=false)
    # TODO: mod keyword handling and document
    ndims = length(gps) 
    image_extent = Vector{Vector{Float64}}(undef, ndims)
    σ_bounds = zeros(ndims) 
    for i=1:ndims 
        image_extent[i], σ_bounds[i] = bound_extent_dim(gps[i], neg_gps[i], extent[1], extent[2])
        if delta_input_flag
            image_extent[i][1] += extent[1][i]
            image_extent[i][2] += extent[2][i]
        end
    end
    return image_extent, σ_bounds
end

function bound_extent_dim(gp, neg_gp, lbf, ubf; approximate_flag=false)
    # # lbf = lb[findall(.>(0), data_deps[dim_key][:])]
    # ubf = ub[findall(.>(0), data_deps[dim_key][:])]
    # TODO: Avoid deep copy! At all costs! For 2000 dps, 404MB vs 38MB!
    # x_lb, μ_L_lb, μ_L_ub = compute_μ_bounds_bnb(deepcopy(gp), lbf, ubf) 
    if approximate_flag
        μ_L_lb, μ_U_ub = compute_μ_bounds_approx(gp, lbf, ubf) 
    else
        _, μ_L_lb, _ = compute_μ_bounds_bnb(gp, lbf, ubf) 
        # x_ub, μ_U_lb, μ_U_ub = compute_μ_bounds_bnb(deepcopy(gp), lbf, ubf, max_flag=true)
        _, _, μ_U_ub = compute_μ_bounds_bnb(neg_gp, lbf, ubf, max_flag=true)
    end

    # if σ_approx_flag
    _, σ_U_lb, σ_U_ub = compute_σ_ub_bounds_approx(gp, lbf, ubf) 
    # elseif !isnothing(σ_ubs)
    # _, σ_U_lb, σ_U_ub = compute_σ_ub_bounds_from_gp(gp, lbf, ubf)
    # else
    #     _, σ_U_lb, σ_U_ub = compute_σ_ub_bounds(gps[i], gp_info_dict[dim_key].Kinv, lbf, ubf)
    # end
    
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

function compute_μ_bounds_bnb(gp, x_L, x_U; max_iterations=100, bound_epsilon=1e-2, max_flag=false)
    # By default, it calculates bounds on the minimum. 
    theta_vec_train_squared = zeros(gp.nobs);
    theta_vec = ones(gp.dim) * 1 ./ (2*gp.kernel.ℓ2)
    for i = 1:gp.nobs
        @views theta_vec_train_squared[i] = transpose(theta_vec) * (gp.x[:, i].^2)
    end   

    # Preallocated arrays for memory savings 
    m_sub = gp.nobs
    b_i_vec = Array{Float64}(undef, m_sub)
    dx_L = zeros(gp.dim)
    dx_U = zeros(gp.dim)
    H = zeros(gp.dim)
    f = zeros(1, gp.dim)
    x_star_h = zeros(gp.dim)
    vec_h = zeros(2)
    bi_x_h = zeros(1,gp.dim)
    α_h = zeros(gp.nobs)
    K_h = zeros(gp.nobs,1)
    mu_h = zeros(1,1)
    
    x_best, lbest, ubest = compute_μ_lower_bound(gp, x_L, x_U, theta_vec_train_squared, theta_vec, b_i_vec, dx_L, dx_U, H, f, x_star_h, vec_h, bi_x_h, α_h, K_h, mu_h, upper_flag=max_flag)
    if max_flag
        temp = lbest
        lbest = -ubest
        ubest = -temp
    end
    
    candidates = [(x_L, x_U)]
    iterations = 0

    split_regions = nothing
    x_avg = zeros(gp.dim)

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

function compute_σ_ub_bounds(gp, K_inv, x_L, x_U; max_iterations=10, bound_epsilon=1e-4)
    
    lbest = -Inf
    ubest = 1.
    x_best = nothing
    
    candidates = [[(x_L, x_U), lbest, ubest]]
    iterations = 0
   
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
            
            bound_pairs = split_region(extent[1], extent[2])

            for pair in bound_pairs
                x_ub1, lb1, ub1 = compute_σ_upper_bound(gp, pair[1], pair[2], K_inv)
                
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
        x_best, lbest, ubest = compute_σ_upper_bound(gp, x_L_n, x_U_n, K_inv)
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

function compute_σ_ub_bounds_approx(gp, x_L, x_U; N=100)
    mt = MersenneTwister(11)
    σ2_best = -Inf
    # Get N samples uniformly dist.
    x_samp = vcat([rand(mt, Uniform(x_L[i], x_U[i]), 1, N) for i=1:length(x_L)]...)
    _, σ2 = predict_f(gp, x_samp)
    σ2_best = maximum(σ2)
    return 0., 0., sqrt(σ2_best[1])
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

# ! Can improve further here
function predict_μ(gp, xpred, K_h, mu_h)
    xtrain = gp.x
    kernel = gp.kernel
    alpha = gp.alpha

    GaussianProcesses.cov!(K_h, kernel, xtrain, xpred)
    # ! Mean zero specialty
    mul!(mu_h, K_h', alpha)
return mu_h  
end

end # module
