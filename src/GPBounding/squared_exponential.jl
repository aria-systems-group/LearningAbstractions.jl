"Computes the lower bound of the posterior mean function of a Gaussian process in an interval."
function compute_μ_lower_bound(gp, x_L, x_U, theta_vec_train_squared, theta_vec; upper_flag=false)
    # Set minmax_factor to -1 if maximizing
    minmax_factor = upper_flag ? -1. : 1.
    # global variables: training_data, theta_vec_train_squared
    @views x_train = gp.x # Confirmed
    m = gp.nobs # Confirmed
    n = gp.dim # Dimension of input
    α_train = gp.alpha 
    
    sigma_prior = gp.kernel.σ2 # confirmed
    α_train *= sigma_prior # confirmed
    
    # Get the theta vector
    H, f, C, a_i_sum = calculate_components(α_train, theta_vec_train_squared, theta_vec, x_train, x_L, x_U, n)
    x_mu_lb, f_val = separate_quadratic_program(H, f, x_L, x_U)
    x_mu_lb = hcat(x_mu_lb) # TODO: get around hcat?
    
    lb = minmax_factor*(f_val + C + a_i_sum)
    ub, _ = predict_f(gp, x_mu_lb)  # TODO: This is very slow!!!! 1/2 a millesecond. Maybe not too slow.
    ub = ub[1]*minmax_factor
    
    if upper_flag
        return x_mu_lb, ub, lb
    else
        return x_mu_lb, lb, ub
    end
end

function calculate_components(α_train, theta_vec_train_squared, theta_vec, x_train, x_L, x_U, n)
    # For each training point
    # could replace z with matrix
    a_i_sum = 0. 
    # b_i_vec = zeros((1,length(α_train)))
    b_i_vec = Array{Float64}(undef, length(α_train))
    C = 0.
    for idx=1:length(α_train)
        z_interval = @views compute_z_intervals(x_train[:, idx], x_L, x_U, theta_vec, n)
        a_i, b_i = linear_lower_bound(α_train[idx], z_interval[1], z_interval[2]) # Confirmed!
        b_i_vec[idx] = b_i
        a_i_sum += a_i
        C += b_i * theta_vec_train_squared[idx] 
    end

    # Hessian object, with respect to each "flex" point
    H = 2*sum(b_i_vec)*theta_vec   # nx1 vector
    f = -2*theta_vec' .* (b_i_vec'*x_train')

    return H, f, C, a_i_sum
end

function split_region(x_L, x_U)

    n = length(x_L)
    x_avg = (x_L + x_U)/2
    # if n == 1
    #     return [(x_L, x_avg), (x_avg, x_U)]
    # end
    
    # if n == 2
    #     return [(x_L, x_avg), (x_avg, x_U),
    #             ([x_avg[1], x_L[2]], [x_U[1], x_avg[2]]),
    #             ([x_L[1], x_avg[2]], [x_avg[1], x_U[2]])]
    # end

    # pairs = []
    # for i=1:n
    #     push!(pairs, [[x_L[i], x_avg[i]], [x_avg[i], x_U[i]]])
    # end

    # if n==1
    #     return pairs
    # end

    lowers = []
    uppers = []
    for i=1:n
        push!(lowers, [x_L[i], x_avg[i]])
        push!(uppers, [x_avg[i], x_U[i]])
    end
    # lower_ranges = collect(Base.product(lowers...))
    # upper_ranges = collect(Base.product(uppers...))

    return [[[lower...], [upper...]] for (lower, upper) in zip(Base.product(lowers...), Base.product(uppers...))]
end

# Don't know if this will even help!
# function split_region!(pairs::Vector{Any}, x_L, x_U)

#     n = length(x_L)
#     # TODO: Does not work in general? Is this still true>
#     x_avg = (x_L + x_U)/2

#     num_regions = 2^n

#     # pairs = []
#     # for i=1:n
#     #     pairs = [[x_L[i], x_avg[i]], [x_avg[i], x_U[i]]]
#     # end
   
    
#     lowers = [[x_L[i], x_avg[i]] for i=1:n]
#     uppers = [[x_avg[i], x_U[i]] for i=1:n]
#     # for i=1:n # For each dim
#     #     lowers[i] = [x_L[i], x_avg[i]]
#     #     # push!(lowers, [x_L[i], x_avg[i]])
#     #     # push!(uppers, [x_avg[i], x_U[i]])
#     # end
#     lower_ranges = collect(Base.product(lowers...))
#     upper_ranges = collect(Base.product(uppers...))

#     return [[[lower...], [upper...]] for (lower, upper) in zip(lower_ranges, upper_ranges)]
# end

"Computes the upper bound of the posterior covariance function of a Gaussian process in an interval."
function compute_σ_upper_bound(gp, x_L, x_U, R_inv)
#     σ_noise = exp(gp.logNoise.value)
#     R_inv = [σ_noise^2 0, 0 σ_noise^2] # TODO: Generalize to any dimension
    # global variables: training_data, theta_vec_train_squared
    x_train = gp.x # Confirmed
    m = gp.nobs # Confirmed
    n = gp.dim # Dimension of input
    α_train = gp.alpha 
    # TODO: Check the others too (k matrix, etc.)
    
    sigma_prior = gp.kernel.σ2 # confirmed
    α_train *= sigma_prior # confirmed
    
    # Get the theta vector
    theta_vec = ones(n) * 1 ./ (2*gp.kernel.ℓ2) # confirmed
    # Create the bounds on ϕ (z bounds)
    # z_an(x) = compute_z_intervals(x, x_L, x_U, theta_vec) # FAIL it hates this 
    z_i_vector = [compute_z_intervals(view(gp.x, :, i), x_L, x_U, theta_vec) for i=1:m] 
    # z_i_vector = z_an.(eachcol(gp.x))
    
    a_i_sum = 0. 
    b_i_vec = zeros((1,m))

    # For each training point
    # THIS LOOP takes FOREVER: https://discourse.julialang.org/t/improving-performance-of-a-nested-for-loop/29705/4
    for idx=1:(m::Int)

        for subidx=1:(idx::Int)
            z_il_L = z_i_vector[idx][1] + z_i_vector[subidx][1]
            z_il_U = z_i_vector[idx][2] + z_i_vector[subidx][2] 
            
            a_i, b_i = linear_lower_bound(R_inv[idx, subidx], z_il_L, z_il_U) 
            b_i_vec[idx] += b_i 
            if subidx < idx
                a_i *= 2 
                b_i_vec[subidx] += b_i
            end
            a_i_sum += a_i
            
        end
    end

    # Hessian object, with respect to each "flex" point
    H = 4*sum(b_i_vec)*theta_vec   # nx1 vector
    f = -2*theta_vec' .* (b_i_vec*x_train')

    C = 0.    
    theta_vec_train_squared = zeros(m);
    for i = 1:m
        theta_vec_train_squared[i] = transpose(theta_vec) * (gp.x[:, i].^2);
    end

    # TODO: Check this multiplication, doesn't matter for 1D
                               # Nx1         1x?
#     theta_vec_train_squared = theta_vec*transpose(x_train.^2)
    for idx=1:m
       C += 2 * b_i_vec[idx] * theta_vec_train_squared[idx] 
    end
    x_σ_ub, f_val = separate_quadratic_program(H, f, x_L, x_U)
    
#     x_mu_lb = transpose(x_mu_lb)
    x_σ_ub = hcat(x_σ_ub)
#     @info x_mu_lb
    
    σ2_ub = sigma_prior*(1. - (f_val + C + a_i_sum))
    if σ2_ub < 0
        @warn "σ2_ub less than zero. Using trivial UB."
        σ_ub = sqrt(sigma_prior)
    else
        σ_ub = sqrt(σ2_ub)
    end
    
    _, σ2_lb = predict_f(gp, x_σ_ub)
    
    return x_σ_ub, sqrt(σ2_lb[1]), σ_ub
end

function compute_z_intervals(x_i, x_L, x_U, theta_vec, n)
    z_i_L = 0.
    dx_L = (x_i - x_L).^2
    dx_U = (x_i - x_U).^2
    for idx=1:n
        if x_L[idx] > x_i[idx] || x_i[idx] > x_U[idx]
            z_i_L += theta_vec[idx] * min(dx_L[idx], dx_U[idx]) 
        end
    end
    z_i_U = transpose(theta_vec) * max(dx_L, dx_U) # Vector with largest component 
    return z_i_L, z_i_U
end

function linear_lower_bound(α, z_i_L, z_i_U)
 # Now compute the linear under approximation (inlined for computational reasons)
    if α >= 0.
        z_i_avg = (z_i_L + z_i_U)/2
        # Where does this come from?
        a_i = (1 + z_i_avg)*α*exp(-z_i_avg)
        b_i = -α*exp(-z_i_avg)
    else
        # This looks more like a slope equation
        # dz = z_i_L - z_i_U 
        a_i = α*(exp(-z_i_L) - z_i_L*(exp(-z_i_L) - exp(-z_i_U))/(z_i_L - z_i_U ) )
        b_i = α*(exp(-z_i_L) - exp(-z_i_U))/(z_i_L - z_i_U )
    end
    
    return a_i, b_i
end

"A simple quadratic program solver."
function separate_quadratic_program(H, f, x_L, x_U; C=0.)

    # By default, set the optimal points to the lower bounds
    x_star = Array(x_L)   # Same size as number of dimensions
    f_val = 0.    # Value at x*
    n = length(x_L) # Number of dimensions. 
    calc_f_part(ddf, df, point) = 0.5*ddf*point.^2 + df*point
    
    for idx=1:n
        # This is odd notation. shouldn't the lower bound always be less than upper?
#         if x_L[idx] < x_U[idx]
        x_critic = -f[idx]/H[idx]

        if H[idx] >= 0 && (x_critic <= x_U[idx]) && (x_critic >= x_L[idx])
            x_star[idx] = x_critic
            f_val_partial = calc_f_part(H[idx], f[idx], x_critic)
        else
            vec = [calc_f_part(H[idx], f[idx], x_L[idx]), 
                   calc_f_part(H[idx], f[idx], x_U[idx])]
            f_val_partial = minimum(vec)
            if f_val_partial == vec[2]
                x_star[idx] = x_U[idx]
            end
        end
#         else
#             f_val_partial = calc_f_part(H[idx], f[idx], x_L[idx])
#         end
        f_val += f_val_partial
    end
    
    return x_star, f_val + C
end
