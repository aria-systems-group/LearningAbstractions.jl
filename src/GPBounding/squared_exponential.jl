"Computes the lower bound of the posterior mean function of a Gaussian process in an interval."
function compute_μ_lower_bound(gp::GPE, x_L, x_U, theta_vec_train_squared::Vector{Float64}, theta_vec::Vector{Float64}, 
                               b_i_vec::Vector{Float64}, dx_L::Vector{Float64}, dx_U::Vector{Float64}, H::Vector{Float64}, f::Matrix{Float64}, x_star_h::Vector{Float64}, vec_h::Vector{Float64}, bi_x_h::Matrix{Float64}, α_h::Vector{Float64},
                               K_h, mu_h;
                               upper_flag=false)
    # Set minmax_factor to -1 if maximizing
    minmax_factor = upper_flag ? -1. : 1.
    x_train = gp.x # Confirmed
    n = gp.dim # Dimension of input
    α_h .= gp.alpha .* gp.kernel.σ2
    
    H, f, C, a_i_sum = calculate_components(α_h, theta_vec_train_squared, theta_vec, x_train, x_L, x_U, n, b_i_vec, dx_L, dx_U, H, f, bi_x_h)
    f_val = separate_quadratic_program(H, f, x_L, x_U, x_star_h, vec_h)
    x_mu_lb = hcat(x_star_h) # TODO: get around hcat?
    
    lb = minmax_factor*(f_val + C + a_i_sum)
    ub = predict_μ(gp, x_mu_lb, K_h, mu_h) 
    ub = ub[1]*minmax_factor
    
    if upper_flag
        return x_mu_lb, ub, lb
    else
        return x_mu_lb, lb, ub
    end
end


function calculate_components(α_train::Vector{Float64}, theta_vec_train_squared::Vector{Float64}, theta_vec::Vector{Float64}, x_train::Matrix{Float64}, x_L, x_U, n::Int, 
                              b_i_vec::Vector{Float64}, dx_L::Vector{Float64}, dx_U::Vector{Float64}, H::Vector{Float64}, f::Matrix{Float64}, bi_x_h::Matrix{Float64})
    a_i_sum = 0. 
    b_i_vec_sum = 0.
    C = 0.
    
    length(α_train)
    for idx=1:length(α_train)  
        @views z_i_L, z_i_U = compute_z_intervals(x_train[:, idx], x_L, x_U, theta_vec, n, dx_L, dx_U)           
        a_i, b_i = linear_lower_bound(α_train[idx], z_i_L, z_i_U ) # Confirmed!     
        b_i_vec[idx] = b_i
        b_i_vec_sum += b_i
        a_i_sum += a_i 
        C += b_i * theta_vec_train_squared[idx]
    end

    # Hessian object, with respect to each "flex" point
    H .= 2*b_i_vec_sum.*theta_vec   # nx1 vector
    mul!(bi_x_h, b_i_vec', x_train')
    @tullio f[i] = -2*theta_vec[i] .* bi_x_h[i]  

    return H, f, C, a_i_sum
end

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

function compute_z_intervals(x_i, x_L, x_U, theta_vec::Vector{Float64}, n::Int, dx_L::Vector{Float64}, dx_U::Vector{Float64})
    z_i_L = 0.
    dx_L .= (x_i .- x_L).^2       # TODO: This still takes much time, improve further
    dx_U .= (x_i .- x_U).^2
    @inbounds for idx=1:n
        if x_L[idx] > x_i[idx] || x_i[idx] > x_U[idx]
            minval = dx_L[idx] < dx_U[idx] ? dx_L[idx] : dx_U[idx] 
            z_i_L += theta_vec[idx] * minval
        end
    end

    # TODO: Review what this max does, can we simplify it somehow, like put it in the loop above?
    z_i_U = transpose(theta_vec) * max(dx_L, dx_U) # Vector with largest component 
    return z_i_L, z_i_U
end

function linear_lower_bound(α::Float64, z_i_L::Float64, z_i_U::Float64)
 # Now compute the linear under approximation (inlined for computational reasons)
    if α >= 0.
        z_i_avg = (z_i_L + z_i_U)/2
        e_avg = exp(-z_i_avg)
        αe = e_avg*α
        a_i = (1 + z_i_avg)*αe
        b_i = -αe
    else
        dz = z_i_L - z_i_U 
        ezL = exp(-z_i_L) 
        ezU = exp(-z_i_U) 
        de = ezL - ezU
        b_i = α*(de)/(dz)
        a_i = α*ezL - z_i_L*b_i
    end
    
    return a_i, b_i
end

"A simple quadratic program solver."
function separate_quadratic_program(H::Vector{Float64}, f::Matrix{Float64}, x_L::Vector{Float64}, x_U::Vector{Float64}, x_star_h::Vector{Float64}, vec_h::Vector{Float64}; C=0.)

    # By default, set the optimal points to the lower bounds
    f_val = 0.    # Value at x*
    n = length(x_L) # Number of dimensions. 
    calc_f_part(ddf::Float64, df::Float64, point::Float64) = 0.5*ddf*point.^2 + df*point
    
    for idx=1:n
        x_critic = -f[idx]/H[idx]
        if H[idx] >= 0 && (x_critic <= x_U[idx]) && (x_critic >= x_L[idx])
            x_star_h[idx] = x_critic
            f_val_partial = calc_f_part(H[idx], f[idx], x_critic)
        else
            vec_h[1] = calc_f_part(H[idx], f[idx], x_L[idx])
            vec_h[2] = calc_f_part(H[idx], f[idx], x_U[idx])   
            f_val_partial = minimum(vec_h)
            if f_val_partial == vec_h[2]
                x_star_h[idx] = x_U[idx]
            end
        end
        f_val += f_val_partial
    end
    
    return f_val + C
end
