using LearningAbstractions
using GaussianProcesses
using Random
using Test

@testset "GPBounding.jl" begin

    # Initialize the GP
    Random.seed!(35)
    # Training data
    n=100;                          #number of training points
    x = 2π * rand(3,n);              #predictors
    obs_noise = 0.01
    y = sin.(x[1,:].*x[2,:]) + obs_noise*randn(n);   #regressors
    logObsNoise = log10(obs_noise)

    #Select mean and covariance function
    mZero = MeanZero()                   #Zero mean function
    kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

    gp = GP(x,y,mZero,kern,logObsNoise)       #Fit the GP

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

    # Test compute_z_intervals
    x_t = gp.x[:,1]
    x_L = [0.3, 0.3, 0.3]
    x_U = [0.5, 0.5, 0.5]
    theta_vec = ones(gp.dim) * 1 ./ (2*gp.kernel.ℓ2)
    theta_vec_train_squared = zeros(gp.nobs);
    for i = 1:gp.nobs
        @views theta_vec_train_squared[i] = transpose(theta_vec) * (gp.x[:, i].^2)
    end   
    
    z_interval = @views LearningAbstractions.GPBounding.compute_z_intervals(x_t, x_L, x_U, theta_vec, gp.dim, dx_L, dx_U)
    @test z_interval[1] ≈ 16.7870941340354 && z_interval[2] ≈ 18.346537704280646

    α_train = gp.alpha 
    sigma_prior = gp.kernel.σ2 # confirmed
    α_train *= sigma_prior # confirmed

    # Test linear_lower_bound
    a_i, b_i = LearningAbstractions.GPBounding.linear_lower_bound(α_train[1], z_interval[1], z_interval[2]) 
    @test a_i ≈ 3.856541789101923e-7
    @test b_i ≈ -2.0771153254783878e-8

    # Test the whole components
    H, f, C, a_i_sum = LearningAbstractions.GPBounding.calculate_components(α_train, theta_vec_train_squared, theta_vec, gp.x, x_L, x_U, gp.dim, b_i_vec, dx_L, dx_U, H, f, bi_x_h)
    @test H ≈ [-0.025487184466174893, -0.025487184466174893, -0.025487184466174893]
    @test f ≈ [0.19695021438849705 -0.06555594478731178 0.14561785192536125]
    @test C ≈ -0.22911041378634575
    @test a_i_sum ≈ 0.22007839142970828

    # Test separate_quadratic_program
    f_val = LearningAbstractions.GPBounding.separate_quadratic_program(H, f, x_L, x_U, x_star_h, vec_h)
    @test x_star_h == [0.3, 0.5, 0.3]
    @test f_val ≈ 0.064512702840274

    # Test μ prediction
    μ = LearningAbstractions.GPBounding.predict_μ(gp, hcat(x_star_h), K_h, mu_h)
    @test μ[1] ≈ 0.07346107351793149
end