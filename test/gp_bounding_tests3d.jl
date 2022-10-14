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

    # Test compute_z_intervals
    x_t = gp.x[:,1]
    x_L = [0.3, 0.3, 0.3]
    x_U = [0.5, 0.5, 0.5]
    theta_vec = ones(gp.dim) * 1 ./ (2*gp.kernel.ℓ2)
    theta_vec_train_squared = zeros(gp.nobs);
    for i = 1:gp.nobs
        @views theta_vec_train_squared[i] = transpose(theta_vec) * (gp.x[:, i].^2)
    end   
    z_interval = @views LearningAbstractions.GPBounding.compute_z_intervals(x_t, x_L, x_U, theta_vec, gp.dim)
    @info z_interval
    @test z_interval[1] ≈ 16.7870941340354 && z_interval[2] ≈ 18.346537704280646

    α_train = gp.alpha 
    sigma_prior = gp.kernel.σ2 # confirmed
    α_train *= sigma_prior # confirmed

    # Test linear_lower_bound
    a_i, b_i = LearningAbstractions.GPBounding.linear_lower_bound(α_train[1], z_interval[1], z_interval[2]) 
    @info a_i, b_i
    @test a_i ≈ 3.856541789101923e-7
    @test b_i ≈ -2.0771153254783878e-8

    # Test the whole components
    H, f, C, a_i_sum = LearningAbstractions.GPBounding.calculate_components(α_train, theta_vec_train_squared, theta_vec, gp.x, x_L, x_U, gp.dim)
    @info H, f, C, a_i_sum 
    @test H ≈ [-0.025487184466174893, -0.025487184466174893, -0.025487184466174893]
    @test f ≈ [0.19695021438849705 -0.06555594478731178 0.14561785192536125]
    @test C ≈ -0.22911041378634575
    @test a_i_sum ≈ 0.22007839142970828

    # Test separate_quadratic_program
    x_mu_lb, f_val = LearningAbstractions.GPBounding.separate_quadratic_program(H, f, x_L, x_U)
    @info x_mu_lb, f_val  
    @test x_mu_lb == [0.3, 0.5, 0.3]
    @test f_val ≈ 0.064512702840274

    # Test μ prediction
    μ, _ = predict_f(gp, hcat(x_mu_lb))
    @info μ 
    @test μ[1] ≈ 0.07346107351793149
end
