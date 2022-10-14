using LearningAbstractions
using GaussianProcesses
using Random
using Test

@testset "GPBounding.jl" begin

    # Initialize the GP
    Random.seed!(35)
    # Training data
    n=100;                          #number of training points
    x = 2π * rand(2,n);              #predictors
    obs_noise = 0.01
    y = sin.(x[1,:].*x[2,:]) + obs_noise*randn(n);   #regressors
    logObsNoise = log10(obs_noise)

    #Select mean and covariance function
    mZero = MeanZero()                   #Zero mean function
    kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

    gp = GP(x,y,mZero,kern,logObsNoise)       #Fit the GP

    x_test = [-0.6; 0.3]

    # Test compute_z_intervals
    x_t = gp.x[:,1]
    x_L = [0.3, 0.3]
    x_U = [0.5, 0.5]
    theta_vec = ones(gp.dim) * 1 ./ (2*gp.kernel.ℓ2)
    theta_vec_train_squared = zeros(gp.nobs);
    for i = 1:gp.nobs
        @views theta_vec_train_squared[i] = transpose(theta_vec) * (gp.x[:, i].^2)
    end   
    z_interval = @views LearningAbstractions.GPBounding.compute_z_intervals(x_t, x_L, x_U, theta_vec, gp.dim)
    @test z_interval[1] ≈ 2.5678897035248642 && z_interval[2] ≈ 3.0407797802964613

    α_train = gp.alpha 
    sigma_prior = gp.kernel.σ2 # confirmed
    α_train *= sigma_prior # confirmed

    # Test linear_lower_bound
    a_i, b_i = LearningAbstractions.GPBounding.linear_lower_bound(α_train[1], z_interval[1], z_interval[2]) 
    @test a_i ≈ -1.6496254854710353
    @test b_i ≈ 0.4315114239506915

    # Test the whole components
    H, f, C, a_i_sum = LearningAbstractions.GPBounding.calculate_components(α_train, theta_vec_train_squared, theta_vec, gp.x, x_L, x_U, gp.dim)

    @test H ≈ [-0.026814730657513094, -0.026814730657513094]
    @test f ≈ [0.3859478820145441 -0.041233297215278686]
    @test C ≈ 0.3880304599824592
    @test a_i_sum ≈ -0.4985337971591764

    # Test separate_quadratic_program
    x_mu_lb, f_val = LearningAbstractions.GPBounding.separate_quadratic_program(H, f, x_L, x_U)
    @test x_mu_lb == [0.3, 0.5]
    @test f_val ≈ 0.09060921178494666 

    # Test μ prediction
    μ, _ = predict_f(gp, hcat(x_mu_lb))
    @test μ[1] ≈ 0.16165996767145518

    # Test split_region
    new_regions = LearningAbstractions.GPBounding.split_region(x_L, x_U)
    @test new_regions[1] == [[0.3, 0.3], [0.4, 0.4]]  
    @test new_regions[2] == [[0.4, 0.3], [0.5, 0.4]]  
    @test new_regions[3] == [[0.3, 0.4], [0.4, 0.5]]
    @test new_regions[4] == [[0.4, 0.4], [0.5, 0.5]]
end
