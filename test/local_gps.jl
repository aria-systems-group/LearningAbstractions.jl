using LearningAbstractions
using LinearAlgebra
using GaussianProcesses
using Random
using Test

using PosteriorBounds

@testset "Local GPs Implicit and Explicit" begin

    # Initialize the GP
    Random.seed!(35)
    # Training data
    n=50;                          #number of training points
    x = 2π * rand(2,n);              #predictors
    obs_noise = 0.01
    y = sin.(x[1,:].*x[2,:]) + obs_noise*randn(n)
    logObsNoise = log(obs_noise)

    #Select mean and covariance function
    mZero = MeanZero()                   #Zero mean function
    kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
    gp_explicit = GP(x,y,mZero,kern,logObsNoise)       #Fit the GP

    #== 
    Local GP Explicit
    ==#
    nns = 10
    center = [0.; 0.]
    tree = LearningAbstractions.create_data_tree(x, LearningAbstractions.Euclidean()) 
    y = vcat(y', y') # hack 

    # do explicit construction
    local_gps_explicit, sub_idxs = LearningAbstractions.create_local_gps(x, y, center; num_neighbors=10, kernel_params=[0.0, 0.0], lnoise=logObsNoise, tree=tree)
    local_gp_explicit = local_gps_explicit[1] # Just look at first one

    #==
    Local GP Implicit
    ==#
    gpnobs = nns 
    gpdim = 2
    local_gp_implicit = LearningAbstractions.LocalGP(
        Vector{Int}(undef, gpnobs),
        Vector{Float64}(undef, gpnobs),
        [PosteriorBounds.PosteriorGP(
            gpdim,
            gpnobs,
            x[:,1:gpnobs], 
            Matrix{Float64}(undef, gpnobs, gpnobs),
            Matrix{Float64}(undef, gpnobs, gpnobs),
            UpperTriangular(zeros(gpnobs, gpnobs)),
            Symmetric(Matrix{Float64}(undef, gpnobs, gpnobs)),
            Vector{Float64}(undef, gpnobs), 
            SEKernel(gp_explicit.kernel.σ2, gp_explicit.kernel.ℓ2)
        )]
    )

    LearningAbstractions.create_local_gps!(local_gp_implicit, [gp_explicit], center, tree, x, y) 

    @test sub_idxs == local_gp_implicit.sub_idxs
    @test local_gp_explicit.cK ≈ local_gp_implicit.gps[1].cK

    K_h = zeros(nns,1)
    mu_h = zeros(1,1)
    σ2_h = zeros(1,1)

    # test prediction
    preallocs = PosteriorBounds.preallocate_matrices(gpdim, gpnobs)

    x_pred = rand(2,1)
    μ_ex, σ2_ex = predict_f(local_gp_explicit, hcat(x_pred))
    μ_im = PosteriorBounds.compute_μ!(mu_h, K_h, local_gp_implicit.gps[1], x_pred)

    @test local_gp_explicit.alpha ≈ local_gp_implicit.gps[1].alpha

    σ2_im = PosteriorBounds.compute_σ2!(σ2_h, local_gp_implicit.gps[1], x_pred)
    @test μ_ex[1] ≈ μ_im[1]
    @test σ2_ex[1] ≈ σ2_im 

    # Test out the new distance metric
    met = LearningAbstractions.GeneralMetric([2], [1.0, 1/2*pi])
    x = [0.0 0.0; pi/2 -0.99*pi/2]
    tree = LearningAbstractions.create_data_tree(x, met) 

    # find nearest point
    x_test = [0.0; 0.0]
    idxs = LearningAbstractions.select_knn_idxs(x_test, tree, 1)
    @test idxs[1] == 2
end
