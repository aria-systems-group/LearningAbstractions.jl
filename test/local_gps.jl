using LearningAbstractions
using LinearAlgebra
using GaussianProcesses
using Random
using Test

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
    tree = LearningAbstractions.create_data_tree(x, "") 
    y = vcat(y', y') # hack 

    # do explicit construction
    local_gps_explicit, sub_idxs = LearningAbstractions.create_local_gps(x, y, center; num_neighbors=10, kernel_params=[0.0, 0.0], lnoise=logObsNoise, tree=tree)
    local_gp_explicit = local_gps_explicit[1] # Just look at first one

    #==
    Local GP Implicit
    ==#
    gpnobs = nns 
    local_gp_implicit = LearningAbstractions.GPBounding.LocalGP(
        Vector{Int}(undef, gpnobs),
        gp_explicit.x[:, 1:nns], 
        Vector{Float64}(undef, gpnobs), 
        Matrix{Float64}(undef, gpnobs, gpnobs),
        Matrix{Float64}(undef, gpnobs, gpnobs),
        UpperTriangular(zeros(gpnobs, gpnobs)),
        Symmetric(Matrix{Float64}(undef, gpnobs, gpnobs)),
        gp_explicit.kernel,
        Vector{Float64}(undef, gpnobs)
    ) 

    LearningAbstractions.create_local_gps!([local_gp_implicit], [gp_explicit], center, tree, x, y) 

    @test sub_idxs == local_gp_implicit.sub_idxs
    @test local_gp_explicit.cK ≈ local_gp_implicit.cK

    K_h = zeros(nns,1)
    mu_h = zeros(1,1)

    # test prediction
    gpdim = 2
    preallocs = LearningAbstractions.ImageBoundPreallocation(
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

    x_pred = rand(2,1)
    μ_ex, σ_ex = predict_f(local_gp_explicit, hcat(x_pred))
    μ_im = LearningAbstractions.GPBounding.predict_μ(local_gp_implicit, x_pred, K_h, mu_h)

    @test local_gp_explicit.alpha ≈ local_gp_implicit.alpha

    σ_im = LearningAbstractions.GPBounding.compute_σ_single(local_gp_implicit, x_pred, preallocs)
    σ_ex_s = LearningAbstractions.GPBounding.compute_σ_single(local_gp_explicit, x_pred, preallocs) 
    @test μ_ex[1] ≈ μ_im[1]
    @test σ_ex[1] ≈ σ_im        
    @test σ_ex[1] ≈ σ_ex_s
end
