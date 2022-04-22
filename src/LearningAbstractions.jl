module LearningAbstractions

using GaussianProcesses
using StatsBase

using Meshes
using LinearAlgebra: norm, I, nullspace
using Plots
using SparseArrays
using StaticArrays
using ConvexBodyProximityQueries

using ProgressMeter


# Write your package code here.
include("gpwrapper.jl")
include("GPBounding/GPBounding.jl")
include("rkhs.jl")
include("discretization.jl")
include("transitions.jl")
include("imdptools.jl")
include("plotting.jl")

function find_state_images(grid, gps)
	n_states = prod(grid.dims)
	all_states_SA = Vector{SMatrix}(undef, n_states)
	all_state_images = Vector{Any}(undef, n_states)
	all_state_σ_bounds = Vector{Any}(undef, n_states) 
	all_state_means = Vector{SVector}(undef, n_states) 
	all_image_means = Vector{Any}(undef, n_states) 

	p = Progress(n_states, desc="Computing image bounds...", dt=0.01)

	neg_gps = []
	for gp in gps
		neg_gp = deepcopy(gp)
		neg_gp.alpha *= -1
		push!(neg_gps, neg_gp)
	end

	Threads.@threads for i=1:n_states      # Implicit ordering of the states remains the same
		state = LearningAbstractions.polytope_to_SA(grid[i])
		all_states_SA[i] = state
		image, all_state_σ_bounds[i] = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], gps, neg_gps) 
		all_state_images[i] = extent_to_SA(image)
		all_state_means[i] = (state[:,1] + state[:,end-1])/2
		all_image_means[i] = (all_state_images[i][:,1] + all_state_images[i][:,end-1])/2
		next!(p)
	end

	return all_states_SA, all_state_images, all_state_σ_bounds, all_state_means, all_image_means 
end

end
