module LearningAbstractions

using GaussianProcesses
using NearestNeighbors
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

function find_state_images(grid, gps, grid_spacing; local_gps_flag=false)
	n_states = length(grid)
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

	Threads.@threads for (i, grid_lower) in collect(enumerate(grid))     # Implicit ordering of the states remains the same
		state = LearningAbstractions.lower_to_SA(grid_lower, grid_spacing)
		all_states_SA[i] = state
		all_state_means[i] = (state[:,1] + state[:,end-1])/2

		if local_gps_flag
			local_gps = create_local_gps(gps, all_state_means[i], num_neighbors=200)
			local_neg_gps = []
			for gp in local_gps
				neg_gp = deepcopy(gp)
				neg_gp.alpha *= -1
				push!(local_neg_gps, neg_gp)
			end
			image, all_state_σ_bounds[i] = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], local_gps, local_neg_gps) 
		else
			image, all_state_σ_bounds[i] = LearningAbstractions.GPBounding.bound_image([state[:,1], state[:,end-1]], gps, neg_gps) 
		end

		all_state_images[i] = extent_to_SA(image)
		all_image_means[i] = (all_state_images[i][:,1] + all_state_images[i][:,end-1])/2
		next!(p)
	end

	return all_states_SA, all_state_images, all_state_σ_bounds, all_state_means, all_image_means 
end

end
