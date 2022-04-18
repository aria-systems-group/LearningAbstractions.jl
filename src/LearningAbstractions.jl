module LearningAbstractions

using GaussianProcesses
using StatsBase

using Meshes
using LinearAlgebra: norm
using Plots
using SparseArrays
using StaticArrays
using ConvexBodyProximityQueries


# Write your package code here.
include("gpwrapper.jl")
include("GPBounding/GPBounding.jl")

include("discretization.jl")
include("transitions.jl")
include("imdptools.jl")
include("plotting.jl")

end
