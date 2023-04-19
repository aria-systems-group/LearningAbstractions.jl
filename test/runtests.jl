using LearningAbstractions
using Test

@testset "LearningAbstractions.jl" begin
    # Write your tests here.
    include("abstraction_tests.jl")
    include("local_gps.jl")
    include("transitions.jl")
end
