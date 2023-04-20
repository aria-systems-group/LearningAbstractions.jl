using LearningAbstractions
using Random
using Test

@testset "Abstractions" begin

    # dummy state - first col is lower, second-to-last is upper
    state = [0. 1.0 0.0;]
    new_states = LearningAbstractions.uniform_refinement(state)
    using StaticArrays
    new_states_exp = StaticArraysCore.SMatrix{1, 2, Float64, 2}[[0.0 0.5], [0.5 1.0]] 
    @test new_states == new_states_exp

    @test LearningAbstractions.get_hot_idxs(5, [2, 4]) == [1, 3, 5]
    @test LearningAbstractions.pre(2, [0. 1.; 0. 0.; 0. 0.3]) == [1, 3]
    @test LearningAbstractions.post(3, [0. 1.; 0. 0.; 0. 0.3]) == [2, ]

    @test LearningAbstractions.find_states_to_refine(nothing, [1.0 0.0 0.0 1.0; 2.0 0.0 0.0 0.0; 3.0 0.0 0.0 1.0; 4.0 0.0 0.0 0.0], nothing) == [1,3]

    # Test the discretization scheme
    L = [0.0,]
    U = [1.0,]
    δ = [0.25,]
    gens, spacing = LearningAbstractions.grid_generator(L, U, δ)
    @test length(gens)[1] == 4
    @test maximum(gens)[1] == 0.75
    @test spacing[1] == 0.25 

    L = [0.0,]
    U = [1.0,]
    δ = [0.26,]
    gens, spacing = LearningAbstractions.grid_generator(L, U, δ)
    @test length(gens)[1] == 3
    @test maximum(gens)[1] == 2. /3
    @test spacing[1] == 1. /3

    # dummy state - first col is lower, second-to-last is upper
    state = [0. 0. 1.0 1.0; 0.0 1.0 1.0 0.0]
    new_states = LearningAbstractions.uniform_refinement(state)
    new_state_exp = [0.0 0.5 0.5 0.0; 0.0 0.0 0.5 0.5]
    @test new_states[1] == new_state_exp

    state = [0. 0. 1.0 1.0; 0.0 1.0 1.0 0.0]
    idx_refine = 2
    new_states = LearningAbstractions.dimension_refinement(state, idx_refine)
    new_state_exp1 = [0.0 1.0 1.0 0.0; 0.0 0.0 0.5 0.5]
    new_state_exp2 = [0.0 1.0 1.0 0.0; 0.5 0.5 1.0 1.0]
    @test new_states[1] == new_state_exp1
    @test new_states[2] == new_state_exp2
end
