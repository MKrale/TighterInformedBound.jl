using TighterInformedBound, POMDPs
using Test

include("ABCModel.jl")
using .ABCModel

@testset "TighterInformedBound.jl" begin
    pomdp = ABC()
    solver = STIBSolver()

    # Standard TIB
    policy = solve(solver, pomdp)
    Vs, _Bsao, _Vsao = get_heuristic_pointset(policy)
    @test (isapprox(Vs, [1.0, 1.0, 0]))
    @test (isapprox(POMDPs.value(policy, POMDPs.initialstate(pomdp)), 0.68 * 0.95^2))
end
