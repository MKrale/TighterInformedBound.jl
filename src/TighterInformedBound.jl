module TighterInformedBound
    using POMDPs, POMDPTools, Random, Distributions, SparseArrays, JuMP, Clp

    include("Beliefs.jl")
    export DiscreteHashedBelief, DiscreteHashedBeliefUpdater
    include("Caching.jl")
    export print_model_info
    include("Weights.jl")
    # Both QMDP and FIB use alpha-vectors in the original version, which is slow...
    include("SimpleHeuristics.jl")
    export QMDPSolver_alt, FIBSolver_alt,
    QMDPPlanner_alt, FIBPlanner_alt,
    action_value, get_heuristic_pointset

    include("solver.jl")
    export
    STIBSolver, OTIBSolver, SawTIBSolver, ISawTIBSolver, ETIBSolver, CTIBSolver, ICTIBSolver, MultiTIBSolver
    STIBPolicy, OTIBPolicy, SawTIBPolicy, ISawTIBPolicy, ETIBPolicy, CTIBPolicy, ICTIBPolicy, MultiTIBPolicy
    include("Convenience.jl")
    export 
    sparseTabularPOMDPFiniteReward
end