module TIB
    using POMDPs, POMDPTools, Random, Distributions, SparseArrays, Optimization, JuMP, Gurobi, HiGHS, Tulip, Cbc, Clp, Memoization, LRUCache, InteractiveUtils

    # Surpressing Gurobis printing
    oldstd = stdout
    redirect_stdout(devnull)
    const GRB_ENV=Gurobi.Env()
    redirect_stdout(oldstd)

    # Convenience function for debugging
    printdb(x) = print(x,"\n")
    function printdb(x,y...)
        print(x,", ")
        printdb(y...)
    end

    include("Beliefs.jl")
    export DiscreteHashedBelief, DiscreteHashedBeliefUpdater
    include("Caching.jl")
    export test_get_three_step_beliefs
    include("Weights.jl")
    # Both QMDP and FIB use alpha-vectors in the original version, which is slow...
    include("SimpleHeuristics.jl")
    export QMDPSolver_alt, FIBSolver_alt,
    QMDPPlanner_alt, FIBPlanner_alt,
    action_value, get_heuristic_pointset

    include("solver.jl")
    export
    STIBSolver, OTIBSolver, ETIBSolver, CTIBSolver, ICTIBSolver, MultiTIBSolver
    STIBPolicy, OTIBPolicy, ETIBPolicy, CTIBPolicy, ICTIBPolicy, MultiTIBPolicy
    include("Convenience.jl")
end