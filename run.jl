# File used for ad-hoc testing.
# It is very messy: please use run_upperbound.jl or run_sarsoptest.jl for testing instead!

using POMDPs
using POMDPTools, POMDPFiles
include("TIB/TIB.jl")
using .TIB
using Statistics, POMDPModels
using SparseArrays
import RockSample
using Profile, PProf

##################################################################
#                           Set Solvers
##################################################################

solvers, solverargs = [], []

iters, tol = 1_000, 10^-4


## FIB
# using FIB
# push!(solvers, FIBSolver_alt)
# push!(solverargs, (name="FIB", sargs=(max_iterations=iters,precision=tol), pargs=(), get_Q0=true))

### TIB
 push!(solvers, STIBSolver)
 push!(solverargs, (name="TIBSolver (standard)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true))

# ### ETIB
# push!(solvers, ETIBSolver)
# push!(solverargs, (name="TIBSolver (entropy)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true))

# ### CTIB (unused)
# push!(solvers, CTIBSolver)
# push!(solverargs, (name="TIBSolver (closeness)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true)) 

### OTIBs
# push!(solvers, OTIBSolver)
# push!(solverargs, (name="TIBSolver (worst-case)", sargs=(max_iterations=iters, precision=tol), pargs=(), get_Q0=true))

# ### SARSOP
# include("Sarsop_altered/NativeSARSOP.jl")
# import .NativeSARSOP_alt
# max_time = 120.0

# push!(solvers, NativeSARSOP_alt.SARSOPSolver)
# solver = FIBSolver_alt(max_iterations=1_000, precision=1e-5)
# push!(solverargs, (name="SARSOP+FIB (max $max_time s)", sargs=( epsilon=1e-2, precision=1e-2, max_time=max_time, verbose=false, heuristic_solver=solver), pargs=(), get_Q0=true))


# push!(solvers, NativeSARSOP_alt.SARSOPSolver)
# push!(solverargs, (name="SARSOP (max $max_time s)", sargs=( epsilon=1e-2, precision=1e-2, max_time=max_time, verbose=false, heuristic_solver=NativeSARSOP_alt.FIBSolver_alt()), pargs=(), get_Q0=true))

# ### SARSOP
# using NativeSARSOP
# push!(solvers, NativeSARSOP.SARSOPSolver)
# push!(solverargs, (name="SARSOP", sargs=(epsilon=0.5, precision=0.001, kappa=0.5, delta=1e-1, max_time=2.0, max_steps=1, verbose=true), pargs=(), get_Q0=true))

### SARSOP (Wrapped)
# using SARSOP
# push!(solvers, SARSOP.SARSOPSolver)
# push!(solverargs, (name="SARSOP (wrapped)", sargs=(timeout=60., precision=1e-5, fast=false), pargs=(), is_AMSolver=false))

# ## POMCP 
# using BasicPOMCP
# push!(solvers, POMCPSolver)
# push!(solverargs, (name="Basic POMCP", sargs=(tree_queries=1000, max_time=0.5), pargs=(),is_AMSolver=false))

### AdaOPS
# using AdaOPS
# push!(solvers, AdaOPSSolver)
# push!(solvernames, "AdaOPS")
# push!(solverargs, Dict())
# push!(policyargs, Dict())

### ARDESPOT
# using ARDESPOT
# push!(solvers, DESPOTSolver)
# push!(solvernames, "ARDESPOT")
# push!(solverargs, Dict())
# push!(policyargs, Dict())

##################################################################
#                           Set Envs 
##################################################################

envs, envargs = [], []
discount = 0.95

# ### ABC
# include("Environments/ABCModel.jl"); using .ABCModel
# abcmodel = SparseTabularPOMDP(ABC(discount=discount))
# push!(envs, abcmodel)
# push!(envargs, (name="ABCModel",))

 # ### Tiger
#  tiger = POMDPModels.TigerPOMDP()
#  tiger.discount_factor = discount
#  push!(envs, tiger)
#  push!(envargs, (name="Tiger",))

# # ### RockSample
import RockSample
### This env is very difficult to work with for some reason...
POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
POMDPs.discount(M::RockSample.RockSamplePOMDP) = discount

# map_size, rock_pos = (7,7), [(1,2), (2,6), (3,3), (3,4), (4,7),(6,1),(6,4),(7,3)] # HSVI setting!
# rocksamplelarge = SparseTabularPOMDP(RockSample.RockSamplePOMDP(map_size, rock_pos))
# push!(envargs, (name="RockSample (7,8)",))
# push!(envs, rocksamplelarge)

# map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
# rocksamplesmall = RockSample.RockSamplePOMDP(map_size, rock_pos)
# push!(envargs, (name="RockSample (5x5)",))
# push!(envs, rocksamplesmall)

# map_size, rock_pos = (10,10), [(2,3), (4,6), (7,4), (8,9) ] # Big Boy!
# rocksamplelarge = RockSample.RockSamplePOMDP(map_size, rock_pos)
# push!(envargs, (name="RockSample (10x10)",))
# push!(envs, rocksamplelarge)

# # ### K-out-of-N
include("Environments/K-out-of-N.jl"); using .K_out_of_Ns

# k_model1 = K_out_of_N(N=1, K=1, discount=discount)
# push!(envs, k_model1)
# push!(envargs, (name="K-out-of-N (1)",))

# k_model2 = K_out_of_N(N=2, K=2, discount=discount)
# push!(envs, k_model2)
# push!(envargs, (name="K-out-of-N (2)",))

# k_model3 = K_out_of_N(N=3, K=3, discount=discount)
# push!(envs, k_model3)
# push!(envargs, (name="K-out-of-N (3)",))

# include("Environments/HeavenOrHell.jl"); using .HeavenOrHellModel
# HHmodel = HeavenOrHell2(discount=discount)
# push!(envs, HHmodel)
# push!(envargs, (name="HeavenOrHell",))

# # # Frozen Lake variants
# include("Environments/CustomGridworld.jl"); using .CustomGridWorlds

#lakesmall = FrozenLakeSmall
#lakesmall.discount = discount
#push!(envs, lakesmall)
#push!(envargs, (name="Frozen Lake (4x4)",))

# lakelarge = FrozenLakeLarge
# lakelarge.discount = discount
# push!(envs, lakelarge)
# push!(envargs, (name="Frozen Lake (10x10)",))

### Hallway Envs

# minihallway = CustomMiniHallway
# minihallway.discount = discount
# push!(envs, minihallway)
# push!(envargs, (name="MiniHallway", ))

# hallway1 = Hallway1
# hallway1.discount = discount
# push!(envs, hallway1)
# push!(envargs, (name="Hallway1",))

# hallway2 = Hallway2
# hallway2.discount = discount
# push!(envs, hallway2)
# push!(envargs, (name="Hallway2",))

# tigergrid = TigerGrid
# tigergrid.discount = discount
# push!(envs, tigergrid)
# push!(envargs, (name="TigerGrid",))

# ### Explicit Sparse Hallways
include("Environments/Sparse_models/SparseModels.jl"); using .SparseModels

# hallway1 = SparseHallway1(discount=discount)
# push!(envs, hallway1)
# push!(envargs, (name="Hallway1 (Sparse)",))

# hallway2 = SparseHallway2(discount=discount)
# push!(envs, hallway2)
# push!(envargs, (name="Hallway2 (Sparse)",))

# tigergrid = SparseTigerGrid(discount=discount)
# push!(envs, tigergrid)
# push!(envargs, (name="TigerGrid (Sparse)",))

# Grid = Sparse_Grid(discount=discount)
# push!(envs, Grid)
# push!(envargs, (name="grid",))


# # ### DroneSurveilance
# # import DroneSurveillance
# # dronesurv = DroneSurveillance.DroneSurveillancePOMDP()
# # push!(envs, dronesurv)
# # push!(envargs, (name="DroneSurveilance",))

 ### Tag
#  using TagPOMDPProblem
#  tag = TagPOMDPProblem.TagPOMDP(discount_factor=discount)
#  push!(envs, tag)
#  push!(envargs, (name="Tag",))

# ### Mini Hallway
# minihall = POMDPModels.MiniHallway()
# push!(envs, minihall)
# push!(envargs, (name="MiniHallway",))

# ### TMaze (Does not work with FIB)
# tmaze = POMDPModels.TMaze()
# POMDPs.reward(tmaze::TMaze, s::POMDPTools.ModelTools.TerminalState,a ) = 0
# push!(envs, tmaze)
# push!(envargs, (name="TMaze",))

# For some reason, the envs below do not work:

### SubHunt (No 'observations' defined)
# import SubHunt
# subhunt = SubHunt.SubHuntPOMDP()
# push!(envs, subhunt)
# push!(envargs, (name="SubHunt",))

# ### LaserTag (No 'observations' defined)
# import LaserTag
# lasertag = LaserTag.gen_lasertag()
# push!(envs, lasertag)
# push!(envargs, (name="LaserTag",))

##################################################################
#                           Run Solvers 
##################################################################

verbose = true
sims, steps = 100, 100

policy_names = map(sarg -> sarg.name, solverargs)
env_names = map(envarg -> envarg.name, envargs)
nr_pols, nr_envs = length(policy_names), length(env_names)

# upperbounds_init = zeros(nr_envs, nr_pols)
# upperbounds_sampled = zeros( nr_envs, nr_pols)
# return_means = zeros( nr_envs, nr_pols)
# time_solve = zeros( nr_envs, nr_pols)
# time_online = zeros( nr_envs, nr_pols)

models_to_skip_OTIB = ["RockSample (10x10)","Frozen Lake (10x10)","2-out-of-2","3-out-of-3", "Tag"]

# t = time()
# open("run_out_$t.txt", "w") do file
#     write(file, "Policies:  ")
#     for p in policy_names
#         write(file, "$p \t")
#     end
#     write(file, "\n")

for (m_idx,(model, modelargs)) in enumerate(zip(envs, envargs))
    # model = SparseTabularPOMDP(model)
    verbose && println("Testing in $(modelargs.name) environment")
    policies = []
    upperbounds_init = zeros(nr_pols)
    upperbounds_sampled = zeros(nr_pols)
    return_means = zeros(nr_pols)
    time_solve = zeros(nr_pols)
    time_online = zeros(nr_pols)
    
    # Calculate & print model size
    constants = TIB.get_constants(model)
    SAO_probs, SAOs = TIB.get_all_obs_probs(model, constants)
    print(".")
    B, B_idx = TIB.get_belief_set(model, SAOs, constants)
    print(".")
    Br = TIB.get_Br(model, B, constants)
    print(".")
    Data = TIB.TIB_Data(zeros(2,2), B, B_idx, Br, SAO_probs, SAOs, Dict(zip(constants.S, 1:constants.ns)), constants)
    BBao_data = TIB.get_Bbao(model, Data, constants)
    print(".")
    ns, na, no, nb = constants.ns, constants.na, constants.no, length(B)
    nbao = length(BBao_data.Bbao) + length(B)
    verbose && println("|S| = $ns, |A| = $na, |O| = $no, |B| = $nb, |BAO| = $nbao")


    for (s_idx,(solver, solverarg)) in enumerate(zip(solvers, solverargs))
        if solverarg.name == "TIBSolver (worst-case)" && modelargs.name in models_to_skip_OTIB         
            continue
        end 

        verbose && println("\nRunning $(solverarg.name):")
        solver = solver(;solverarg.sargs...)

        # Compute policy & get upper bound
        policy, info = solve_info(solver, model; solverarg.pargs...)
        t = @elapsed begin
            policy, info = POMDPTools.solve_info(solver, model; solverarg.pargs...) 
        end
        # @profile (policy, info = solve_info(solver, model; solverarg.pargs...))
        # pprof(;webport=58699)
        (info isa Nothing) ? val = POMDPs.value(policy, POMDPs.initialstate(model)) : val = info.value        
        verbose && println("Upperbound $val (computed in $t seconds)")
        upperbounds_init[s_idx] = val
        push!(policies, policy)
        time_solve[s_idx] = t

        # Simulate policy & get avg returns
        # verbose && print("Simulating policy...")
        # rs = []
        # for i=1:sims
        #     rtot = 0
        #     for (t,(b,s,a,o,r)) in enumerate(stepthrough(model,policy,"b,s,a,o,r";max_steps=steps))
        #         rtot += POMDPs.discount(model)^(t-1) * r
        #     end
        #     push!(rs,rtot)
        # end
        # rs_avg, rs_min, rs_max = mean(rs), minimum(rs), maximum(rs)
        # verbose && println("Returns: mean = $rs_avg, min = $rs_min, max = $rs_max")
        # return_means[s_idx] = rs_avg

    end

    # Simulate upper bounds on different sampled beliefs & get avg differences
    # if !(model isa RockSample.RockSamplePOMDP)
    #     upperbounds_sampled[:], time_online[:] = sample_avg_accuracy(model, policies)
    # end
    verbose && println("...")

    # data = [upperbounds_init, time_solve, time_online, return_means, upperbounds_sampled]
    # data = [upperbounds_init, time_solve, return_means]
    # names = ["bound:\t\t", "solve time:\t", "comp time:\t", "retuns:\t\t", "avg bound:\t" ]
    # for (d_idx, d) in enumerate(data) 
    #     write(file, names[d_idx])
    #     for p_idx in 1:length(policy_names)
    #         write(file, "$(round(d[p_idx],sigdigits=3))\t")
    #     end
    #     write(file, "\n")
    # end


# end
end
