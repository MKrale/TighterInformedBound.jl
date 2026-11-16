# File used to find the upper bounds as computed by different algorithms on different environments.

import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles, ArgParse, JSON
using Statistics, POMDPModels

using TighterInformedBound
include("../Sarsop_altered/NativeSARSOP.jl")
import .NativeSARSOP_alt

##################################################################
#                     Parsing Arguments
##################################################################

s = ArgParseSettings()
@add_arg_table s begin
    "--env"
        help = "The environment to be tested."
        required = true
    "--precision"
        help = "Precision parameter of SARSOP."
        arg_type = Float64
        default = 0.0
    "--timeout", "-t"
        help = "Time untill timeout."
        arg_type = Float64
        default = -1.0
    "--path"
        help = "File path for data output."
        default = "Data/UpperBounds/"
    "--filename"
        help = "Filename (default: generated automatically)"
        default = ""
    "--solvers"
        help = "Solver to be run. Available options: QMDP, FIB, TIB, ETIB, CTIB, CTIB_once, OTIB_pre, OTIB, OATIB_pre, OATIB, MultiTIB_pre, MultiTIB, SawTIB_pre, SawTIB, ISawTIB_pre, ISawTIB, SARSOP, TIBSARSOP, ETIBSARSOP. (default: TIB, ETIB, CTIB, OTIB_pre, OTIB, OATIB_pre, OATIB, MultiTIB_pre, MultiTIB, FIB, QMDP, SARSOP)"
        default = "All"
    "--discount"
        help = "Discount factor"
        arg_type = Float64
        default = 0.95
    "--precompile"
        help = "Option to precomile all code by running at low horizon. Particularly relevant for small environments. (default: true)"
        arg_type = Bool 
        default = true
end

parsed_args = parse_args(ARGS, s)
env_name = parsed_args["env"]
timeout = parsed_args["timeout"]
path = parsed_args["path"]
filename = parsed_args["filename"]
solver_names = [parsed_args["solvers"]]
solver_names == ["All"] && (solver_names = ["TIB", "ETIB", "CTIB", "OTIB_pre", "OTIB", "OATIB_pre", "OATIB", "MultiTIB_pre", "MultiTIB", "FIB", "QMDP", "SARSOP"])

discount = parsed_args["discount"]
discount_str = string(discount)[3:end]
precompile = parsed_args["precompile"]
heuristicprecision = parsed_args["precision"]

if timeout == -1.0
	timeout = 1200.0
	discount == 0.95 && (timeout = 1200.0)
	discount == 0.99 && (timeout = 1200.0)
end

##################################################################
#                       Defining Solvers 
##################################################################

# We define both a vector with the solvers to run, as well as their arguments.
# Moreover, we add a vector with the arguments to use in the precomp run (i.e., using only a few iterations)
solvers, solverargs, precomp_solverargs = [], [], []
SARSOPprecision = 1e-3 # Hardcoded
if heuristicprecision == 0.0 # i.e. precision is not set
    # Precision picked based on discount. All bounds iterate until converged to this precision (or until timeout).
    heuristicprecision = 1e-4
    discount == 0.95 && (heuristicprecision = 1e-3)
    discount == 0.99 && (heuristicprecision = 1e-4)
end

timeout_sarsop = 1200.0

# Solvers used to initialize the Q-values of the TIB-style bounds (default precision, but limited to the same time budget)
init_FIB = FIBSolver_alt(max_time=timeout)
init_STIB = STIBSolver(max_time=timeout, precomp_solver=init_FIB)

if "FIB" in solver_names
    push!(solvers, FIBSolver_alt)
    push!(solverargs, (name="FIB", sargs=(precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "TIB" in solver_names
    push!(solvers, STIBSolver)
    push!(solverargs, (name="TIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_FIB), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=3, precomp_solver=FIBSolver_alt(max_iterations=3)), pargs=()))
end
if "ETIB" in solver_names
    push!(solvers, ETIBSolver)
    push!(solverargs, (name="ETIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))    
end
if "CTIB_once" in solver_names
    push!(solvers, CTIBSolver)
    push!(solverargs, (name="CTIB_once", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))    
end
if "CTIB" in solver_names
    push!(solvers, ICTIBSolver)
    push!(solverargs, (name="CTIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))    
end
if "OTIB_pre" in solver_names
    push!(solvers, OTIBSolver)
    push!(solverargs, (name="OTIB_pre", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "OTIB" in solver_names
    push!(solvers, OTIBSolver)
    push!(solverargs, (name="OTIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=true, dynamic_precision=heuristicprecision), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "OATIB_pre" in solver_names
    push!(solvers, OATIBSolver)
    push!(solverargs, (name="OATIB_pre", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "OATIB" in solver_names
    push!(solvers, OATIBSolver)
    push!(solverargs, (name="OATIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=true, dynamic_precision=heuristicprecision), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "SawTIB_pre" in solver_names
    push!(solvers, SawTIBSolver)
    push!(solverargs, (name="SawTIB_pre", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "SawTIB" in solver_names
    push!(solvers, SawTIBSolver)
    push!(solverargs, (name="SawTIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=true, dynamic_precision=heuristicprecision), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "ISawTIB_pre" in solver_names
    push!(solvers, ISawTIBSolver)
    push!(solverargs, (name="ISawTIB_pre", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "ISawTIB" in solver_names
    push!(solvers, ISawTIBSolver)
    push!(solverargs, (name="ISawTIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=true, dynamic_precision=heuristicprecision), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "MultiTIB_pre" in solver_names
    push!(solvers, MultiTIBSolver)
    push!(solverargs, (name="MultiTIB_pre", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes = 0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "MultiTIB" in solver_names
    push!(solvers, MultiTIBSolver)
    push!(solverargs, (name="MultiTIB", sargs=(precision=heuristicprecision, max_time=timeout, precomp_solver=init_STIB, dynamic_recompute=true, dynamic_precision=heuristicprecision), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes = 0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "SARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.FIBSolver_alt(precision=heuristicprecision, max_time=timeout)
    push!(solverargs, (name="SARSOP", sargs=(precision=SARSOPprecision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
    precomp_h_solver = NativeSARSOP_alt.FIBSolver_alt(max_iterations=2)
    push!(precomp_solverargs, ( sargs=(max_its=1, verbose=false, heuristic_solver=precomp_h_solver), pargs=()))
end
if "TIBSARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.STIBSolver(precision=1e-5)
    push!(solverargs, (name="TIB-SARSOP", sargs=( precision=SARSOPprecision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if "ETIBSARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.ETIBSolver(precision=1e-5)
    push!(solverargs, (name="ETIB-SARSOP", sargs=( precision=SARSOPprecision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if "QMDP" in solver_names
    push!(solvers, QMDPSolver_alt)
    push!(solverargs, (name="QMDP", sargs=(precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end

isempty(solvers) && println("Warning: no solver selected!")

##################################################################
#                       Selecting env 
##################################################################

include("./run_util.jl")
envs, envargs = get_envs(env_name)

##################################################################
#                           Run 
##################################################################

sims, steps = 1_000, 1_000

policy_names = map(sarg -> sarg.name, solverargs)
env_names = map(envarg -> envarg.name, envargs)
nr_pols, nr_envs = length(policy_names), length(env_names)

upperbounds_init = zeros(nr_envs, nr_pols)
upperbounds_sampled = zeros( nr_envs, nr_pols)
return_means = zeros( nr_envs, nr_pols)
time_solve = zeros( nr_envs, nr_pols)
time_online = zeros( nr_envs, nr_pols)

verbose = true

for (m_idx,(model, modelargs)) in enumerate(zip(envs, envargs))   
    for (s_idx,(solver, solverarg)) in enumerate(zip(solvers, solverargs))
        env_data = Dict() # Model sizes are collected separately (see RunFiles/run_modelsizes.jl)

        # Precompile
        if precompile
            thissolver = solver(;precomp_solverargs[s_idx].sargs...)
            _t = @elapsed begin
                policy, info = POMDPTools.solve_info(thissolver, model; precomp_solverargs[s_idx].pargs...) 
            end
        end

        # Compute policy & get upper bound
        thissolver = solver(;solverarg.sargs...)
	GC.gc()
        t = @elapsed begin
            policy, info = POMDPTools.solve_info(thissolver, model; solverarg.pargs...) 
        end
        if !isdefined(info, :ub) || !isdefined(info, :lb)
            ub = POMDPs.value(policy, POMDPs.initialstate(model))
            lb = -1
        else
            ub = info.ub
            lb =  info.lb
        end       

        ### Policy simulation (very slow, so not used)
        #rs = []
        #t0_sims = time()
        #for i=1:sims
        #    rtot = 0
        #    for (t,(b,s,a,o,r)) in enumerate(stepthrough(model,policy,"b,s,a,o,r";max_steps=steps))
        #        rtot += POMDPs.discount(model)^(t-1) * r
        #    end
        #    push!(rs,rtot)
        #end
        #t_sims = time() - t0_sims
        #rs_avg, rs_min, rs_max = mean(rs), minimum(rs), maximum(rs)
	    rs_avg, rs_min, rs_max = -1.0, -1.0, -1.0   # Comment out when doing simulations
        t_sims = -1.0                               # Comment out when doing simulations

	    ### Writing data to files
        data_dict = Dict(
            "env" => env_name,
            "env_full" => modelargs.name,
            "env_data" => env_data,
            "solver" => solverarg.name,
            # Solving data
            "solvetime" => t,
            "ub" => ub,
            "lb" => lb,
            # Simulation data
            "simtime" => t_sims,
            "ravg" => rs_avg,
            "additional_info" => info
        )
        json_str = JSON.json(data_dict)
        verbose && println("In $(env_name), $(solverarg.name) found bound $ub in $(t)s.")
        if filename == ""
		thisfilename =  path * "UpperBoundTest_$(env_name)_$(solverarg.name)_d$(discount_str).json"
        else
            thisfilename = path * filename * solverarg.name
        end
        open(thisfilename, "w") do file
            write(file, json_str)
        end
    end
end
