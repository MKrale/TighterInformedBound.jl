# File used to find the upper bounds as computed by different algorithms on different environments.

import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles, ArgParse, JSON
using Statistics, POMDPModels
using Profile, FlameGraphs, ProfileSVG
using InteractiveUtils

include("../TIB/TIB.jl")
using .TIB
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
        help = "Solver to be run. Availble options: FIB, TIB, ETIB, OTIB, SARSOP, TIBSARSOP, ETIBSARSOP. (default: run all but TIBSARSOP & ETIBSARSOP)"
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
solver_names == ["All"] && (solver_names = ["TIB", "ETIB", "CTIB", "OTIB_pre", "OTIB", "MultiTIB_pre", "MultiTIB", "FIB", "QMDP", "SARSOP"])

discount = parsed_args["discount"]
discount_str = string(discount)[3:end]
precompile = parsed_args["precompile"]
heuristicprecision = parsed_args["precision"]

if timeout == -1.0
	discount == 0.95 && (timeout = 1200.0)
	discount == 0.99 && (timeout = 1200.0)
end

##################################################################
#                       Defining Solvers 
##################################################################

# We define both a vector with the solvers to run, as well as their arguments.
# Moreover, we add a vector with the arguments to use in the precomp run (i.e., using only one iteration)
solvers, solverargs, precomp_solverargs = [], [], []
SARSOPprecision = 1e-3 # Hardcoded
heuristicsteps = 1_000
if heuristicprecision == 0.0 # i.e. precision is not set
    # Precision & #steps picked based on discount
    heuristicprecision = 1e-4
    discount == 0.95 && (heuristicprecision = 1e-3;  heuristicsteps = 250)
    discount == 0.99 && (heuristicprecision = 1e-4;  heuristicsteps = 1_000)
end

timeout_sarsop = 1200.0

if "FIB" in solver_names
    push!(solvers, FIBSolver_alt)
    push!(solverargs, (name="FIB", sargs=(max_iterations=heuristicsteps*4, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2,), pargs=()))
end
if "TIB" in solver_names
    push!(solvers, STIBSolver)
    push!(solverargs, (name="TIB", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=3, precomp_solver=FIBSolver_alt(max_iterations=3)), pargs=()))
end
if "ETIB" in solver_names
    push!(solvers, ETIBSolver)
    push!(solverargs, (name="ETIB", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))    
end
if "CTIB_once" in solver_names
    push!(solvers, CTIBSolver)
    push!(solverargs, (name="CTIB_once", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))    
end
if "CTIB" in solver_names
    push!(solvers, ICTIBSolver)
    push!(solverargs, (name="CTIB", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))    
end
if "OTIB_pre" in solver_names
    push!(solvers, OTIBSolver)
    push!(solverargs, (name="OTIB_pre", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "OTIB" in solver_names
    push!(solvers, OTIBSolver)
    push!(solverargs, (name="OTIB", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=true, dynamic_precision=heuristicprecision, max_recomputes=100), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "SawTIB_pre" in solver_names
    push!(solvers, SawTIBSolver)
    push!(solverargs, (name="SawTIB_pre", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "SawTIB" in solver_names
    push!(solvers, SawTIBSolver)
    push!(solverargs, (name="SawTIB", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=true, dynamic_precision=heuristicprecision, max_recomputes=100), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "ISawTIB_pre" in solver_names
    push!(solvers, ISawTIBSolver)
    push!(solverargs, (name="ISawTIB_pre", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "ISawTIB" in solver_names
    push!(solvers, ISawTIBSolver)
    push!(solverargs, (name="ISawTIB", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=true, dynamic_precision=heuristicprecision, max_recomputes=100), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes=0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "MultiTIB_pre" in solver_names
    push!(solvers, MultiTIBSolver)
    push!(solverargs, (name="MultiTIB_pre", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=false), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes = 0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "MultiTIB" in solver_names
    push!(solvers, MultiTIBSolver)
    push!(solverargs, (name="MultiTIB", sargs=(max_iterations=heuristicsteps, precision=heuristicprecision, max_time=timeout, dynamic_recompute=true, dynamic_precision=heuristicprecision, max_recomputes=100), pargs=(), get_Q0=true))
    push!(precomp_solverargs, ( sargs=(max_iterations=2, max_recomputes = 0, precomp_solver=STIBSolver(max_iterations=2)), pargs=()))
end
if "SARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.FIBSolver_alt(max_iterations=heuristicsteps*4, precision=heuristicprecision)
    push!(solverargs, (name="SARSOP", sargs=(precision=SARSOPprecision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
    precomp_h_solver = NativeSARSOP_alt.FIBSolver_alt(max_iterations=2)
    push!(precomp_solverargs, ( sargs=(max_its=1, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if "TIBSARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.STIBSolver(max_iterations=250, precision=1e-5)
    push!(solverargs, (name="TIB-SARSOP", sargs=( precision=SARSOPprecision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if "ETIBSARSOP" in solver_names
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.ETIBSolver(max_iterations=250, precision=1e-5)
    push!(solverargs, (name="ETIB-SARSOP", sargs=( precision=precision, max_time=timeout_sarsop, verbose=false, heuristic_solver=h_solver), pargs=()))
end
if "QMDP" in solver_names
    push!(solvers, QMDPSolver_alt)
    push!(solverargs, (name="QMDP", sargs=(max_iterations=heuristicsteps*10, precision=heuristicprecision),pargs=(), get_Q0=true))
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
        # print("huh?")
        print_model_info(model)
        env_data = Dict() # Comment out when you want to get env info

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
            # policy, info = POMDPTools.solve_info(thissolver, model; solverarg.pargs...) 
            @profile policy, info = POMDPTools.solve_info(thissolver, model; solverarg.pargs...) 
        end
        if !isdefined(info, :ub) || !isdefined(info, :lb)
            ub = POMDPs.value(policy, POMDPs.initialstate(model))
            lb = -1
        else
            ub = info.ub
            lb =  info.lb
        end       

        fg = flamegraph(Profile.fetch(); norepl=true, combine=true)
        ProfileSVG.save("flamegraph.svg", fg; width=3600, fontsize=10, maxdepth=40, maxframes=10_000)

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
