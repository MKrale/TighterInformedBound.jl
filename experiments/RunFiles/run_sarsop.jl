import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles, ArgParse, JSON
using TighterInformedBound
using Statistics, POMDPModels

##################################################################
#                     Parsing Arguments
##################################################################

s = ArgParseSettings()
@add_arg_table s begin
    "--env"
        help = "The environment to be tested."
        required = true
    "--timeout", "-t"
        help = "Time untill timeout."
        arg_type = Float64
        default = -1.0
    "--precision"
        help = "Precision parameter of SARSOP."
        arg_type= Float64
        default = 1e-2
    "--path"
        help = "File path for data output."
        default = "Data/SarsopTest/"
    "--filename"
        help = "Filename (default: generated automatically)"
        default = ""
    "--solvers"
        help = "Solver to be run. Available options: standard, TIB, ETIB, CTIB, OTIB_pre, OTIB, OATIB_pre, OATIB, MultiTIB_pre, MultiTIB. (default: run all)"
        default = ""
    "--discount"
        help = "Discount factor"
        arg_type = Float64
        default = 0.95
    "--sims"
        help = "Number of samples to simulate performance"
        arg_type = Int 
        default = 0
    "--onlyBs"
        help = "Option to only use heuristic to initialize Bs"
        arg_type = Bool 
        default = false
    "--precompile"
        help = "Option to precomile all code by running at low horizon. Particularly relevant for small environments. (default: true)"
        arg_type = Bool 
        default = true
end

parsed_args = parse_args(ARGS, s)
timeout = parsed_args["timeout"]
env_name = parsed_args["env"]
precision = parsed_args["precision"]
path = parsed_args["path"]
filename = parsed_args["filename"]
solver_name = [parsed_args["solvers"]]
solver_name == [""] && (solver_name = ["standard", "TIB", "ETIB", "CTIB", "OTIB_pre", "OTIB", "OATIB_pre", "OATIB", "MultiTIB_pre", "MultiTIB"])
discount = parsed_args["discount"]
discount_str = string(discount)[3:end]
sims = parsed_args["sims"]
onlyBs = parsed_args["onlyBs"]
onlyBs in [true, "true"] ? (onlyBs = true) : (onlyBs = false)
precompile = parsed_args["precompile"]

if timeout == -1.0
    timeout = 3600.0
	discount == 0.95 && (timeout = 3600.0)
	discount == 0.99 && (timeout = 3600.0)
end

##################################################################
#                       Defining Solvers 
##################################################################

solvers, precomp_solvers, solverargs = [], [], []
include("../Sarsop_altered/NativeSARSOP.jl")
import .NativeSARSOP_alt

# Heuristic settings. h_timeout bounds the time spent iterating the heuristic (it is part of the SARSOP budget `timeout`).
# All heuristics iterate until converged to h_precision (or until h_timeout).
h_precision, h_timeout = 1e-3, 1200.0

# Solvers used to initialize the Q-values of the TIB-style heuristics (default precision, but limited to the same time budget)
init_FIB = NativeSARSOP_alt.FIBSolver_alt(max_time=h_timeout)
init_STIB = NativeSARSOP_alt.STIBSolver(max_time=h_timeout, precomp_solver=init_FIB)

# Settings shared by all TIB-style heuristics, and by all weighted heuristics that recompute weights dynamically.
h_args = (precision=h_precision, max_time=h_timeout)
h_init_args = (h_args..., precomp_solver=init_STIB)
h_dynamic_args = (dynamic_recompute=true, dynamic_precision=h_precision)

# Small heuristic solvers used only to precompile (same solver types as in the actual runs).
pre_STIB = NativeSARSOP_alt.STIBSolver(max_iterations=2, precomp_solver=NativeSARSOP_alt.FIBSolver_alt(max_iterations=2))
pre_weighted_args = (max_iterations=2, max_recomputes=0, precomp_solver=pre_STIB)

"""Adds a SARSOP run with heuristic h_solver, and a precompile run with heuristic precomp_h_solver."""
function add_sarsop_run!(name, h_solver, precomp_h_solver)
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    push!(solverargs, (name=name, sargs=(precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver, use_only_Bs=onlyBs), pargs=()))
    push!(precomp_solvers, (sargs=(max_its=2, verbose=false, heuristic_solver=precomp_h_solver), pargs=()))
end

if "standard" in solver_name
    add_sarsop_run!("SARSOP",
        NativeSARSOP_alt.FIBSolver_alt(;h_args...),
        NativeSARSOP_alt.FIBSolver_alt(max_iterations=2))
end
if "TIB" in solver_name
    add_sarsop_run!("TIB-SARSOP",
        NativeSARSOP_alt.STIBSolver(;h_args..., precomp_solver=init_FIB),
        pre_STIB)
end
if "ETIB" in solver_name
    add_sarsop_run!("ETIB-SARSOP",
        NativeSARSOP_alt.ETIBSolver(;h_init_args...),
        NativeSARSOP_alt.ETIBSolver(max_iterations=2, precomp_solver=pre_STIB))
end
if "CTIB" in solver_name
    add_sarsop_run!("CTIB-SARSOP",
        NativeSARSOP_alt.ICTIBSolver(;h_init_args...),
        NativeSARSOP_alt.ICTIBSolver(max_iterations=2, precomp_solver=pre_STIB))
end
if "OTIB_pre" in solver_name
    add_sarsop_run!("OTIB_pre-SARSOP",
        NativeSARSOP_alt.OTIBSolver(;h_init_args..., dynamic_recompute=false),
        NativeSARSOP_alt.OTIBSolver(;pre_weighted_args...))
end
if "OTIB" in solver_name
    add_sarsop_run!("OTIB-SARSOP",
        NativeSARSOP_alt.OTIBSolver(;h_init_args..., h_dynamic_args...),
        NativeSARSOP_alt.OTIBSolver(;pre_weighted_args...))
end
if "OATIB_pre" in solver_name
    add_sarsop_run!("OATIB_pre-SARSOP",
        NativeSARSOP_alt.OATIBSolver(;h_init_args..., dynamic_recompute=false),
        NativeSARSOP_alt.OATIBSolver(;pre_weighted_args...))
end
if "OATIB" in solver_name
    add_sarsop_run!("OATIB-SARSOP",
        NativeSARSOP_alt.OATIBSolver(;h_init_args..., h_dynamic_args...),
        NativeSARSOP_alt.OATIBSolver(;pre_weighted_args...))
end
if "MultiTIB_pre" in solver_name
    add_sarsop_run!("MultiTIB_pre-SARSOP",
        NativeSARSOP_alt.MultiTIBSolver(;h_init_args..., dynamic_recompute=false),
        NativeSARSOP_alt.MultiTIBSolver(;pre_weighted_args...))
end
if "MultiTIB" in solver_name
    add_sarsop_run!("MultiTIB-SARSOP",
        NativeSARSOP_alt.MultiTIBSolver(;h_init_args..., h_dynamic_args...),
        NativeSARSOP_alt.MultiTIBSolver(;pre_weighted_args...))
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


ubs, lbs = Tuple{Vector{Float64}, Vector{Float64}}[], Tuple{Vector{Float64}, Vector{Float64}}[]
# env = SparseTabularPOMDP(env) #breaks RockSample...

verbose = true
for (env, env_arg) in zip(envs, envargs)
    for (i, (solver, solverarg)) in enumerate(zip(solvers, solverargs))
        
        # Precomputation:
        if precompile
            precomp_solver = solver(;precomp_solvers[i].sargs...)
            _p, _i = POMDPTools.solve_info(precomp_solver, env; precomp_solvers[i].pargs...)
        end
        
        solver = solver(;solverarg.sargs...)
        policy, info = POMDPTools.solve_info(solver, env; solverarg.pargs...)

        rs_avg, rs_sigma = nothing, nothing
        if sims > 0
            max_steps = Int(ceil(log(discount, 1e-5)))
            rs = []
            for i=1:sims
                rtot = 0
                for (t,(s,a,o,r)) in enumerate(stepthrough(env,policy,"s,a,o,r", max_steps=max_steps))
                    rtot += discount^(t-1) * r
                end
                push!(rs,rtot)
            end
            rs_avg, rs_sigma = mean(rs), std(rs)
        end
        
        data_dict = Dict(
            "env" => env_name,
            "env_full" => env_arg.name,
            "solver" => solverarg.name,
            "timeout" => info.timeout,
            "runtime" => last(info.times),
            "final_ub" => last(info.ubs),
            "final_lw" => last(info.lbs),
            "ubs" => info.ubs,
            "lbs" => info.lbs,
            "times" => info.times,
            "sim_r" => rs_avg,
            "sim_rsigma" => rs_sigma
        )
        verbose && println("In $(env_name), $(solverarg.name) found bounds ($(last(info.lbs)), $(last(info.ubs))) in $(last(info.times))s.")

        json_str = JSON.json(data_dict)
        if filename == ""
            onlyBs && (thisfilename =  path * "Sarsoptest_$(env_name)_$(solverarg.name)_t$(Int(ceil(timeout)))_d$(discount_str)_OnlyBs.json")
            !onlyBs && (thisfilename =  path * "Sarsoptest_$(env_name)_$(solverarg.name)_t$(Int(ceil(timeout)))_d$discount_str.json")
        else
            thisfilename = path * filename * solverarg.name
        end
        open(thisfilename, "w") do file
            write(file, json_str)
        end
    end
end






