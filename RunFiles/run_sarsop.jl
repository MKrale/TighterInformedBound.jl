import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles, ArgParse, JSON
include("../TIB/TIB.jl")
using .TIB
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
        help = "Solver to be run. Availble options: standard, TIB, ETIB, OTIB. (default: run all except OTIB)"
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
solver_name == [""] && (solver_name = ["standard","TIB","ETIB","CTIB","MultiTIB"])
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

h_iterations, h_precision = 10_000, 1e-3
discount == 0.95 && (h_iterations = 250; h_precision = 1e-3; h_timeout = 1200.0)
discount == 0.99 && (h_iterations = 1000; h_precision = 1e-3; h_timeout = 1200.0)

if "standard" in solver_name
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.FIBSolver_alt(max_iterations=(h_iterations*4), precision=h_precision)
    push!(solverargs, (name="SARSOP", sargs=(precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver), pargs=()))

    precomp_h_solver = NativeSARSOP_alt.FIBSolver_alt(max_iterations=2)
    push!(precomp_solvers, (sargs=(max_its = 2, verbose=false, heuristic_solver=precomp_h_solver),pargs=()))
end
if "TIB" in solver_name
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.STIBSolver(max_iterations=h_iterations, precision=h_precision)
    push!(solverargs, (name="TIB-SARSOP", sargs=( precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver, use_only_Bs=onlyBs), pargs=()))

    precomp_h_solver = NativeSARSOP_alt.STIBSolver(max_iterations=2)
    push!(precomp_solvers, (sargs=(max_its = 2, verbose=false, heuristic_solver=precomp_h_solver),pargs=()))
end
if "ETIB" in solver_name
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.ETIBSolver(max_iterations=h_iterations, precision=h_precision)
    push!(solverargs, (name="ETIB-SARSOP", sargs=( precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver, use_only_Bs=onlyBs), pargs=()))

    precomp_h_solver = NativeSARSOP_alt.ETIBSolver(max_iterations=2)
    push!(precomp_solvers, (sargs=(max_its = 2, verbose=false, heuristic_solver=precomp_h_solver),pargs=()))
end
if "CTIB" in solver_name
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.ICTIBSolver(max_iterations=h_iterations, precision=h_precision)
    push!(solverargs, (name="CTIB-SARSOP", sargs=( precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver, use_only_Bs=onlyBs), pargs=()))

    precomp_h_solver = NativeSARSOP_alt.ETIBSolver(max_iterations=2)
    push!(precomp_solvers, (sargs=(max_its = 2, verbose=false, heuristic_solver=precomp_h_solver),pargs=()))
end
if "OTIB" in solver_name
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.OTIBSolver(max_iterations=h_iterations, precision=h_precision, dynamic_recompute=true, dynamic_precision=h_precision, max_recomputes=100)
    push!(solverargs, (name="OTIB-SARSOP", sargs=( precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver, use_only_Bs=onlyBs), pargs=()))

    precomp_h_solver = NativeSARSOP_alt.OTIBSolver(max_iterations=2, precomp_solver=NativeSARSOP_alt.STIBSolver(max_iterations=2))
    push!(precomp_solvers, (sargs=(max_its = 2, verbose=false, heuristic_solver=precomp_h_solver),pargs=()))
end
if "MultiTIB" in solver_name
    push!(solvers, NativeSARSOP_alt.SARSOPSolver)
    h_solver = NativeSARSOP_alt.MultiTIBSolver(max_iterations=h_iterations, precision=h_precision)
    push!(solverargs, (name="MultiTIB-SARSOP", sargs=( precision=precision, max_time=timeout, verbose=false, heuristic_solver=h_solver, use_only_Bs=onlyBs), pargs=()))

    precomp_h_solver = NativeSARSOP_alt.ETIBSolver(max_iterations=2)
    push!(precomp_solvers, (sargs=(max_its = 2, verbose=false, heuristic_solver=precomp_h_solver),pargs=()))
end

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






