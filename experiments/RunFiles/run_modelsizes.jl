# File used to collect the sizes of all environments: |S|, |A|, |O|, and the number of beliefs reachable in one, two and
# three steps from the state-beliefs (|B1| = |Bsao|, |B2| = |Bbao|, |B3|). Computing |B3| is expensive for large models,
# which is why this is kept separate from the other experiments.

import POMDPs, POMDPTools
using POMDPs
using POMDPTools, POMDPFiles, ArgParse, JSON
using POMDPModels
using TighterInformedBound

##################################################################
#                     Parsing Arguments
##################################################################

s = ArgParseSettings()
@add_arg_table s begin
    "--env"
        help = "The environment to be tested."
        required = true
    "--path"
        help = "File path for data output."
        default = "Data/ModelSizes/"
    "--discount"
        help = "Discount factor (does not influence the sizes, but is required for constructing some environments)"
        arg_type = Float64
        default = 0.95
    "--threestep"
        help = "Option to also compute the number of beliefs reachable in three steps (|B3|), which is expensive."
        arg_type = Bool
        default = true
end

parsed_args = parse_args(ARGS, s)
env_name = parsed_args["env"]
path = parsed_args["path"]
discount = parsed_args["discount"]
threestep = parsed_args["threestep"]

include("./run_util.jl")
envs, envargs = get_envs(env_name)

##################################################################
#                           Run
##################################################################

mkpath(path)
for (model, modelargs) in zip(envs, envargs)
    t0 = time()
    Data = TighterInformedBound.get_TIB_Data(model)
    t_B1 = time() - t0
    Bbao_data = TighterInformedBound.get_Bbao(model, Data)
    t_B2 = time() - t0 - t_B1
    nb3, t_B3 = -1, -1.0
    if threestep
        nb3 = length(TighterInformedBound.get_three_step_beliefs(model, Data, Bbao_data))
        t_B3 = time() - t0 - t_B1 - t_B2
    end

    ns, na, no = Data.constants.ns, Data.constants.na, Data.constants.no
    nb1, nb2 = length(Data.B), length(Bbao_data.Bbao)
    data_dict = Dict(
        "env" => env_name,
        "env_full" => modelargs.name,
        "ns" => ns, "na" => na, "no" => no,
        "nb" => nb1,        # |B1| (= |Bsao|)
        "nbao" => nb2,      # |B2| (= |Bbao|), excluding beliefs already in B1
        "nb3" => nb3,       # |B3|, excluding beliefs already in B1 or B2 (-1 if not computed)
        "time_B1" => t_B1, "time_B2" => t_B2, "time_B3" => t_B3,
    )
    println("In $(env_name): |S| = $ns, |A| = $na, |O| = $no, |B1| = $nb1, |B2| = $nb2, |B3| = $nb3")
    println(" & $ns & $na & $no & $nb1 & $nb2 & $nb3 ")

    open(path * "ModelSizes_$(env_name).json", "w") do file
        write(file, JSON.json(data_dict))
    end
end
