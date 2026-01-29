include("../Environments/K-out-of-N.jl"); using .K_out_of_Ns
include("../Environments/Sparse_models/SparseModels.jl"); using .SparseModels

include("../Environments/ABCModel.jl"); using .ABCModel

using TagPOMDPProblem
Base.:(==)(s1::TagState, s2::TagState) = s1.r_pos == s2.r_pos && s1.t_pos == s2.t_pos
Base.isless(s1::TagState, s2::TagState) = s1.r_pos == s2.r_pos ? (s1.t_pos < s2.t_pos) : (s1.r_pos < s2.r_pos)
Base.:(<)(s1::TagState, s2::TagState) = isless(s1,s2)

import RockSample
# This env is very difficult to work with for some reason...
POMDPs.states(M::RockSample.RockSamplePOMDP) = map(si -> RockSample.state_from_index(M,si), 1:length(M))
POMDPs.discount(M::RockSample.RockSamplePOMDP) = discount

using DroneSurveillance

function get_envs(env_name)
    envs, envargs = [], []

    ###ABC
    if env_name == "ABC"
        abcmodel = ABC(discount=discount)
        push!(envs, SparseTabularPOMDP(abcmodel))
        push!(envargs, (name="ABCModel",))
        ### Tiger
    end
    if env_name == "Tiger"
        tiger = POMDPModels.TigerPOMDP()
        tiger.discount_factor = discount
        push!(envs, SparseTabularPOMDP(tiger))
        push!(envargs, (name="Tiger",))
        ### RockSample
    end
    if env_name == "RockSample5"
        map_size, rock_pos = (5,5), [(1,1), (3,3), (4,4)] # Default
        rocksamplesmall = SparseTabularPOMDP(RockSample.RockSamplePOMDP(map_size, rock_pos))
        rocksamplesmall = sparseTabularPOMDPFiniteReward(rocksamplesmall)
        push!(envargs, (name="RockSample ()",))
        push!(envs, rocksamplesmall)
    end
    if env_name == "RockSample7"
        map_size, rock_pos = (7,7), [(1,2), (2,6), (3,3), (3,4), (4,7),(6,1),(6,4),(7,3)] # HSVI setting!
        rocksamplelarge = SparseTabularPOMDP(RockSample.RockSamplePOMDP(map_size, rock_pos))
        rocksamplelarge = sparseTabularPOMDPFiniteReward(rocksamplelarge)
        push!(envargs, (name="RockSample (7,8)",))
        push!(envs, rocksamplelarge)
    end
    if env_name == "K-out-of-N2"
        # ### K-out-of-N
        k_model2 = K_out_of_N(N=2, K=2, discount=discount)
        push!(envs, SparseTabularPOMDP(k_model2))
        push!(envargs, (name="K-out-of-N (2)",))
    end
    if env_name == "K-out-of-N3"
        k_model3 = K_out_of_N(N=3, K=3, discount=discount)
        push!(envs, SparseTabularPOMDP(k_model3))
        push!(envargs, (name="K-out-of-N (3)",))
    end
    if env_name == "HeavenOrHell"
        model = HeavenOrHell(size=10)
        push!(envs, SparseTabularPOMDP(model))
        push!(envargs, (name="HeavenOrHell",))
    end
    if env_name == "SparseHallway1"
        hallway1 = SparseHallway1(discount=discount)
        push!(envs, hallway1)
        push!(envargs, (name="SparseHallway1",))
    end
    if env_name == "SparseHallway2"
        hallway2 = SparseHallway2(discount=discount)
        push!(envs, hallway2)
        push!(envargs, (name="SparseHallway2",))
    end
    if env_name == "SparseTigerGrid"
        tigergrid = SparseTigerGrid(discount=discount)
        push!(envs, tigergrid)
        push!(envargs, (name="TigerGrid",))
    end
    if env_name == "aloha10"
        aloha10 = Sparse_aloha10(discount=discount)
        push!(envs, aloha10)
        push!(envargs, (name="aloha10",))
    end
    if env_name == "aloha30"
        aloha30 = Sparse_aloha30(discount=discount)
        push!(envs, aloha30)
        push!(envargs, (name="aloha30",))
    end
    if env_name == "baseball"
        baseball = Sparse_baseball(discount=discount)
        push!(envs, baseball)
        push!(envargs, (name="baseball",))
    end
    if env_name == "cit"
        cit = Sparse_cit(discount=discount)
        push!(envs, cit)
        push!(envargs, (name="cit",))
    end
    if env_name == "fourth"
        fourth = Sparse_fourth(discount=discount)
        push!(envs, fourth)
        push!(envargs, (name="fourth",))
    end
    if env_name == "mit"
        mit = Sparse_mit(discount=discount)
        push!(envs, mit)
        push!(envargs, (name="mit",))
    end
    if env_name == "pentagon"
        pentagon = Sparse_pentagon(discount=discount)
        push!(envs, pentagon)
        push!(envargs, (name="pentagon",))
    end
    if env_name == "sunysb"
        sunysb = Sparse_sunysb(discount=discount)
        push!(envs, sunysb)
        push!(envargs, (name="sunysb",))
    end
    if env_name == "iff"
        model = Sparse_iff(discount=discount)
        push!(envs, model)
        push!(envargs, (name="iff",))
    end
    if env_name == "grid"
        Grid = Sparse_Grid(discount=discount)
        push!(envs, Grid)
        push!(envargs, (name="grid",))
    end
    if env_name == "Tag"
        ### Tag
        tag = TagPOMDPProblem.TagPOMDP(discount_factor=discount)
        tag = SparseTabularPOMDP(tag)
        push!(envs, tag)
        push!(envargs, (name="Tag",))
    end
    if env_name == "DroneSurveilance"
        drone = DroneSurveillancePOMDP()
        drone = SparseTabularPOMDP(drone)
        push!(envs, drone)
        push!(envargs, (name="DroneSurveilance",))
    end
    isempty(envs) && println("Warning: no environment selected!")

    return envs, envargs
end