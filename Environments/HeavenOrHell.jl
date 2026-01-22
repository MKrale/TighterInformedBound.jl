# A variant of the Heaven or Hell environment from Baziunas & Boutilier (2004): https://cdn.aaai.org/AAAI/2004/AAAI04-109.pdf
# We slightly simplify the statespace: the agent find heaven or hell by moving n steps up, or finds a sign n steps down.
# Rewards are picked so that, with no observation error, the optimal policy goes to the sign.

module HeavenOrHellModel
using POMDPs, POMDPTools, Distributions
export HeavenOrHell

function SafeSparseCat(vals, probs)
    vals_new = unique(vals)
    probs_new = []
    for v in vals_new
        push!(probs_new, sum(probs[findall(isequal(v), vals )]))
    end
    return SparseCat(vals_new, probs_new)
end

UP, RIGHT, DOWN, LEFT = 1,2,3,4

@kwdef mutable struct HeavenOrHell <: POMDP{Tuple{Int,Int}, Int, Int}
    size::Int = 10
    slipchance::Float64 = 0.0
    obserror::Float64 = 0.2
    discount::Float64 = 0.99
    # rewards::NTuple{3,Float64} = (0.0, -15.0, -1.0)
end
endcorridor(M::HeavenOrHell) = 2*M.size + 1
correctsign(M::HeavenOrHell) = 2*M.size + 2
wrongsign(M::HeavenOrHell) = 2*M.size + 3

POMDPs.states(M::HeavenOrHell) = vec([(x,y) for x in 0:wrongsign(M), y in 1:2])
POMDPs.statetype(M::HeavenOrHell) = Tuple{Int,Int}
POMDPs.stateindex(M::HeavenOrHell, s) = findfirst(isequal(s), states(M))
# POMDPs.stateindex(M::HeavenOrHell, s) = (first(s)+1) + (maxstate(M)+1) * (last(s)-1)
POMDPs.actions(M::HeavenOrHell) = [UP,DOWN, LEFT, RIGHT]
POMDPs.actiontype(M::HeavenOrHell) = Int 
POMDPs.actionindex(M::HeavenOrHell, a) = findfirst(isequal(a), actions(M))
POMDPs.observations(M::HeavenOrHell) = 1:endcorridor(M)
POMDPs.obstype(M::HeavenOrHell) = Int
POMDPs.obsindex(M::HeavenOrHell, o) = o
POMDPs.discount(M::HeavenOrHell) = M.discount
POMDPs.initialstate(M::HeavenOrHell) = SparseCat([(M.size+1,1), (M.size+1,2)], [0.5,0.5])
# POMDPs.initialstate(M::HeavenOrHell) = SparseCat([(6,1), (6,2)], [0.5,0.5])

POMDPs.isterminal(M::HeavenOrHell, s) = first(s) == 0

function POMDPs.transition(M::HeavenOrHell, s, a)
    pos, goal = s
    # nextpos = pos
    pos == 0 && return Deterministic(s)
    # Picking heaven or hell:
    if pos == 1 && a in [LEFT,RIGHT]
        return Deterministic((0,goal))
    # Viewing sign:
    elseif pos == endcorridor(M) && a == DOWN
        return SparseCat([(endcorridor(M),goal),(correctsign(M), goal), (wrongsign(M), goal)], [M.slipchance, (1-M.slipchance) * (1-M.obserror), (1-M.slipchance)*M.obserror])
    # Movement
    elseif pos in [correctsign(M), wrongsign(M)] && a == UP
        nextpos = endcorridor(M)
    elseif pos > 1 && a == UP
        nextpos = pos - 1
    elseif pos < endcorridor(M) && a == DOWN
        nextpos = pos + 1
    else
        nextpos = pos
    end
    snext = (nextpos, goal)
    return SafeSparseCat([snext,s],[1.0-M.slipchance, M.slipchance])
end

function POMDPs.observation(M::HeavenOrHell,a,sp)
    pos, goal = sp 
    pos == 0 && (return Deterministic(1))
    notgoal = 3 - goal
    pos == correctsign(M) && (return Deterministic(goal))
    pos == wrongsign(M) && (return Deterministic(3-goal))
    return Deterministic(pos)
end

function wrongReward(M::HeavenOrHell)
    return -M.size * 5.0 - 10
end


function POMDPs.reward(M::HeavenOrHell, s, a)::Float64
    pos, goal = s
    pos == 0 && return 0.0
    if pos == 1 && a == LEFT
        goal == 1 && return 0.0
        goal == 2 && return wrongReward(M)
    elseif pos == 1 && a == RIGHT 
        goal == 2 && return 0.0
        goal == 1 && return wrongReward(M)
    end
    return -1.0
end
end