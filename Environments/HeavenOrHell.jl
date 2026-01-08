module HeavenOrHellModel
using POMDPs, POMDPTools, Distributions
export HeavenOrHell, HeavenOrHell2

@kwdef mutable struct HeavenOrHell <: POMDP{Tuple{Int,Int}, Int, Int}
    size::Int = 7
    slipchance::Float64 = 0.1
    discount::Float64 = 0.99
end

maxstate(M::HeavenOrHell) = 2*M.size + floor(Int, M.size/2) + 1
tsplitstate(M::HeavenOrHell) = floor(Int, M.size/2)+2
belowtsplitstate(M::HeavenOrHell) = M.size + 3
turnstate(M::HeavenOrHell) = M.size + 1
initpos(M::HeavenOrHell) = 2*M.size + 1 - floor(Int, M.size/2)

POMDPs.states(M::HeavenOrHell) = vec([(x,y) for x in 0:maxstate(M), y in 1:2])
POMDPs.statetype(M::HeavenOrHell) = Tuple{Int,Int}
POMDPs.stateindex(M::HeavenOrHell, s) = findfirst(isequal(s), states(M))
# POMDPs.stateindex(M::HeavenOrHell, s) = (first(s)+1) + (maxstate(M)+1) * (last(s)-1)
POMDPs.actions(M::HeavenOrHell) = 1:4 # UP, RIGHT, DOWN, LEFT
POMDPs.actiontype(M::HeavenOrHell) = Int 
POMDPs.actionindex(M::HeavenOrHell, a) = a 
POMDPs.observations(M::HeavenOrHell) = 1:maxstate(M)
POMDPs.obstype(M::HeavenOrHell) = Int
POMDPs.obsindex(M::HeavenOrHell, o) = o
POMDPs.discount(M::HeavenOrHell) = M.discount
POMDPs.initialstate(M::HeavenOrHell) = SparseCat([(initpos(M),1), (initpos(M),2)], [0.5,0.5])
POMDPs.isterminal(M::HeavenOrHell, s) = first(s) == 0

function POMDPs.transition(M::HeavenOrHell, s, a)
    pos = first(s)
    posnext = first(s)
    #H (sinkstates)
    if pos < 3
        return Deterministic((0,last(s)))
    # Reaching H (no slipping)
    elseif pos==3 && a==1
        return Deterministic((1, last(s)))
    elseif pos==M.size+2 && a==1 
        return Deterministic((2, last(s)))
    # Moving left/right on top
    elseif pos >= 3 && pos < M.size+2 && a==2
        posnext = pos+1
    elseif pos > 3 && pos <= M.size+2 && a==4
        posnext = pos-1
    # Moving up/down on vertical
    elseif pos >= belowtsplitstate(M) && pos < turnstate(M) && a==3
        posnext = pos+1
    elseif pos > belowtsplitstate(M) && pos <= turnstate(M) && a==1
        posnext = pos-1
    # Moving up/down on t-split
    elseif pos == tsplitstate(M) && a==3
        posnext = belowtsplitstate(M)
    elseif pos == belowtsplitstate(M) && a==1
        posnext = tsplitstate(M)
    # Moving left/right on bottom
    elseif pos >= turnstate(M) && pos < maxstate(M) && a==2
        posnext = pos+1
    elseif pos > turnstate(M) && a==4
        posnext = pos-1
    else
        return Deterministic(s)
    end
    snext = (posnext, last(s))
    return SparseCat([snext,s],[1.0-M.slipchance, M.slipchance])
end

function POMDPs.observation(M::HeavenOrHell,a,sp)
    first(sp) == 0 && (return Deterministic(1))
    first(sp) == maxstate(M) && (return Deterministic(last(sp)))
    return Deterministic(first(sp))
end

function POMDPs.reward(M::HeavenOrHell, s, a)::Float64
    pos, goal = s
    pos == 0 && return 0.0
    pos == goal && return 100.0
    pos!=goal && pos <= 2 && return -100.0
    return -1.0
end







UP, RIGHT, DOWN, LEFT = 1,2,3,4
@kwdef mutable struct HeavenOrHell2 <: POMDP{Tuple{Int,Int}, Int, Int}
    size::Int = 3
    slipchance::Float64 = 0.0
    obserror::Float64 = 0.5
    discount::Float64 = 0.99
    rewards::NTuple{3,Float64} = (0.0, -30.0, -1.0)
end
endcorridor(M::HeavenOrHell2) = 2*M.size + 1
correctsign(M::HeavenOrHell2) = 2*M.size + 2
wrongsign(M::HeavenOrHell2) = 2*M.size + 3


POMDPs.states(M::HeavenOrHell2) = vec([(x,y) for x in 0:wrongsign(M), y in 1:2])
POMDPs.statetype(M::HeavenOrHell2) = Tuple{Int,Int}
POMDPs.stateindex(M::HeavenOrHell2, s) = findfirst(isequal(s), states(M))
# POMDPs.stateindex(M::HeavenOrHell, s) = (first(s)+1) + (maxstate(M)+1) * (last(s)-1)
POMDPs.actions(M::HeavenOrHell2) = [UP,DOWN] # LEFT, RIGHT
POMDPs.actiontype(M::HeavenOrHell2) = Int 
POMDPs.actionindex(M::HeavenOrHell2, a) = findfirst(isequal(a), actions(M))
POMDPs.observations(M::HeavenOrHell2) = 1:endcorridor(M)
POMDPs.obstype(M::HeavenOrHell2) = Int
POMDPs.obsindex(M::HeavenOrHell2, o) = o
POMDPs.discount(M::HeavenOrHell2) = M.discount
POMDPs.initialstate(M::HeavenOrHell2) = SparseCat([(M.size+1,1), (M.size+1,2)], [0.5,0.5])

POMDPs.isterminal(M::HeavenOrHell2, s) = first(s) == 0

function POMDPs.transition(M::HeavenOrHell2, s, a)
    pos, goal = s
    # nextpos = pos
    pos == 0 && return Deterministic(s)
    # Picking heaven or hell:
    if pos == 1 && a in [UP,DOWN]
        return Deterministic((0,goal))
    # Viewing sign:
    elseif pos == endcorridor(M) && a == DOWN
        println("end, down:", SparseCat([(endcorridor(M),goal),(correctsign(M), goal), (wrongsign(M), goal)], [M.slipchance, (1-M.slipchance) * (1-M.obserror), (1-M.slipchance)*M.obserror]))
        return SparseCat([(endcorridor(M),goal),(correctsign(M), goal), (wrongsign(M), goal)], [M.slipchance, (1-M.slipchance) * (1-M.obserror), (1-M.slipchance)*M.obserror])
    elseif pos in [correctsign(M), wrongsign(M)] && a == DOWN
        println("sign, up:", SparseCat([(correctsign(M), goal), (wrongsign(M), goal)], [1-M.obserror, M.obserror]))
        return SparseCat([(correctsign(M), goal), (wrongsign(M), goal)], [1-M.obserror, M.obserror])
    # Movement
    elseif pos in [correctsign(M), wrongsign(M)] && a == UP
        nextpos = endcorridor(M)
    elseif pos > 1 && a == UP
        nextpos = pos - 1
    elseif pos < endcorridor(M) && a == DOWN
        nextpos = pos + 1
    else
        println(s, " ", ["UP","DOWN"][a])
        nextpos = pos
    end
    snext = (nextpos, goal)
    println(pos," ", ["UP", "RIGHT", "DOWN", "LEFT"][a], nextpos)
    return SparseCat([snext,s],[1.0-M.slipchance, M.slipchance])
end

function POMDPs.observation(M::HeavenOrHell2,a,sp)
    pos, goal = sp 
    pos == 0 && (return Deterministic(1))
    notgoal = 3 - goal
    # println("(not)goal = $goal, $notgoal")
    pos == correctsign(M) && (return Deterministic(goal))
    pos == wrongsign(M) && (return Deterministic(3-goal))
    return Deterministic(pos)
end

function POMDPs.reward(M::HeavenOrHell2, s, a)::Float64
    pos, goal = s
    pos == 0 && return 0.0
    if pos == 1 && a == UP
        goal == 1 && return M.rewards[1]
        goal == 2 && return M.rewards[2]
    elseif pos == 1 && a == DOWN 
        goal == 2 && return M.rewards[1]
        goal == 1 && return M.rewards[2]
    end
    return M.rewards[3]
end

end