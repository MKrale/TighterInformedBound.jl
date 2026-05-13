# Some random convenience functions used in other code:

function breward(model::POMDP{S,A,O}, b::DiscreteHashedBelief{S},a::A) where S where A where O
    r = 0.0
    for (s,p) in zip(b.state_list, b.probs)
        s == POMDPTools.ModelTools.TerminalState() || ( r += p * POMDPs.reward(model,s,a) )
    end
    return r
end

function add_to_dict!(dict::Dict{K,V}, key::K, value::V; func=+, minvalue::V=0.0) where K where V <: Number
    if haskey(dict, key)
        dict[key] = func(dict[key], value)
    elseif isnothing(minvalue) || value > minvalue
        dict[key] = value
    end
end

function sparseTabularPOMDPFiniteReward(m::SparseTabularPOMDP, min_value::Float64 = -100_000.0)
    R_finite = map(r -> max(r, min_value), m.R)
    return SparseTabularPOMDP(m.T, R_finite, m.O, m.initial_probs, m.terminal_states, m.discount)
end


# Convenience function for debugging
printdb(x) = print(x,"\n")
function printdb(x,y...)
    print(x,", ")
    printdb(y...)
end