# Some random convenience functions used in other code:

function breward(model::POMDP, b::DiscreteHashedBelief,a)
    r = 0.0
    for (s,p) in zip(b.state_list, b.probs)
        s == POMDPTools.ModelTools.TerminalState() || ( r += p * POMDPs.reward(model,s,a) )
    end
    return r
end

function add_to_dict!(dict, key, value; func=+, minvalue=1e-10)
    if haskey(dict, key)
        dict[key] = func(dict[key], value)
    elseif isnothing(minvalue) || value > minvalue
        dict[key] = value
    end
end