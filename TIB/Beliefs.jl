#########################################
#          Belief Definitions
#########################################

"""A custom belief implementation that includes a pre-computed hash value"""
struct DiscreteHashedBelief{S}
    state_list::Vector{S}       # assumed sorted!
    probs::Vector{Float64}
    hash::UInt
end

function DiscreteHashedBelief(state_list::Vector, probs::Vector{<:Float64}) # evt. Test verschil rationals -> B1/B2 worden kleiner?
    nonzero_els = findall(>(0),probs)
    state_list, probs = state_list[nonzero_els], probs[nonzero_els]
    idxs = sortperm(state_list)
    ordered_state_list, ordered_probs = state_list[idxs], probs[idxs]
    hash = makeDBhash(ordered_state_list, ordered_probs)
    return DiscreteHashedBelief(ordered_state_list, ordered_probs, hash)
end

#Note: This method should only be used when b is a belief, but currently this is not checked.
# I don't see any way to do this though: the beliefs used throughout the POMDP framework do not have a consistent supertype (even though they should all be distributions...)
# Maybe checking for the existance of a support/pdf function would be enough, but the way of doing this in Julia (method_exists()) seems to be removed and is the only thing I can find.
function DiscreteHashedBelief(b) 
    S,P = [], Float64[]
    for (s,p) in weighted_iterator(b)
        if p>0
            push!(S,s)
            push!(P,p)
        end
    end
    return DiscreteHashedBelief(S,P)
end

function POMDPs.rand(rng::AbstractRNG, s::Random.SamplerTrivial{DiscreteHashedBelief})
    d = s[]
    r = rand(rng)
    tot = 0.0
    for x in support(d)
        tot += pdf(d,x)
        r < tot && return x
    end
    tot < 1.0 && throw("Trying to sample from non-normalized belief (with total probability $tot)")
    throw("Error: sampling from DiscretizedBelief failed for unknown reason.")
end

function POMDPs.pdf(d::DiscreteHashedBelief, s) 
    # possible_ks = searchsorted(d.state_list, s)
    # for k in possible_ks
    #     d.state_list[k] == s && return d.probs[k]
    # end
    # return 0
    k=findfirst( ==(s), d.state_list)               # This could use the fact that states are sorted...
    isnothing(k) ? (return 0) : (return d.probs[k])
end
POMDPs.support(d::DiscreteHashedBelief) = d.state_list

Base.length(d::DiscreteHashedBelief) = length(d.state_list)
mean(d::DiscreteHashedBelief) = throw("Function not implemented")
mode(d::DiscreteHashedBelief) = throw("Function not implemented")

POMDPTools.weighted_iterator(b::DiscreteHashedBelief) = zip(b.state_list, b.probs)

#########################################
#          Hashing & Equality
#########################################

Base.:(==)(x::DiscreteHashedBelief, y::DiscreteHashedBelief) = (x.hash == y.hash) && all( map( s -> isapprox( pdf(x,s), pdf(y,s); atol=10^-3 ),  collect(support(x))))

function Base.:(<)(x::DiscreteHashedBelief, y::DiscreteHashedBelief)
    (x.hash != y.hash) && return (x.hash < y.hash)
    for k in sort(vcat(collect(support(x)), collect(support(y))))
        pdf(x,k) < pdf(y,k) && return true
        pdf(x,k) > pdf(y,k) && return false
    end
    return false
end
Base.isless(x::DiscreteHashedBelief{<:Any}, y::DiscreteHashedBelief{<:Any}) = x < y

makeDBhash(states_list::Vector, probs::Vector{Float64}) = hash(hash_alt(states_list), hash_alt(probs))
hash_alt(v::Vector) = foldr( (x,y) -> hash(x,y), v; init=UInt(0))

Base.hash(x::DiscreteHashedBelief, h::UInt) = hash(x.hash, h)
Base.hash(x::DiscreteHashedBelief) = hash(x,UInt(0))

#########################################
#          Belief Updater
#########################################

"""Struct for updating DiscreteHashedBelief"""
struct DiscreteHashedBeliefUpdater <: Updater
    model::POMDP
end

"""Given a distribution d, create a DiscreteHashedBelief"""
function initialize_belief(bu::DiscreteHashedBeliefUpdater, d)
    S,P = [], []
    for (s,p) in weighted_iterator(d)
        push!(S,s); push!(P,p)
    end
    return DiscreteHashedBelief(S,P)
end

function POMDPs.update(bu::DiscreteHashedBeliefUpdater, b::DiscreteHashedBelief,a,o)
    model = bu.model
    bnext = Dict{Any, Float64}()

    ### Collect possible next states with transition probs
    # for (s, ps) in weighted_iterator(b)
    #     for (snext, psnext) in weighted_iterator(transition(model,s,a))
    #         add_to_dict!(bnext, snext, ps * psnext)
    #     end
    # end

    # ### Alter weights according to obs
    # bnext_ = Dict{Any, Float64}()
    # for (snext, psnext) in bnext
    #     po = pdf(observation(model,a,snext), o)
    #     bnext_[snext] = psnext * po
    # end
    # ### Normalize
    # states, probs = collect(keys(bnext_)), collect(values(bnext_))
    # probs ./= sum(probs)

    # bp = DiscreteHashedBelief(states,probs)

    # length(support(bp)) == 0 && println(sum(probs), bnext_, bnext, DiscreteHashedBelief(states,probs))
    # println(sum(probs), bnext_, bnext, DiscreteHashedBelief(states,probs))

    for (s, ps) in weighted_iterator(b)
        ss_next = transition(model, s, a)
        for (snext, psnext) in weighted_iterator(ss_next)
            po = obs_weight(model,s,a,snext,o)
            add_to_dict!(bnext, snext, ps*psnext*po)
        end
    end

    states, probs = collect(keys(bnext)), collect(values(bnext))
    probs ./= sum(probs)


    return DiscreteHashedBelief(states,probs)
end

#Note: again, we never type-check b, but I don't know how to do this...
POMDPs.update(bu::DiscreteHashedBeliefUpdater, b, a, o) = update(bu, DiscreteHashedBelief(b),a,o) 