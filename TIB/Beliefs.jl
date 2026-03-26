#########################################
#          Belief Definitions
#########################################

"""A custom belief implementation that includes a pre-computed hash value"""
struct DiscreteHashedBelief{S}
    state_list::Vector{S}       # assumed sorted!
    probs::Vector{Float64}
    hash::UInt
end

function DiscreteHashedBelief{S}(state_list::Vector{S}, probs::Vector{<:Float64}) where S
    nonzero_els = findall(>(0),probs)
    state_list = state_list[nonzero_els]
    probs = Float64.(probs[nonzero_els])
    idxs = sortperm(state_list)
    ordered_state_list = state_list[idxs]
    ordered_probs = probs[idxs]
    hash = makeDBhash(ordered_state_list, ordered_probs)
    return DiscreteHashedBelief{S}(ordered_state_list, ordered_probs, hash)
end

#Note: This method should only be used when b is a belief, but currently this is not checked.
# I don't see any way to do this though: the beliefs used throughout the POMDP framework do not have a consistent supertype (even though they should all be distributions...)
# Maybe checking for the existance of a support/pdf function would be enough, but the way of doing this in Julia (method_exists()) seems to be removed and is the only thing I can find.
function DiscreteHashedBelief{S}(b::SparseCat{Vector{S},Vector{M}}) where M<:Float64 where S 
    return DiscreteHashedBelief{S}(b.vals, b.probs)
end
function DiscreteHashedBelief{Bool}(b::BoolDistribution)
    return DiscreteHashedBelief{Bool}([true, false], [b.p, 1-b.p])
end

function DiscreteHashedBelief{S}(b::Distribution{F,S}) where F where S 
    states, probs = [], Float64[]
    for (s,p) in weighted_iterator(b)
        if p>0
            push!(states,s)
            push!(probs,p)
        end
    end
    return DiscreteHashedBelief{S}(states,probs)
end


function POMDPs.rand(rng::AbstractRNG, s::Random.SamplerTrivial{DiscreteHashedBelief})
    d = s[]
    r = rand(rng)
    tot = 0.0
    for (x, px) in weighted_iterator(d)
        tot += px
        r < tot && return x
    end
    tot < 1.0 && throw("Trying to sample from non-normalized belief (with total probability $tot)")
    throw("Error: sampling from DiscretizedBelief failed for unknown reason.")
end

function POMDPs.pdf(d::DiscreteHashedBelief, s) 
    possible_ks = searchsorted(d.state_list, s)
    for k in possible_ks
        d.state_list[k] == s && return d.probs[k]
    end
    return 0
end
POMDPs.support(d::DiscreteHashedBelief{S}) where S = d.state_list
POMDPTools.weighted_iterator(b::DiscreteHashedBelief{S}) where S = zip(b.state_list, b.probs)

Base.length(d::DiscreteHashedBelief) = length(d.state_list)
mean(d::DiscreteHashedBelief) = throw("Function not implemented")
mode(d::DiscreteHashedBelief) = throw("Function not implemented")

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
struct DiscreteHashedBeliefUpdater{S,A,O} <: Updater
    model::X where X<:POMDP{S,A,O}
end

"""Given a distribution d, create a DiscreteHashedBelief"""
function initialize_belief(bu::DiscreteHashedBeliefUpdater{Ss}, d) where Ss
    S,P = [], []
    for (s,p) in weighted_iterator(d)
        push!(S,s); push!(P,p)
    end
    return DiscreteHashedBelief{S}(S,P)
end

function POMDPs.update(bu::DiscreteHashedBeliefUpdater{S}, b::DiscreteHashedBelief{S},a,o) where S
    model = bu.model
    bnext = Dict{S, Float64}()

    ### Collect possible next states with transition probs
    for (s, ps) in weighted_iterator(b)
        for (snext, psnext) in weighted_iterator(transition(model,s,a))
            add_to_dict!(bnext, snext, ps * psnext)
        end
    end

    ### Alter weights according to obs
    bnext_ = Dict{S, Float64}()
    for (snext, psnext) in bnext
        po = pdf(observation(model,a,snext), o)
        bnext_[snext] = psnext * po
    end
    ### Normalize
    states, probs = collect(keys(bnext_)), collect(values(bnext_))
    probs ./= sum(probs)

    return DiscreteHashedBelief{S}(states,probs)
end

#TODO: again, we never type-check b, but I don't know how to do this...
POMDPs.update(bu::DiscreteHashedBeliefUpdater, b, a, o) = update(bu, DiscreteHashedBelief(b),a,o) 

#########################################
#          Other
#########################################

function dot(b::DiscreteHashedBelief{Int64}, alpha::Vector{Float64})
    product = 0.0
    for (s, p) in weighted_iterator(b)
        product += p * alpha[s]
    end
    return product
end
