# Contains code for all the caching required to make (E/O)TIB run efficiently

"""Struct containing parameter vectors & sizes, to prevent calling (possibly expensive) POMDP functions"""
struct C{S,A,O}
    S::Vector{S}; A::Vector{A}; O::Vector{O} 
    ns::Int; na::Int; no::Int
end
function get_constants(model::POMDP{S,A,O}) where S where A where O 
    return C( sort(collect(states(model))), sort(collect(actions(model))), sort(collect(observations(model))),
            length(states(model)), length(actions(model)), length(observations(model)))
end

#########################################
#             One-step beliefs:
#########################################

"""Precomputed data used by TIB solvers & policies"""
struct TIB_Data{S}
    Q::Union{Array{Float64,2}, Nothing}     # bi, ai -> Q
    B::Vector{DiscreteHashedBelief{S}}              # bi -> b
    B_idx::Array{Int64,3}                     # si,ai,oi -> bi
    Br::Array{Float64,2}                    # bi -> reward
    SAO_probs::Array{Float64,3}             # si,ai,oi -> p
    SAOs::Array{Vector{Int64},2}              # si,ai -> [oi]
    BAOs::Array{Vector{Int64}, 2}           # bi,ai -> [oi]
    S_dict::Dict{S, Int64}                            # s -> si
    constants::C{S}                     
end

TIB_Data{S}(Q::Array{Float64,2}, D::TIB_Data{S}) where S = TIB_Data{S}(Q,D.B, D.B_idx, D.Br, D.SAO_probs, D.SAOs, D.BAOs, D.S_dict, D.constants) 

"""Precomputed data used by FIB & QMDP"""
struct Simple_Data{S}
    Q::Union{Array{Float64,2}, Nothing}     # si, ai -> Q
    V::Union{Array{Float64,1}, Nothing}     # si -> V
    S_dict::Dict{S, Int}                  # s -> si
    constants::C
end

"""
Computes the probabilities of each observation for each state-action pair. \n 
Returns: SAO_probs ((s,a,o)->p), SAOs ((s,a) -> os with p>0)
"""
function get_all_obs_probs(model::POMDP{Ss, As, Os}, constants::C{Ss,As,Os}) where Ss where As where Os
    S,A,O = constants.S , constants.A, constants.O
    ns, na, no = constants.ns, constants.na, constants.no
    O_dict = Dict{Os, Int}( zip(O, 1:no))   

    SAO_probs = zeros(Float64, no,ns,na)
    SAOs = Array{Vector{Int}}(undef,ns,na)
    for si in 1:ns
        for ai in 1:na
            SAOs[si,ai] = []
        end
    end
    for (si, s) in enumerate(constants.S)
        for (ai, a) in enumerate(constants.A)
            obs_probs = Dict{Int, Float64}()
            for (sp, psp) in weighted_iterator(transition(model,s,a))
                for (o, po) in weighted_iterator(observation(model,a,sp))
                    oi = O_dict[o]
                    add_to_dict!(obs_probs, oi, po*psp)
                end
            end
            for oi in keys(obs_probs)
                push!(SAOs[si,ai], oi)
                SAO_probs[oi,si,ai] += obs_probs[oi]
            end
        end
    end
    return SAO_probs, SAOs
end

"""Computes the probability of an observation given a state-action pair."""
function get_obs_prob(model::POMDP, o,s,a)
    p = 0
    for (sp, psp) in weighted_iterator(transition(model,s,a))
        dist = observation(model,a,sp)
        p += psp*pdf(dist,o)
    end
    return p
end
"""Computes the probability of an observation given a belief-action pair."""
function get_obs_prob(model::POMDP{S,A,O}, o, b::DiscreteHashedBelief{S}, a) where S where A where O 
    sum = 0
    for (s,p) in zip(b.state_list, b.probs)
        sum += p*get_obs_prob(model,o,s,a)
    end
    return sum
end

"""Computes observations possible given a belief and action"""
function get_possible_obs(b::DiscreteHashedBelief{S}, ai, S_dict, SAOs) where S
    os = Set{Int64}()
    for s in support(b)
        si = S_dict[s]
        for oi in SAOs[si,ai]
            push!(os, oi)
        end
    end
    return collect(os)
end
get_possible_obs(b::DiscreteHashedBelief{S}, ai, Data::TIB_Data{S}) where S = get_possible_obs(b::DiscreteHashedBelief{S}, ai, Data.S_dict, Data.SAOs)
get_possible_obs(bi::Int, ai::Int, Data::TIB_Data) = Data.BAOs[bi,ai]

"""Computes observations-prob dictionary of possible observations given a belief and action"""
function get_possible_obs_probs(b::DiscreteHashedBelief{S}, ai, Data::TIB_Data{S}) where S
    possible_os = Dict{Int, Float64}()
    for (s, ps) in weighted_iterator(b)
        si = Data.S_dict[s]
        for oi in Data.SAOs[si,ai]
            add_to_dict!(possible_os, oi, ps * Data.SAO_probs[oi,si,ai])
        end
    end
    return possible_os
end

function get_BAOs(B::Vector{DiscreteHashedBelief{S}}, S_dict, SAOs, constants::C{S}) where S
    BAOs = Array{Vector{Int}}(undef,length(B),constants.na)
    for (bi, b) in enumerate(B)
        for ai in 1:constants.na
            BAOs[bi,ai] = get_possible_obs(b,ai,S_dict,SAOs)
        end
    end
    return BAOs
end

"""Constructs TIB_DATA from the given model"""
function get_TIB_Data(model::POMDP{S}) where S
    constants = get_constants(model)
    SAO_probs, SAOs = get_all_obs_probs(model, constants)
    S_dict = Dict{eltype(constants.S),Int}( zip(constants.S, 1:constants.ns))                                   
    B, B_idx = get_belief_set(model, SAOs, constants)
    Br = get_Br(model, B, constants)
    BAOs = get_BAOs(B, S_dict, SAOs, constants)
    return TIB_Data{S}(nothing,B,B_idx,Br,SAO_probs,SAOs, BAOs, S_dict,constants)
end

"""
Computes all (unique) one-step beliefs

Returns: a vector with these beliefs, as well as a 3D array mapping (s,a,o)-indexes to these beliefs
"""
function get_belief_set(model::POMDP{Ss,As,Os}, SAOs, constants::C{Ss,As,Os}) where Ss where As where Os
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")
    S, A, O = constants.S, constants.A, constants.O
    ns,na,no = constants.ns, constants.na, constants.no
    U = DiscreteHashedBeliefUpdater{Ss, As, Os}(model)

    B = Array{DiscreteHashedBelief{Ss},1}()       
    B_idx = zeros(Int,ns,na,no)
    B_dict = Dict()   

    ### Initialize with unit beliefs and initial belief
    ### Note: technically we should ignore the unit beliefs for ETIB, but we currently do not
    for (si, s) in enumerate(S) 
        bs = DiscreteHashedBelief{Ss}([s],[1.0])
        push!(B, bs)
        B_dict[bs] = si
    end
    b_init = DiscreteHashedBelief{Ss}(initialstate(model))
    k = get(B_dict, b_init, nothing)
    if isnothing(k)
        push!(B,b_init)
        B_dict[b_init] = length(B)
    end

    # Loop through all 1-step transitions and records new beliefs
    for (si,s) in enumerate(S)
        b_s = DiscreteHashedBelief{Ss}([s],[1.0])
        for (ai,a) in enumerate(A)
            for oi in SAOs[si,ai]
                o = O[oi]
                b = update(U, b_s, a, o)
                k = get(B_dict, b, nothing)
                if isnothing(k)
                    push!(B,b)
                    k=length(B)
                    B_dict[b] = k
                end
                B_idx[si,ai,oi] = k
            end
        end
    end

    return B, B_idx
end

"""Returns a vector with expected rewards for each belief in B"""
function get_Br(model, B, constants::C)
    A, na = constants.A, constants.na
    Br = zeros(Float64, length(B), na)
    for (bi, b) in enumerate(B)
        for (ai, a) in enumerate(A)
            Br[bi,ai] = breward(model,b,a)
        end
    end
    return Br
end

#########################################
#          Two-step beliefs:
#########################################

"""
Pre-computed data regarding two-step beliefs, as used by ETIB, CTIB and OTIB
"""
struct BBAO_Data{S}
    Bbao::Vector{DiscreteHashedBelief{S}}                   # Vector of 2-step beliefs (excluding already-found 1-step beliefs!)
    Bbao_idx::Array{Dict{Int,Tuple{Bool,Int}},2}            # bi, ai, oi -> bpi     : gives Bbao-index for each transition
    BAO_probs::AbstractArray{Float64,3}                     # bi, ai, oi -> p       : gives probability of each transition
    B_in_Bbao::AbstractArray{Bool}                          # bi -> Bool            : is the one-step also a 2-step belief?
    B_overlap::AbstractArray{Vector{Int}}                   # bi -> [bi]            : Gives vector of all beliefs who's support is a subset of the support of bi
    Bbao_overlap::AbstractArray{Vector{Int}}                # bi -> [bi]            : Gives vector all beliefs who's support is a subset of the support of bi
end

"""Convenience function to get belief given transition"""
function get_bao(Bbao_data::BBAO_Data{S}, bi::Int, ai::Int, oi::Int, B::Vector{DiscreteHashedBelief{S}}) where S
    (in_B, baoi) = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return B[baoi]) : (return Bbao_data.Bbao[baoi])
end
"""Convenience function to get overlapping beliefs given transition"""
function get_overlap(Bbao_data::BBAO_Data{S}, bi::Int, ai::Int, oi::Int) where S
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return Bbao_data.B_overlap[baoi]) : (return Bbao_data.Bbao_overlap[baoi])
end

"""Computes all possible observations after the given belief-action pair"""
function get_possible_obs(bi::Tuple{Bool, Int}, ai, Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    if !first(bi)
        b = Data.B[last(bi)]
        possible_os = Set{Int}
        for s in support(b)
            si = Data.S_dict[s]
            union!(possible_os, SAOs[si,ai])
        end
        return 
    else
        return keys(Bbao_data.Bbao_idx[last(bi), ai])
    end
end

"""Constructs BBAO_Data"""
function get_Bbao(model::POMDP{S,A,O}, Data::TIB_Data{S}) where S where A where O

    # Preperations:
    constants = Data.constants
    B = Data.B
    S_dict = Data.S_dict
    O_dict = Dict{O,Int}( zip(constants.O, 1:constants.no) )
    U = DiscreteHashedBeliefUpdater{S, A, O}(model)
    nb = length(B)

    Bbao = DiscreteHashedBelief{S}[]
    BAO_probs = zeros(Float64, constants.no,nb,constants.na)
    B_in_Bboa = zeros(Bool, length(B))
    Bbao_valid_weights = Dict{Int,Float64}[]
    Bbao_idx = Array{Dict{Int, Tuple{Bool,Int}}}(undef, nb, constants.na)

    Bs_found = Dict{DiscreteHashedBelief{S}, Tuple{Bool,Int}}(zip(B, map( idx -> (true, idx), 1:length(B))))

    # Record all two-step beliefs: reference B if it's already in there, otherwise add to Bbao
    for (bi,b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            Bbao_idx[bi,ai] = Dict{Int, Tuple{Bool, Int}}()
            possible_obs = get_possible_obs_probs(b,ai,Data)
            for oi in keys(possible_obs)
                BAO_probs[oi,bi,ai] = possible_obs[oi]
                o = constants.O[oi]
                bao = POMDPs.update(U,b,a,o)
                if length(support(bao)) > 0 # Ignore impossible beliefs
                    if haskey(Bs_found, bao)
                        (in_B, idx) = Bs_found[bao]
                        in_B && (B_in_Bboa[idx] = true)
                        Bbao_idx[bi,ai][oi] = (in_B, idx)
                    else
                        push!(Bbao, bao)
                        k=length(Bbao)
                        Bbao_idx[bi,ai][oi] = (false, k)
                        Bs_found[bao] = (false, k)
                    end
                end              
            end
        end
    end
    nbao = length(Bbao)
    B_overlap = Array{Vector{Int}}(undef, nb)
    Bbao_overlap = Array{Vector{Int}}(undef, nbao)
    # For each belief, determine the state with the lowest index with non-zero support (speeds up overlap computations)
    Bs_lowest_support_state = Dict{Int, Vector{Int}}()
    for (bi, b) in enumerate(B)
        sidx_lowest = S_dict[support(b)[1]]
        if haskey(Bs_lowest_support_state, sidx_lowest)
            push!(Bs_lowest_support_state[sidx_lowest], bi)
        else
            Bs_lowest_support_state[sidx_lowest] = [bi]
        end
    end

    # Record overlap for b
    for (bi,b) in enumerate(B)
        B_overlap[bi] = []
        sidx_lowest = S_dict[support(b)[1]]
        sidx_highest = S_dict[support(b)[length(support(b))]]
        for sidx=sidx_lowest:sidx_highest
            # we will check only those beliefs whos lowest index lies between the lowest and highest index of our belief:
            # Depending on the env, this may significantly reduce the search space.
            if haskey(Bs_lowest_support_state, sidx) 
                for bpi in Bs_lowest_support_state[sidx]
                    have_overlap(b,B[bpi]) && push!(B_overlap[bi], bpi)
                end
            end
        end
    end
    # Repeat the process above for beliefs in Bbao
    for (bi,b) in enumerate(Bbao)
        Bbao_overlap[bi] = []
        sidx_lowest = S_dict[support(b)[1]]
        sidx_highest = S_dict[support(b)[length(support(b))]]
        for sidx=sidx_lowest:sidx_highest
            if haskey(Bs_lowest_support_state, sidx)
                for bpi in Bs_lowest_support_state[sidx]
                    have_overlap(b,B[bpi]) && push!(Bbao_overlap[bi], bpi)
                end
            end
        end
    end

    return BBAO_Data(Bbao, Bbao_idx, BAO_probs, B_in_Bboa, B_overlap, Bbao_overlap)
end

"""Returns true if the support of bp is a subset of the support of b"""
function have_overlap(b::DiscreteHashedBelief{S},bp::DiscreteHashedBelief{S}) where S
    bidx, bpidx = 1, 1
    sup_b, sup_bp = support(b), support(bp)
    n_sup_b, n_sup_bp = length(sup_b), length(sup_bp)
    while bidx <= n_sup_b && bpidx <= n_sup_bp
        if sup_b[bidx] == sup_bp[bpidx]     # both beliefs contain state: fine
            bidx += 1
            bpidx += 1
        elseif sup_b[bidx] <= sup_bp[bpidx] # b contains state not in bp: fine
            bidx += 1
        else    # bp contains a state not in b -> not allowed
            return false
        end
    end
    return (bpidx > n_sup_bp)  # all elements in support bp have been checked
end

"""Finds all beliefs in B where the support is a subset of that of b"""
function get_overlapping_beliefs(b::DiscreteHashedBelief{S}, B::Vector{DiscreteHashedBelief{S}}) where S
    Bs, B_idxs = DiscreteHashedBelief{S}[], Int[]
    for (bpi, bp) in enumerate(B)
        if have_overlap(b,bp)
            push!(Bs, bp)
            push!(B_idxs, bpi)
        end
    end
    return Bs, B_idxs
end

"""Computes the state-entropy of a belief"""
function get_entropy(b::DiscreteHashedBelief{S}) where S
    entropy = 0
    for (s,p) in weighted_iterator(b)
        entropy += -log(p) * p
    end
    return entropy
end

#########################################
#          Three-step beliefs:
#########################################

function print_model_info(model::POMDP{S}) where S
    Data = get_TIB_Data(model)
    Bbao_data = get_Bbao(model, Data)
    three_step_beliefs = get_three_step_beliefs(model, Data, Bbao_data)
    println("==================")
    println("Model info:")
    println("|S| = $(Data.constants.ns), |A| = $(Data.constants.na), |O| = $(Data.constants.no)")
    println("|B1| = $(length(Data.B)), |B2| = $(length(Bbao_data.Bbao)), |B3| = $(length(three_step_beliefs))")
    println("==================")
end

function get_three_step_beliefs(model::POMDP{S,A,O}, Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S where A where O
    
    constants = Data.constants
    three_step_beliefs = DiscreteHashedBelief{S}[]
    U = DiscreteHashedBeliefUpdater{S, A, O}(model)
    
    existing_beliefs = Dict{DiscreteHashedBelief{S}, Tuple{Int,Int}}()
    for (bidx, b) in enumerate(Data.B)
        existing_beliefs[b] = (1, bidx)
    end
    for (bidx, b) in enumerate(Bbao_data.Bbao)
        existing_beliefs[b] = (2, bidx)
    end

    for (bidx, b) in enumerate(Bbao_data.Bbao)
        for (aidx, a) in enumerate(Data.constants.A)
            possible_obs = get_possible_obs_probs(b, aidx, Data)
            for oidx in keys(possible_obs)
                o = constants.O[oidx]
                bao = POMDPs.update(U,b,a,o)
                if length(support(bao)) > 0
                    if !haskey(existing_beliefs, bao)
                        push!(three_step_beliefs, bao)
                        existing_beliefs[bao] = (3, length(three_step_beliefs))
                    end
                end
            end
        end
    end
    return three_step_beliefs
end

