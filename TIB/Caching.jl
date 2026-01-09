# Contains code for all the caching required to make (E/O)TIB run efficiently

"""Struct containing parameter vectors & sizes, to prevent calling (possibly expensive) POMDP functions"""
struct C
    S; A; O 
    ns; na; no
end
get_constants(model) = C( sort(collect(states(model)), by=objectid), sort(collect(actions(model)),by=objectid), sort(collect(observations(model)), by=objectid),
                         length(states(model)), length(actions(model)), length(observations(model)))

#########################################
#             TIB_Data:
#########################################

"""Precomputed data used by TIB solvers & policies"""
struct TIB_Data
    Q::Union{Array{Float64,2}, Nothing}     # bi, ai -> Q
    B::Vector                               # bi -> b
    B_idx::Array{Int,3}                     # si,ai,oi -> bi
    Br::Array{Float64,2}                    # bi -> reward
    SAO_probs::Array{Float64,3}             # si,ai,oi -> p
    SAOs::Array{Vector{Int},2}              # si,ai -> oi
    S_dict::Dict{Any, Int}                  # s -> si
    constants::C                     
end

TIB_Data(Q::Array{Float64,2}, D::TIB_Data) = TIB_Data(Q,D.B, D.B_idx, D.Br, D.SAO_probs, D.SAOs, D.S_dict, D.constants) 

"""Precomputed data used by FIB & QMDP"""
struct Simple_Data
    Q::Union{Array{Float64,2}, Nothing}     # si, ai -> Q
    V::Union{Array{Float64,1}, Nothing}     # si -> V
    S_dict::Dict{Any, Int}                  # s -> si
    constants::C
end

"""
Computes the probabilities of each observation for each state-action pair. \n 
Returns: SAO_probs ((s,a,o)->p), SAOs ((s,a) -> os with p>0)
"""
function get_all_obs_probs(model::POMDP, constants::C)
    S,A,O = constants.S , constants.A, constants.O
    ns, na, no = constants.ns, constants.na, constants.no
    O_dict = Dict( zip(O, 1:no))    

    SAO_probs = zeros(no,ns,na)
    SAOs = Array{Vector{Int}}(undef,ns,na)
    for si in 1:ns
        for ai in 1:na
            SAOs[si,ai] = []
        end
    end

    for (si, s) in enumerate(constants.S)
        for (ai, a) in enumerate(constants.A)
            obs_probs = Dict()
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
function get_obs_prob(model::POMDP, o, b::DiscreteHashedBelief, a) 
    sum = 0
    for (s,p) in zip(b.state_list, b.probs)
        sum += p*get_obs_prob(model,o,s,a)
    end
    return sum
end
"""Computes observations possible given a belief and action"""
function get_possible_obs(b::DiscreteHashedBelief, ai, SAOs, S_dict)
    possible_os = Set{Int}()
    for s in support(b)
        si = S_dict[s]
        for oi in SAOs[si,ai]
            push!(possible_os, oi)
        end
    end
    return collect(possible_os)
end
"""Computes observations-prob dictionary of possible observations given a belief and action"""
function get_possible_obs_probs(b::DiscreteHashedBelief, ai, SAOs,SAO_probs, S_dict)
    possible_os = Dict{Int, Float64}()
    for (s,p) in weighted_iterator(b)
        si = S_dict[s]
        for oi in SAOs[si,ai]
            add_to_dict!(possible_os, oi, p * SAO_probs[oi,si,ai])
        end
    end
    return possible_os
end

"""Constructs TIB_DATA from the given model"""
function get_TIB_Data(model)
    constants = get_constants(model)
    SAO_probs, SAOs = get_all_obs_probs(model, constants)
    S_dict = Dict( zip(constants.S, 1:constants.ns))                                   
    B, B_idx = get_belief_set(model, SAOs, constants)
    Br = get_Br(model, B, constants)
    return TIB_Data(nothing,B,B_idx,Br,SAO_probs,SAOs,S_dict,constants)
end

"""
Computes all (unique) one-step beliefs

Returns: a vector with these beliefs, as well as a 3D array mapping (s,a,o)-indexes to these beliefs
"""
function get_belief_set(model::POMDP, SAOs, constants::C)
    isnothing(constants) && throw("Not implemented error! (get_obs_probs)")
    S, A, O = constants.S, constants.A, constants.O
    ns,na,no = constants.ns, constants.na, constants.no
    U = DiscreteHashedBeliefUpdater(model)

    B = Array{DiscreteHashedBelief,1}()
    B_dict = Dict()       
    B_idx = zeros(Int,ns,na,no)

    # Initialize with unit beliefs and initial belief
    # Note: technically we should ignore the unit beliefs for ETIB, but we currently do not
    for (si, s) in enumerate(S) 
        bs = DiscreteHashedBelief([s],[1.0])
        push!(B, bs)
        B_dict[bs] = si
    end
    b_init = DiscreteHashedBelief(initialstate(model))
    k = get(B_dict, b_init, nothing)
    if isnothing(k)
        push!(B,b_init)
        B_dict[b_init] = length(B)
    end

    # Loop through all 1-step transitions and records new beliefs
    for (si,s) in enumerate(S)
        b_s = DiscreteHashedBelief([s],[1.0])
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
#          Bbao_data:
#########################################

"""
Pre-computed data regarding two-step beliefs, as used by ETIB, CTIB and OTIB
"""
struct BBAO_Data
    Bbao::Vector                                    # Vector of 2-step beliefs (excluding already-found 1-step beliefs!)
    Bbao_idx::Array{Dict{Int,Tuple{Bool,Int}},2}    # bi, ai, oi -> bpi     : gives Bbao-index for each transition
    BAO_probs::Array{Float64,3}                     # bi, ai, oi -> p       : gives probability of each transition
    B_in_Bbao::BitVector                            # bi -> Bool            : is the one-step also a 2-step belief?
    B_overlap::Array{Vector{Int}}                   # bi -> [bi]            : Gives vector of all beliefs who's support is a subset of the support of bi
    Bbao_overlap::Array{Vector{Int}}                # bi -> [bi]            : Gives vector all beliefs who's support is a subset of the support of bi
    B_entropies::Array{Float64}                     # bi -> e               : entropy of all one-step beliefs
    Valid_weights::Array{Dict{Int,Float64}}         # bi -> {(bi,p)}        : a valid initial weighting for each belief in Bbao
end

"""Convenience function to get belief given transition"""
function get_bao(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int, B)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return B[baoi]) : (return Bbao_data.Bbao[baoi])
end
"""Convenience function to get overlapping beliefs given transition"""
function get_overlap(Bbao_data::BBAO_Data, bi::Int, ai::Int, oi::Int)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    in_B ? (return Bbao_data.B_overlap[baoi]) : (return Bbao_data.Bbao_overlap[baoi])
end

"""Computes all possible observations after the given belief-action pair"""
function get_possible_obs(bi::Tuple{Bool, Int}, ai, Data::TIB_Data, Bbao_data::BBAO_Data)
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
function get_Bbao(model, Data::TIB_Data, constants)

    # Preperations:
    B = Data.B
    S_dict = Data.S_dict
    O_dict = Dict( zip(constants.O, 1:constants.no) )
    U = DiscreteHashedBeliefUpdater(model)
    nb = length(B)

    Bbao = []
    BAO_probs = zeros(constants.no,nb,constants.na)
    B_in_Bboa = zeros(Bool, length(B))
    Bbao_valid_weights = Dict{Int,Float64}[]
    Bbao_idx = Array{Dict{Int, Tuple{Bool,Int}}}(undef, nb, constants.na)

    Bs_found = Dict(zip(B, map( idx -> (true, idx), 1:length(B))))

    # Record all two-step beliefs: reference B if it's already in there, otherwise add to Bbao
    for (bi,b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            Bbao_idx[bi,ai] = Dict{Int, Tuple{Bool, Int}}()
            possible_obs = get_possible_obs_probs(b,ai,Data.SAOs, Data.SAO_probs, S_dict)
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
                        valid_weights = Dict{Int, Float64}()
                        for (s,ps) in weighted_iterator(b)
                            valid_weights[Data.B_idx[S_dict[s],ai,oi]] = ps
                        end
                        push!(Bbao_valid_weights, valid_weights)
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
    Bs_lowest_support_state = Dict()
    for (bi, b) in enumerate(B)
        s_lowest = support(b)[1]
        if haskey(Bs_lowest_support_state, s_lowest)
            push!(Bs_lowest_support_state[s_lowest], bi)
        else
            Bs_lowest_support_state[s_lowest] = [bi]
        end
    end

    # Record overlap for b
    for (bi,b) in enumerate(B)
        B_overlap[bi] = []
        sidx_lowest, sidx_highest = S_dict[support(b)[1]], S_dict[support(b)[length(support(b))]]
        for sidx=sidx_lowest:sidx_highest
            # we will check only those beliefs whos lowest index lies between the lowest and highest index of our belief:
            # Depending on the env, this may significantly reduce the search space.
            s = constants.S[sidx]
            if haskey(Bs_lowest_support_state, s) 
                for bpi in Bs_lowest_support_state[s]
                    have_overlap(b,B[bpi]) && push!(B_overlap[bi], bpi)
                end
            end
        end
    end
    # Repeat the process above for beliefs in Bbao
    for (bi,b) in enumerate(Bbao)
        Bbao_overlap[bi] = []
        sidx_lowest, sidx_highest = S_dict[support(b)[1]], S_dict[support(b)[length(support(b))]]
        for sidx=sidx_lowest:sidx_highest
            s = constants.S[sidx]
            if haskey(Bs_lowest_support_state, s)
                for bpi in Bs_lowest_support_state[s]
                    have_overlap(b,B[bpi]) && push!(Bbao_overlap[bi], bpi)
                end
            end
        end
    end

    B_entropy = map( b -> get_entropy(b), B)
    return BBAO_Data(Bbao, Bbao_idx, BAO_probs, B_in_Bboa, B_overlap, Bbao_overlap, B_entropy, Bbao_valid_weights)
end

"""Returns true if the support of bp is a subset of the support of b"""
function have_overlap(b::DiscreteHashedBelief,bp::DiscreteHashedBelief)
    bidx, bpidx = 1, 1
    sup_b, sup_bp = support(b), support(bp)
    n_sup_b, n_sup_bp = length(sup_b), length(sup_bp)
    while bidx <= n_sup_b && bpidx <= n_sup_bp
        if sup_b[bidx] == sup_bp[bpidx]     # both beliefs contain state: fine
            bidx += 1
            bpidx += 1
        elseif objectid(sup_b[bidx]) <= objectid(sup_bp[bpidx]) # b contains state not in bp: fine
            bidx += 1
        else    # bp contains a state not in b -> not allowed
            return false
        end
    end
    return (bpidx >= n_sup_bp)  # all elements in support bp have been checked
end

"""Finds all beliefs in B where the support is a subset of that of b"""
function get_overlapping_beliefs(b, B::Vector)
    Bs, B_idxs = [], []
    for (bpi, bp) in enumerate(B)
        if have_overlap(b,bp)
            push!(Bs, bp)
            push!(B_idxs, bpi)
        end
    end
    return Bs, B_idxs
end

"""Computes the state-entropy of a belief"""
function get_entropy(b)
    entropy = 0
    for (s,p) in weighted_iterator(b)
        entropy += -log(p) * p
    end
    return entropy
end

#########################################
#               Weights:
#########################################

"""Precomputed data for weighting"""
struct Weights_Data
    # We have seperate vectors for belief sets B and Bbao. 
    # For each belief, vector ..._idx gives the index with non-zero weights, and ..._weights the corresponding weight.
    B_idxs::Vector{Vector{Int}}
    B_weights::Vector{Vector{Float64}}
    Bbao_idxs::Vector{Vector{Int}}
    Bbao_weights::Vector{Vector{Float64}}
end

"""Convenience function to get weights for a given bi,ai,oi tuple."""
function get_weights(Bbao_data::BBAO_Data, weights_data::Weights_Data, bi, ai, oi)
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    if in_B
        return (weights_data.B_idxs[baoi], weights_data.B_weights[baoi])
    else
        return (weights_data.Bbao_idxs[baoi], weights_data.Bbao_weights[baoi])
    end
end

########## Entropy weights ##########

"""Computes all max-entropy weights for Bbao"""
function get_entropy_weights_all(B, Bbao_data::BBAO_Data) #TODO: this can probably be combined in some way with get_closeness_weights_all
    # Define model: this setting performed best in our testing, but others are available.
    model = Model(Clp.Optimizer; add_bridges=false)
    set_silent(model)
    set_string_names_on_creation(model, false)

    # First, we do this for all beliefs in B which are also in Bbao:
    B_idxs = Array{Vector{Int}}(undef, length(B))
    B_weights = Array{Vector{Float64}}(undef, length(B))
    for (bi, b) in enumerate(B)
        if Bbao_data.B_in_Bbao[bi]
            empty!(model)
            B_overlap, valid_weights = Bbao_data.B_overlap[bi], Dict(bi => 1.0)
            B_entropies = map(bi -> Bbao_data.B_entropies[bi], B_overlap)
            (this_idxs, this_weights) = get_entropy_weights(b, B, B_overlap, B_entropies; model=model, initial_weights=valid_weights) 
            B_idxs[bi] = this_idxs; B_weights[bi] = this_weights
        end
    end

    # Then for all beliefs in Bbao, we do the same:
    Bbao_idxs = Array{Vector{Int}}(undef, length(Bbao_data.Bbao))
    Bbao_weights = Array{Vector{Float64}}(undef, length(Bbao_data.Bbao))
    for (bi, b) in enumerate(Bbao_data.Bbao)        
        empty!(model)
        B_overlap, valid_weights = Bbao_data.Bbao_overlap[bi], Bbao_data.Valid_weights[bi]
        B_entropies = map(bi -> Bbao_data.B_entropies[bi], B_overlap)
        (this_idxs, this_weights) = get_entropy_weights(b, B, B_overlap, B_entropies;model=model, initial_weights=valid_weights) 
        Bbao_idxs[bi] = this_idxs; Bbao_weights[bi] = this_weights
    end
    return Weights_Data(B_idxs,B_weights, Bbao_idxs, Bbao_weights) 
end 

"""Computes all max-entropy weights for Bbao"""
function get_entropy_weights(b, B, Bi_overlap, B_entropies; model=nothing, initial_weights=nothing)
    B_overlap = map(bi -> B[bi], Bi_overlap)
    if model isa Nothing
        model = Model(Clp.Optimizer; add_bridges=false)
        set_silent(model)
        set_string_names_on_creation(model, false)
    end

    @variable(model, 0.0 <= b_ps[1:length(B_overlap)] <= 1.0)
    # Build the constraint that probabilities for each state match that of b
    for (s, ps) in weighted_iterator(b)
        Idx, Ps = [], []
        for (bpi, bp) in enumerate(B_overlap)
            if ps > 0
                push!(Idx, bpi)
                push!(Ps,ps)
            end
        end
        length(Idx) > 0 && @constraint(model, sum(b_ps[Idx[i]] * Ps[i] for i in 1:length(Idx)) == ps )
    end
    @objective(model, Max, sum( b_ps.*B_entropies))
    # !(isnothing(initial_weights) && set_start_value.(b_ps, B_start) # Warm start is not used: it did not lead to improvements
    optimize!(model)

    # Unpack weight & idxs from problem:
    weights=[]
    idxs = []
    cumprob = 0.0
    for (bpi,bp) in enumerate(B_overlap)
        prob = Float64(JuMP.value(b_ps[bpi]))
        if prob > 0.0
            cumprob += prob
            real_bpi = Bi_overlap[bpi]
            push!(idxs, real_bpi)
            push!(weights, prob)
        end
    end
    return(idxs, weights)
end

########## Closest belief weights ##########

"""Returns the belief that has the lowest minratio with b (as well as that ratio)"""
function get_best_minratio(b::DiscreteHashedBelief, B::Vector{DiscreteHashedBelief}, B_overlap::Vector)
    best_bi, best_ratio = nothing, -Inf
    sup_b, n_sup_b = support(b), length(support(b))
    for bpi in B_overlap
        this_ratio = Inf
        bp = B[bpi]
        # sup_bp, n_sup_bp = support(bp), length(support(bp))
        # bidx, bpidx = 1, 1
        # while bidx <= n_sup_b && bpidx <= n_sup_bp
        #     sb, sbp = sup_b[bidx], sup_bp[bpidx]
        #     if sb == sbp
        #         this_ratio = min(this_ratio, b.probs[bidx] / bp.probs[bpidx])
        #         this_ratio <= best_ratio && break 
        #         bidx += 1; bpidx += 1
        #     elseif objectid(sb) < objectid(sbp)
        #         bidx += 1
        #     else
        #         print("huh?")
        #         break
        #     end
        # end
        # (bidx >= n_sup_b && bpidx <= n_sup_bp) && (this_ratio = 0.0)
        # if this_ratio > best_ratio
        #     best_bi = bpi
        #     best_ratio = this_ratio
        # end
        # # end
        # best_ratio == 1 && break
        for (sp, psp) in weighted_iterator(bp)
            this_ratio = min(this_ratio, pdf(b,sp) / psp)
            this_ratio <= best_ratio && break # Current belief is already suboptimal
        end
        this_ratio > best_ratio && (best_bi = bpi; best_ratio = this_ratio)
        best_ratio == 1 && break
    end
    return best_bi, best_ratio
end

"""Compute a weighting for b using only the belief in B with the highest minratio, plus exterior beliefs"""
function get_closeness_weight(b, B; B_overlap=nothing, Data=nothing)
    closest_bi, min_ratio = get_best_minratio(b,B,B_overlap)                # TODO: try this iteratively -> if its better, test both (may imply 'improved sawtooth' is valuable more generally)
    closest_b = B[closest_bi]
    weights, idxs = [min_ratio], [closest_bi]
    for (s, ps) in weighted_iterator(b)
        si = Data.S_dict[s]
        this_weight = ps - (min_ratio * pdf(closest_b,s))
        if this_weight != 0
            push!(weights, this_weight)
            push!(idxs, si)
        end
    end
    return idxs, weights
end

"""Computes all closest-belief weights for Bbao"""
function get_closeness_weights_all(B, Bbao_data, Data) #TODO: can probably be combined with get_entropy_weights_all
    B_idxs = Array{Vector{Int}}(undef, length(B))
    B_weights = Array{Vector{Float64}}(undef, length(B))
    for (bi, b) in enumerate(B)
        if Bbao_data.B_in_Bbao[bi]
            B_overlap = Bbao_data.B_overlap[bi]
            idxs, weights = get_closeness_weight(b, B; B_overlap=B_overlap, Data=Data)
            B_idxs[bi] = idxs; B_weights[bi] = weights
        end
    end 

    Bbao_idxs = Array{Vector{Int}}(undef, length(Bbao_data.Bbao))
    Bbao_weights = Array{Vector{Float64}}(undef, length(Bbao_data.Bbao))
    for (bi, b) in enumerate(Bbao_data.Bbao)
        B_overlap = Bbao_data.Bbao_overlap[bi]
        idxs, weights = get_closeness_weight(b,B;  B_overlap=B_overlap, Data=Data)
        Bbao_idxs[bi] = idxs; Bbao_weights[bi] = weights
    end
    return Weights_Data(B_idxs, B_weights,Bbao_idxs, Bbao_weights)
end
