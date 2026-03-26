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
function get_weights(Bbao_data::BBAO_Data{S}, weights_data::Weights_Data, bi::Int, ai::Int, oi::Int) where S
    in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
    if in_B
        return (weights_data.B_idxs[baoi], weights_data.B_weights[baoi])
    else
        return (weights_data.Bbao_idxs[baoi], weights_data.Bbao_weights[baoi])
    end
end

function get_all_weights(B::Vector{DiscreteHashedBelief{S}}, Bbao_data::BBAO_Data{S}, Data::TIB_Data{S}, compute_single_weights=get_single_closeness_weights; values = nothing) where S
    B_idxs = Array{Vector{Int}}(undef, length(B))
    B_weights = Array{Vector{Float64}}(undef, length(B))

    ### Compute weights for all beliefs in B and Bbao
    for (bi,b) in enumerate(B)
        if Bbao_data.B_in_Bbao[bi]
            relevant_belief_idxs = Bbao_data.B_overlap[bi]
            idxs, weights = compute_single_weights(b, relevant_belief_idxs, Data; values=values)
            B_idxs[bi] = idxs
            B_weights[bi] = weights 
        end
    end

    ### Compute weights for all beliefs in only Bbao
    Bbao_idxs = Array{Vector{Int}}(undef, length(Bbao_data.Bbao))
    Bbao_weights = Array{Vector{Float64}}(undef, length(Bbao_data.Bbao))
    for (bi, b) in enumerate(Bbao_data.Bbao)
        relevant_belief_idxs = Bbao_data.Bbao_overlap[bi]
        idxs, weights = compute_single_weights(b, relevant_belief_idxs, Data; values=values)
        Bbao_idxs[bi] = idxs; Bbao_weights[bi] = weights
    end
    return Weights_Data(B_idxs, B_weights,Bbao_idxs, Bbao_weights)
end

########## Entropy weights ##########

function get_all_entropy_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}; entropies=nothing) where S
    return get_all_weights(Data.B, Bbao_data, Data, get_single_entropy_weights; values = entropies)
end

function get_single_entropy_weights(b::DiscreteHashedBelief{S}, Bidxs_overlap, Data::TIB_Data; model=nothing, values=nothing) where S
    if model isa Nothing
        model = Model(Clp.Optimizer; add_bridges=false)
        set_silent(model)
        set_string_names_on_creation(model, false)
    end
    B_overlap = map(bi -> Data.B[bi], Bidxs_overlap)
    if values isa Nothing
        B_entropies = map(b -> get_entropy(b), B_overlap)
    else
        B_entropies = values[Bidxs_overlap]
    end
    
    @variable(model, 0.0 <= b_ps[1:length(B_overlap)] <= 1.0)
    # Build the constraint that probabilities for each state match that of b
    for (s, ps) in weighted_iterator(b)
        Idx, Ps = [], []
        for (bpi, bp) in enumerate(B_overlap)
            p = pdf(bp, s)
            if p > 0
                push!(Idx, bpi)
                push!(Ps,p)
            end
        end
        length(Idx) > 0 && @constraint(model, sum(b_ps[Idx[i]] * Ps[i] for i in 1:length(Idx)) == ps )
    end
    @objective(model, Max, sum( b_ps.*B_entropies))
    optimize!(model)

    # Unpack weight & idxs from problem:
    weights=[]
    idxs = []
    cumprob = 0.0
    for (bpi,bp) in enumerate(B_overlap)
        prob = Float64(JuMP.value(b_ps[bpi]))
        if prob > 0.0
            cumprob += prob
            real_bpi = Bidxs_overlap[bpi]
            push!(idxs, real_bpi)
            push!(weights, prob)
        end
    end
    return(idxs, weights)
end

########## Closest belief weights ##########

function get_all_closeness_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    return return get_all_weights(Data.B, Bbao_data, Data, get_single_closeness_weights)
end

"""Compute a weighting for b using only the belief in B with the highest minratio, plus exterior beliefs"""
function get_single_closeness_weights(b::DiscreteHashedBelief{S}, Bidxs::Vector{Int}, Data::TIB_Data{S}; values=nothing) where S
    weights, idxs = [], []

    ### Find closest non-unit belief & subtract it from b
    non_unit_beliefs = filter(bidx -> length(support(Data.B[bidx])) > 1, Bidxs)
    if length(non_unit_beliefs) > 0
        closest_bi, min_ratio = get_best_minratio(b, Data.B, non_unit_beliefs) 
        push!(weights, min_ratio)
        push!(idxs, closest_bi)
        brest = subtract_scaled_belief(b, Data.B[closest_bi], min_ratio)
        remaining_weight = 1.0-min_ratio
    else
        brest = b
        remaining_weight = 1.0
    end

    ### Add unit beliefs to represent belief
    for (s, ps) in weighted_iterator(brest)
        push!(weights, ps * remaining_weight)
        push!(idxs, Data.S_dict[s])
    end
    return idxs, weights
end

function get_all_iterative_closeness_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    return get_all_weights(Data.B, Bbao_data, Data, get_single_iterative_closeness_weights)
end

function get_single_iterative_closeness_weights(b::DiscreteHashedBelief{S}, Bidxs, Data::TIB_Data{S}; values=nothing) where S
    weights, idxs = [], []
    brest = deepcopy(b)
    remaining_weight = 1.0

    ### Iteratively find closest non-unit belief & 
    non_unit_beliefs = filter(bidx -> length(support(Data.B[bidx])) > 1, Bidxs)
    if length(non_unit_beliefs) > 0
        this_weight = 1.0
        while this_weight > 1e-5
            ### Find current closest belief
            closest_bi, min_ratio = get_best_minratio(brest, Data.B, non_unit_beliefs)
            this_weight = remaining_weight * min_ratio
            push!(idxs, closest_bi)
            push!(weights, this_weight)
            remaining_weight -= this_weight

            ### Compute 'remaining' belief
            brest = subtract_scaled_belief(brest, Data.B[closest_bi], min_ratio)
        end
    end

    ### Add unit beliefs to represent belief
    for (s, ps) in weighted_iterator(brest)
        push!(weights, ps * remaining_weight)
        push!(idxs, Data.S_dict[s])
    end
    return idxs, weights
end

"""Returns the belief that has the lowest minratio with b (as well as that ratio)"""
function get_best_minratio(b::DiscreteHashedBelief{S}, B::Vector{DiscreteHashedBelief{S}}, Bidxs::Vector{Int}) where S
    best_bi, best_ratio = nothing, -Inf
    best_bi_nonunit, best_ratio_nonunit = nothing, -Inf
    sup_b = support(b)
    n_sup_b = length(sup_b)
    for bpi in Bidxs
        bp = B[bpi]
        this_ratio = Inf
        sup_bp = support(bp)
        n_sup_bp = length(sup_bp)
        bidx, bpidx = 1, 1
        while bidx <= n_sup_b && bpidx <= n_sup_bp
            sb, sbp = sup_b[bidx], sup_bp[bpidx]
            if sb == sbp
                this_ratio = min(this_ratio, b.probs[bidx] / bp.probs[bpidx])
                this_ratio <= best_ratio && break 
                bidx += 1; bpidx += 1
            elseif sb < sbp
                bidx += 1
            else
                this_ratio = 0.0
                break
            end
        end
        (bidx > n_sup_b && bpidx <= n_sup_bp) && (this_ratio = 0.0)
        if this_ratio > best_ratio
            best_bi = bpi
            best_ratio = this_ratio
        end
        best_ratio == 1 && break
    end
    return best_bi, best_ratio
end

function subtract_scaled_belief(b::DiscreteHashedBelief{S}, bmin::DiscreteHashedBelief{S}, scale::Float64) where S
    sup_r = support(b)
    probs_r = b.probs
    n_r = length(sup_r)

    sup_b = support(bmin)
    probs_b = bmin.probs
    n_b = length(sup_b)

    new_probs = Vector{Float64}(undef, n_r)
    cum_prob = 0.0

    ridx, bidx = 1, 1

    while ridx <= n_r && bidx <= n_b
        sr = sup_r[ridx]
        sb = sup_b[bidx]

        if sr == sb
            prob = probs_r[ridx] - scale * probs_b[bidx]
            new_probs[ridx] = prob
            cum_prob += prob
            ridx += 1
            bidx += 1

        elseif sr < sb
            # state in brest but not in b_ref → prob_ref = 0
            prob = probs_r[ridx]
            new_probs[ridx] = prob
            cum_prob += prob
            ridx += 1

        else
            # state in b_ref but not in brest → irrelevant
            bidx += 1
        end
    end

    # Remaining states in brest (no overlap with b_ref)
    while ridx <= n_r
        prob = probs_r[ridx]
        new_probs[ridx] = prob
        cum_prob += prob
        ridx += 1
    end
    cum_prob > 1e-10 ? (new_probs = new_probs ./ cum_prob) : (fill!(new_probs, 0.0))
    return DiscreteHashedBelief{S}(b.state_list, new_probs)
end

########## optimal belief weights ##########

function get_all_optimal_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}; Qs=nothing) where S
    return get_all_weights(Data.B, Bbao_data, Data, get_single_optimal_weights; values = Qs)
end

function get_single_optimal_weights(b::DiscreteHashedBelief{S}, Bidxs::Vector{Int}, Data::TIB_Data{S}; model=nothing, values=nothing, return_value=false) where S
    if model isa Nothing
        model = Model(Clp.Optimizer; add_bridges=false)
        set_silent(model)
        set_string_names_on_creation(model, false)
    end

    MIN_Q_VALUE = -100_000
    Qs = map(q -> max(MIN_Q_VALUE, q), Data.Q[Bidxs, :])
    B = Data.B[Bidxs]

    @variable(model, 0.0 <= b_ps[1:length(B)] <= 1.0)
    @variable(model, Qmax)

    # Constraint 1: set must represent b
    for (s, ps) in weighted_iterator(b)
        Idx, Ps = [], []
        for (bpi, bp) in enumerate(B)
            p = pdf(bp,s)
            if p > 0
                push!(Idx, bpi)
                push!(Ps, p)
            end
        end
        length(Idx) > 0 && @constraint(model, sum(b_ps[Idx] .* Ps) == pdf(b,s) )
    end

    # Constraint 2: Qmax is Q of best action
    for ai in 1:Data.constants.na
        @constraint(model, Qmax >= sum(Qs[:,ai] .* b_ps))
    end

    @objective(model, Min, 1.0 * Qmax)
    optimize!(model)
    Q = objective_value(model)
    return_value && (return Q)

    return Bidxs, collect(JuMP.value.(b_ps))
end

########## Sawtooth belief weights ##########

function get_all_sawtooth_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}; Qs=nothing) where S
    return get_all_weights(Data.B, Bbao_data, Data, get_single_sawtooth_weights; values = Qs)
end

function get_single_sawtooth_weights(b::DiscreteHashedBelief{S}, Bidxs::Vector{Int}, Data::TIB_Data{S}; model=nothing, values=nothing, return_value=false) where S
    
    alpha_extremes = Data.Q[1:Data.constants.ns,:]
    Vb = Float64[]
    for ai in 1:Data.constants.na
        push!(Vb, dot(b, alpha_extremes[:,ai]))
    end
    best_value, best_bidx, best_ratio = maximum(Vb), 1, 0.0
    non_unit_beliefs = filter(bidx -> length(support(Data.B[bidx])) > 1, Bidxs)

    for bp_idx in non_unit_beliefs
        bp = Data.B[bp_idx]
        ratio = min_ratio(b, bp)
        for ai in 1:Data.constants.na
            this_value = Vb[ai] + ratio * (Data.Q[bp_idx, ai] - dot(bp, alpha_extremes[:,ai]))
            if this_value < best_value
                best_value = this_value
                best_bidx = bp_idx
                best_ratio = ratio 
            end
        end
    end
    weights, idxs = [best_ratio], [best_bidx]
    brest = subtract_scaled_belief(b, Data.B[best_bidx], best_ratio)
    for (s, ps) in weighted_iterator(brest)
        push!(weights, ps * (1-best_ratio))
        push!(idxs, Data.S_dict[s])
    end
    return idxs, weights
end

function get_all_iterative_sawtooth_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}; Qs=nothing) where S
    return get_all_weights(Data.B, Bbao_data, Data, get_single_iterative_sawtooth_weights; values = Qs)
end

function get_single_iterative_sawtooth_weights(b::DiscreteHashedBelief{S}, Bidxs::Vector{Int}, Data::TIB_Data{S}; model=nothing, values=nothing, return_value=false) where S
    
    alpha_extremes = Data.Q[1:Data.constants.ns,:]
    Vb = []
    for ai in 1:Data.constants.na
        push!(Vb, dot(b, alpha_extremes[:,ai]))
    end
    non_unit_beliefs = filter(bidx -> length(support(Data.B[bidx])) > 1, Bidxs)
    best_idxs, best_weights = [], []
    best_action_value = -Inf
    
    for ai in 1:Data.constants.na
        best_ratio, remaining_weight = 1.0, 1.0
        brest = deepcopy(b)
        weights, idxs = [], []
        while best_ratio > 1e-5
            best_value, best_bidx, best_ratio = maximum(Vb), -1, 0.0
            for bp_idx in non_unit_beliefs
                bp = Data.B[bp_idx]
                ratio = min_ratio(brest, bp)
                this_value = Vb[ai] + ratio * (Data.Q[bp_idx, ai] - dot(bp, alpha_extremes[:,ai]))
                if this_value < best_value
                    best_value = this_value
                    best_bidx = bp_idx
                    best_ratio = ratio 
                end
            end
            if best_ratio > 1e-5
                this_weight = remaining_weight * best_ratio 
                push!(idxs, best_bidx)
                push!(weights, this_weight)
                remaining_weight -= this_weight
                brest = subtract_scaled_belief(brest, Data.B[best_bidx], best_ratio)
                Vb = []
                for ai in 1:Data.constants.na
                    push!(Vb, dot(brest, alpha_extremes[:,ai]))
                end
                # println("$best_ratio, $brest, $(Data.B[best_bidx])")
            end
        end
        for (s, ps) in weighted_iterator(brest)
            push!(weights, ps * remaining_weight)
            push!(idxs, Data.S_dict[s])
        end
        this_value = sum(idx -> weights[idx] * Data.Q[idxs[idx], ai], 1:length(idxs))
        if this_value >= best_action_value 
            best_action_value = this_value
            best_idxs = idxs
            best_weights = weights
        end
    end

    # println("$(sum(best_weights)), $b, $(map(idx -> (best_weights[idx], Data.B[best_idxs[idx]]), 1:length(best_idxs)))")
    return best_idxs, best_weights
end

function min_ratio(b::DiscreteHashedBelief{S},bp::DiscreteHashedBelief{S}) where S
    minratio = Inf
    bidx = 1
    n_sup_b = length(b.state_list)
    n_sup_bp = length(bp.state_list)
    # n_sup_b != n_sup_bp && return 0.0
    bidx, bpidx = 1, 1
    while bidx <= n_sup_b && bpidx <= n_sup_bp
        sb, sbp = b.state_list[bidx], bp.state_list[bpidx]
        if sb == sbp
            minratio = min(minratio, b.probs[bidx] / bp.probs[bpidx])
            bidx += 1; bpidx += 1
        elseif objectid(sb) < objectid(sbp)
            bidx += 1
        else
            return 0.0
        end
    end
    bidx >= n_sup_b && bpidx <= n_sup_bp ? (return 0.0) : (return minratio)
end