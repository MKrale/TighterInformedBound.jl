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
    LP_data(cands) = (entropy_objective(cands, Data, entropies), nothing)
    return get_all_LP_weights(Data, Bbao_data, LP_data, w -> findall(>(0.0), w); maximize=true)
end

entropy_objective(Bidxs, Data::TIB_Data, values) =
    values isa Nothing ? map(bi -> Float64(get_entropy(Data.B[bi])), Bidxs) : Float64.(values[Bidxs])

function get_single_entropy_weights(b::DiscreteHashedBelief{S}, Bidxs_overlap, Data::TIB_Data; values=nothing) where S
    weights, _ = solve_weight_LP(b, Bidxs_overlap, Data, entropy_objective(Bidxs_overlap, Data, values); maximize=true)
    nonzero = findall(>(0.0), weights)
    return Bidxs_overlap[nonzero], weights[nonzero]
end

########## Closest belief weights ##########

function get_all_closeness_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    return return get_all_weights(Data.B, Bbao_data, Data, get_single_closeness_weights)
end

"""Compute a weighting for b using only the belief in B with the highest minratio, plus exterior beliefs"""
function get_single_closeness_weights(b::DiscreteHashedBelief{S}, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data{S}; values=nothing) where S
    weights, idxs = Float64[], Int[]

    ### Find closest non-unit belief & subtract it from b
    closest_bi, min_ratio = get_best_minratio(b, Data.B, Bidxs; skip_unit_beliefs=true)
    if !isnothing(closest_bi)
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

"""
Computes the iterative closeness weights (see get_single_iterative_closeness_weights) of all beliefs in Bbao (and of the beliefs
in B that are also in Bbao), with the same results. Per support, the (non-unit) candidates are stored as positions in the support,
and the remaining belief as a dense vector over it, so that no merging of supports or construction of beliefs is needed.
"""
function get_all_iterative_closeness_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    nb = length(Data.B)
    beliefs, overlaps, classes = support_classes(Data, Bbao_data)
    idxs, weights = Vector{Vector{Int}}(undef, length(beliefs)), Vector{Vector{Float64}}(undef, length(beliefs))
    for ids in classes
        sup = beliefs[ids[1]].state_list
        sidx = [Data.S_dict[s] for s in sup]
        # The non-unit candidates (in the order of the overlap list), with their states as positions in sup
        cand_bi, cand_start, cand_pos, cand_p = Int[], Int[1], Int[], Float64[]
        for bi in overlaps[ids[1]]
            bp = Data.B[bi]
            length(bp.state_list) == 1 && continue
            push!(cand_bi, bi)
            for (s, p) in zip(bp.state_list, bp.probs)
                push!(cand_pos, searchsortedfirst(sup, s)); push!(cand_p, p)
            end
            push!(cand_start, length(cand_pos) + 1)
        end
        r, active, r_new = zeros(length(sup)), falses(length(sup)), zeros(length(sup))
        for id in ids
            idxs[id], weights[id] = dense_iterative_closeness_weights!(r, active, r_new, beliefs[id].probs, sidx, cand_bi, cand_start, cand_pos, cand_p)
        end
    end
    return Weights_Data(idxs[1:nb], weights[1:nb], idxs[nb+1:end], weights[nb+1:end])
end

"""
get_single_iterative_closeness_weights for a belief with probabilities probs (over its support), using the remaining belief as a
dense vector r over the support, with active[k] iff state k is in its support (r[k] > 0). Follows get_best_minratio and
subtract_scaled_belief operation by operation, so the results are the same.
"""
function dense_iterative_closeness_weights!(r, active, r_new, probs, sidx, cand_bi, cand_start, cand_pos, cand_p)
    weights, idxs = Float64[], Int[]
    r .= probs
    fill!(active, true)
    remaining_weight = 1.0
    this_weight = 1.0
    while this_weight > 1e-5
        ### Find current closest belief (see get_best_minratio)
        best, best_ratio = 0, 0.0
        for c in eachindex(cand_bi)
            ratio = Inf
            for e in cand_start[c]:cand_start[c+1]-1
                k = cand_pos[e]
                active[k] || (ratio = 0.0; break)       # not a subset of the support of the remaining belief
                ratio = min(ratio, r[k] / cand_p[e])
                ratio > best_ratio || break             # cannot beat the best ratio
            end
            if ratio > best_ratio
                best, best_ratio = c, ratio
            end
            best_ratio == 1 && break
        end
        best == 0 && break
        this_weight = remaining_weight * best_ratio
        push!(idxs, cand_bi[best])
        push!(weights, this_weight)
        remaining_weight -= this_weight

        ### Compute 'remaining' belief (see subtract_scaled_belief)
        r_new .= r
        for e in cand_start[best]:cand_start[best+1]-1
            k = cand_pos[e]
            r_new[k] = r[k] - best_ratio * cand_p[e]
        end
        cum_prob = 0.0
        for k in eachindex(r)
            active[k] && (cum_prob += r_new[k])
        end
        for k in eachindex(r)
            if active[k]
                r[k] = cum_prob > 1e-10 ? r_new[k] / cum_prob : 0.0
                active[k] = r[k] > 0
            end
        end
    end

    ### Add unit beliefs to represent belief
    for k in eachindex(r)
        active[k] || continue
        push!(weights, r[k] * remaining_weight)
        push!(idxs, sidx[k])
    end
    return idxs, weights
end

function get_single_iterative_closeness_weights(b::DiscreteHashedBelief{S}, Bidxs, Data::TIB_Data{S}; values=nothing) where S
    weights, idxs = Float64[], Int[]
    brest = b   # not modified: subtract_scaled_belief returns a new belief
    remaining_weight = 1.0

    ### Iteratively find closest non-unit belief & 
    this_weight = 1.0
    while this_weight > 1e-5
        ### Find current closest belief
        closest_bi, min_ratio = get_best_minratio(brest, Data.B, Bidxs; skip_unit_beliefs=true)
        isnothing(closest_bi) && break
        this_weight = remaining_weight * min_ratio
        push!(idxs, closest_bi)
        push!(weights, this_weight)
        remaining_weight -= this_weight

        ### Compute 'remaining' belief
        brest = subtract_scaled_belief(brest, Data.B[closest_bi], min_ratio)
    end

    ### Add unit beliefs to represent belief
    for (s, ps) in weighted_iterator(brest)
        push!(weights, ps * remaining_weight)
        push!(idxs, Data.S_dict[s])
    end
    return idxs, weights
end

"""
Returns the belief that has the highest minratio with b (as well as that ratio), or (nothing, 0.0) if no belief has a positive
minratio. If skip_unit_beliefs, beliefs with a single state are ignored.
"""
function get_best_minratio(b::DiscreteHashedBelief{S}, B::Vector{DiscreteHashedBelief{S}}, Bidxs::AbstractVector{<:Integer}; skip_unit_beliefs=false) where S
    best_bi, best_ratio = nothing, 0.0
    best_bi_nonunit, best_ratio_nonunit = nothing, -Inf
    sup_b = support(b)
    n_sup_b = length(sup_b)
    for bpi in Bidxs
        bp = B[bpi]
        skip_unit_beliefs && length(bp.state_list) == 1 && continue
        maybe_support_subset(bp, b) || continue     # supp(bp) is not a subset of supp(b), so the ratio is 0
        this_ratio = Inf
        sup_bp = support(bp)
        n_sup_bp = length(sup_bp)
        bidx, bpidx = 1, 1
        # Stop as soon as bp can no longer beat the best ratio so far
        while this_ratio > best_ratio && bidx <= n_sup_b && bpidx <= n_sup_bp
            sb, sbp = sup_b[bidx], sup_bp[bpidx]
            if sb == sbp
                this_ratio = min(this_ratio, b.probs[bidx] / bp.probs[bpidx])
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
    LP_data(cands) = (zeros(length(cands)), optimal_Qs(cands, Data))
    # A basic LP solution has at most |supp(b)|+1 nonzeros, so dropping the zeros makes later iterations much cheaper
    return get_all_LP_weights(Data, Bbao_data, LP_data, w -> findall(!iszero, w))
end

optimal_Qs(Bidxs, Data::TIB_Data) = map(q -> max(MIN_Q_VALUE, q), Data.Q[Bidxs, :])

function get_single_optimal_weights(b::DiscreteHashedBelief{S}, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data{S}; values=nothing, return_value=false) where S
    # Minimize Qmax, the Q-value of the best action
    weights, Q = solve_weight_LP(b, Bidxs, Data, zeros(length(Bidxs)); Qs=optimal_Qs(Bidxs, Data))
    return_value && (return Q)

    # A basic LP solution has at most |supp(b)|+1 nonzeros, so dropping the zeros makes later iterations much cheaper
    nonzero = findall(!iszero, weights)
    return Bidxs[nonzero], weights[nonzero]
end

const MIN_Q_VALUE = -100_000.0
const MIN_LP_COEF = 1e-12
const LP_PRIMAL_TOLERANCE = 1e-10
const LP_DUAL_TOLERANCE = 1e-9         # for verifying that a basis is optimal (see CachedBasis), relative to the LP's coefficients

"""
The LP over weights w in [0,1]^n (one per belief in Bidxs) subject to sum_j w[j] * B[Bidxs[j]](s) == b(s) for all s in supp(b),
optimizing sum_j obj[j] * w[j]. If Qs is given, a free variable Qmax is added (with objective coefficient 1) and constrained by
Qmax >= sum_j Qs[j,a] * w[j] for every action a. It assumes the support of each belief in Bidxs is a subset of the support of b.
Apart from the right-hand side b(s), the LP only depends on the support of b, so it can be solved for any belief with that support.

The LP is loaded straight into Clp in column-major form: building it through JuMP took 3-5x longer than solving it.
States not covered by any belief are skipped, and Q coefficients below MIN_LP_COEF are dropped: Q-values that should be 0 but
carry rounding noise (e.g. 6.6e-16 in pentagon) make Clp scale their column so that its weight drops below the tolerance,
leaving b unrepresented while Clp still reports optimality.
"""
struct WeightLP
    n::Int                      # number of weights
    na::Int                     # number of actions (0 for the entropy LP)
    neq::Int                    # number of (covered) states, i.e., rows of the constraint that the weights represent b
    covered::BitVector          # states of supp(b) with a row (covered by some belief)
    colstart::Vector{Cint}      # the constraint matrix (columns: weights, then Qmax; rows: covered states, then actions)
    rowidx::Vector{Cint}
    vals::Vector{Float64}
    collb::Vector{Float64}
    colub::Vector{Float64}
    obj::Vector{Float64}
    maximize::Bool
end

function WeightLP(b::DiscreteHashedBelief, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data, obj::Vector{Float64}; Qs=nothing, maximize=false)
    n, ns = length(Bidxs), length(b.state_list)
    na = Qs isa Nothing ? 0 : size(Qs, 2)
    ncols = Qs isa Nothing ? n : n + 1
    colstart = Vector{Cint}(undef, ncols + 1)
    rowidx, vals = Cint[], Float64[]
    covered = falses(ns)
    # Rows are stored as k (state k of b) or -a (action a), and renumbered once we know which states are covered
    for (j, bi) in enumerate(Bidxs)
        colstart[j] = length(rowidx)
        bp = Data.B[bi]
        # Constraint 1: set must represent b
        for (s, p) in zip(bp.state_list, bp.probs)
            k = searchsortedfirst(b.state_list, s)
            push!(rowidx, k); push!(vals, p); covered[k] = true
        end
        # Constraint 2: Qmax is Q of best action (only when Qs is given; for the entropy LP na = 0)
        for ai in 1:na
            q = Qs[j, ai]
            abs(q) < MIN_LP_COEF || (push!(rowidx, -ai); push!(vals, -q))
        end
    end
    if na > 0   # Qmax column
        colstart[n+1] = length(rowidx)
        for ai in 1:na
            push!(rowidx, -ai); push!(vals, 1.0)
        end
    end
    colstart[ncols+1] = length(rowidx)
    staterow = cumsum(covered)
    neq = isempty(staterow) ? 0 : staterow[end]
    for (i, r) in enumerate(rowidx)
        rowidx[i] = r > 0 ? staterow[r] - 1 : neq - r - 1
    end
    collb, colub = zeros(ncols), ones(ncols)
    na > 0 && (collb[end] = -Inf; colub[end] = Inf; obj = [obj; 1.0])
    return WeightLP(n, na, neq, covered, colstart, rowidx, vals, collb, colub, obj, maximize)
end

ncols(L::WeightLP) = length(L.obj)
nrows(L::WeightLP) = L.neq + L.na

"""
Solves the LP for belief b with Clp, from scratch. Returns the values of all columns, the objective value, and Clp's basis status
(of the columns, then the rows). The entropy LP (maximize) is solved with primal simplex, the Qmax LP with dual simplex: both are ~2x faster
than Clp's initialSolve (presolve + algorithm choice), and give the same results up to ties.
"""
function clp_solve(L::WeightLP, b::DiscreteHashedBelief)
    rhs = b.probs[L.covered]
    rowlb, rowub = [rhs; zeros(L.na)], [rhs; fill(Inf, L.na)]
    model = Clp.Clp_newModel()
    try
        Clp.Clp_setLogLevel(model, 0)
        Clp.Clp_loadProblem(model, ncols(L), nrows(L), L.colstart, L.rowidx, L.vals, L.collb, L.colub, L.obj, rowlb, rowub)
        Clp.Clp_setObjSense(model, L.maximize ? -1 : 1)
        # Clp's default (1e-7, relative to its scaled problem) let weights miss b by up to ~1e-5 when b has tiny probabilities
        Clp.Clp_setPrimalTolerance(model, LP_PRIMAL_TOLERANCE)
        L.na > 0 ? Clp.Clp_dual(model, 0) : Clp.Clp_primal(model, 0)
        status = Clp.Clp_status(model)
        status == 0 || error("Clp did not solve the weight LP to optimality (status $status)")
        x = copy(unsafe_wrap(Array, Clp.Clp_getColSolution(model), ncols(L)))
        status = copy(unsafe_wrap(Array, Clp.Clp_statusArray(model), ncols(L) + nrows(L)))
        return x, Clp.Clp_getObjValue(model), status
    finally
        Clp.Clp_deleteModel(model)
    end
end

"""Solves the weight LP (see WeightLP) for a single belief. Returns the weights and the objective value."""
function solve_weight_LP(b::DiscreteHashedBelief, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data, obj::Vector{Float64}; Qs=nothing, maximize=false)
    L = WeightLP(b, Bidxs, Data, obj; Qs=Qs, maximize=maximize)
    x, objective, _ = clp_solve(L, b)
    return x[1:L.n], objective
end

"""
An optimal basis of a WeightLP, written as A x - r = 0, where r are the row activities. The nonbasic columns are at a bound (0 or
1 for the weights), the nonbasic row activities of the states equal b(s), and those of the actions 0 (their only finite bound).
So the basic variables z (columns basic_cols, then row activities basic_rows) solve M z = rhs(b), with M = [A_basic_cols, -I_basic_rows],
of which we store the LU factorization (factors and row permutation).
(We only use which variables Clp reports as basic, and which bound nonbasic columns are at: Clp's signs for rows are confusing.)
"""
struct CachedBasis
    basic_cols::Vector{Int}
    basic_rows::Vector{Int}
    upper_cols::Vector{Int}         # nonbasic columns at their upper bound
    LU::Matrix{Float64}
    perm::Vector{Int}
    basic_state_rows::Bool          # are some row activities of states basic (degenerate)?
end

"""Dense constraint matrix of the LP (rows: states, then actions)"""
function dense_matrix(L::WeightLP)
    A = zeros(nrows(L), ncols(L))
    for j in 1:ncols(L), e in L.colstart[j]+1:L.colstart[j+1]
        A[L.rowidx[e]+1, j] = L.vals[e]
    end
    return A
end

const MAX_BASIS_CONDITION = 1e10    # bases that are worse conditioned are not cached

"""
Returns the basis of the LP given by Clp's basis status, if it is optimal for all right-hand sides, i.e., dual feasible: we check
this ourselves (in the original data), since Clp checks its tolerances on its scaled problem. Returns nothing otherwise.
"""
function CachedBasis(L::WeightLP, A::Matrix{Float64}, status::Vector{UInt8})
    m, nc = nrows(L), ncols(L)
    basic_cols, upper_cols, lower_cols = Int[], Int[], Int[]
    for j in 1:nc
        st = status[j] & 0x07     # 1: basic, 2: at upper bound, 3: at lower bound, 5: fixed; 0 (free) and 4 (superbasic) are not at a bound
        if st == 0x01
            push!(basic_cols, j)
        elseif (st == 0x02 || st == 0x03 || st == 0x05) && isfinite(L.collb[j]) && isfinite(L.colub[j])
            push!(st == 0x02 ? upper_cols : lower_cols, j)
        else
            return nothing
        end
    end
    basic_rows = [i for i in 1:m if status[nc + i] & 0x07 == 0x01]
    length(basic_cols) + length(basic_rows) == m || return nothing
    M = [A[:, basic_cols] -Matrix{Float64}(I, m, m)[:, basic_rows]]
    F = lu(M; check=false)
    issuccess(F) && cond(M) <= MAX_BASIS_CONDITION || return nothing
    # Dual feasibility (for minimizing c = sign * obj): the duals y solve M' y = c_basic (row activities have cost 0), and the
    # reduced costs c_j - y A_j must be >= 0 for columns at their lower bound, <= 0 at their upper bound, and y_i >= 0 for the
    # nonbasic activities of the action rows (at their lower bound 0; column -e_i). State rows are fixed, so they have no condition.
    sgn = L.maximize ? -1.0 : 1.0
    y = F' \ [sgn .* L.obj[basic_cols]; zeros(length(basic_rows))]
    tol = LP_DUAL_TOLERANCE * (1 + maximum(abs, L.obj) + maximum(abs, A))     # (relative to the size of the reduced costs' terms)
    for j in lower_cols; sgn * L.obj[j] - LinearAlgebra.dot(y, view(A, :, j)) >= -tol || return nothing; end
    for j in upper_cols; sgn * L.obj[j] - LinearAlgebra.dot(y, view(A, :, j)) <= tol || return nothing; end
    for i in L.neq+1:m
        i in basic_rows || y[i] >= -tol || return nothing
    end
    return CachedBasis(basic_cols, basic_rows, upper_cols, F.factors, F.p, any(<=(L.neq), basic_rows))
end

"""Solves LU[perm] z = rhs, with LU the factors of an LU factorization (unit lower triangular L, upper triangular U)"""
function lu_solve!(z::Vector{Float64}, LU::Matrix{Float64}, perm::Vector{Int}, rhs::Vector{Float64})
    m = length(z)
    @inbounds for i in 1:m
        s = rhs[perm[i]]
        for k in 1:i-1; s -= LU[i,k] * z[k]; end
        z[i] = s
    end
    @inbounds for i in m:-1:1
        s = z[i]
        for k in i+1:m; s -= LU[i,k] * z[k]; end
        z[i] = s / LU[i,i]
    end
    return z
end

"""
Solves the LP for b with a cached optimal basis: since the basis is dual feasible, it is optimal for b if its basic solution for b
is feasible. Returns the values of all columns, or nothing. rhs and z are work arrays of length nrows(L).
"""
function solve_with_basis!(rhs::Vector{Float64}, z::Vector{Float64}, L::WeightLP, C::CachedBasis, b::DiscreteHashedBelief)
    k = 0
    for (p, c) in zip(b.probs, L.covered)
        c && (k += 1; rhs[k] = p)           # nonbasic row activities of the states (basic ones are moved into M, see below)
    end
    for i in L.neq+1:length(rhs); rhs[i] = 0.0; end
    for j in C.upper_cols, e in L.colstart[j]+1:L.colstart[j+1]
        rhs[L.rowidx[e]+1] -= L.vals[e] * L.colub[j]
    end
    bstate = C.basic_state_rows ? rhs[1:L.neq] : rhs      # (only used if some row activities of states are basic)
    for i in C.basic_rows; rhs[i] = 0.0; end
    lu_solve!(z, C.LU, C.perm, rhs)
    tol = LP_PRIMAL_TOLERANCE
    for (k, j) in enumerate(C.basic_cols)
        L.collb[j] - tol <= z[k] <= L.colub[j] + tol || return nothing
    end
    nb = length(C.basic_cols)
    for (k, i) in enumerate(C.basic_rows)
        # (a basic state row activity must still equal b(s); the activity Qmax - sum_j Q_j,a w_j of an action row must be >= 0)
        (i <= L.neq ? abs(z[nb+k] - bstate[i]) <= tol : z[nb+k] >= -tol) || return nothing
    end
    x = zeros(ncols(L))
    for j in C.upper_cols; x[j] = L.colub[j]; end
    for (k, j) in enumerate(C.basic_cols)
        x[j] = clamp(z[k], L.collb[j], L.colub[j])
    end
    return x
end

const MAX_DUAL_PIVOTS = 50      # dual simplex pivots (see dual_simplex) before falling back to Clp
const MAX_DUAL_SIMPLEX_ROWS = 18    # larger LPs are left to Clp (for them, our dense dual simplex is slower and needs more pivots)

"""
Solves the LP for b with a (dense) dual simplex, starting from the cached optimal basis C of another right-hand side, which is still
dual feasible. Starting from the cached basis that is closest to being feasible for b, only a few pivots are typically needed.
Returns Clp-style basis status of the optimal basis (of the columns, then the rows; 1: basic, 2: at upper bound, 3: at lower bound),
or nothing if it takes too many pivots.

Variables v = 1..nc are the columns, nc+i the activity of row i (column -e_i in A x - r = 0, cost 0, bounds b(s) for states and
[0, Inf] for actions). With costs c (for minimization), basic variables z = B^-1 (-N x_N) and duals y = B^-T c_B, a basic variable
p outside its bounds leaves, and enters the nonbasic variable v that moves it towards that bound and has the smallest |d_v / alpha_v|,
with d_v = c_v - y a_v its reduced cost and alpha_v = (B^-T e_p) a_v: this keeps the basis dual feasible.
"""
function dual_simplex(L::WeightLP, A::Matrix{Float64}, C::CachedBasis, b::DiscreteHashedBelief)
    m, nc = nrows(L), ncols(L)
    nv = nc + m
    sgn = L.maximize ? -1.0 : 1.0
    lbv, ubv, cost = zeros(nv), zeros(nv), zeros(nv)
    for j in 1:nc
        lbv[j], ubv[j], cost[j] = L.collb[j], L.colub[j], sgn * L.obj[j]
    end
    k = 0
    for (p, c) in zip(b.probs, L.covered)
        c && (k += 1; lbv[nc+k] = p; ubv[nc+k] = p)
    end
    for i in L.neq+1:m; ubv[nc+i] = Inf; end
    basic = [C.basic_cols; nc .+ C.basic_rows]
    is_basic, at_upper = falses(nv), falses(nv)
    is_basic[basic] .= true
    at_upper[C.upper_cols] .= true
    tol = LP_PRIMAL_TOLERANCE
    Bm, rhs, e_p, cB = zeros(m, m), zeros(m), zeros(m), zeros(m)
    alphaA, yA = zeros(nc), zeros(nc)
    for _ in 0:MAX_DUAL_PIVOTS
        fill!(Bm, 0.0)
        for (k, v) in enumerate(basic)
            v <= nc ? (Bm[:, k] .= view(A, :, v)) : (Bm[v - nc, k] = -1.0)
            cB[k] = cost[v]
        end
        F = lu(Bm; check=false)
        issuccess(F) || return nothing
        # z = B^-1 (-N x_N), with nonbasic variables at a bound
        fill!(rhs, 0.0)
        for v in 1:nc
            is_basic[v] && continue
            xv = at_upper[v] ? ubv[v] : lbv[v]
            isfinite(xv) || return nothing
            xv == 0.0 || (rhs .-= xv .* view(A, :, v))
        end
        for i in 1:m
            v = nc + i
            is_basic[v] || (rhs[i] += at_upper[v] ? ubv[v] : lbv[v])
        end
        z = F \ rhs
        # Leaving variable: the basic variable furthest outside its bounds
        p, viol, below = 0, tol, false
        for (k, v) in enumerate(basic)
            lbv[v] - z[k] > viol && ((p, viol, below) = (k, lbv[v] - z[k], true))
            z[k] - ubv[v] > viol && ((p, viol, below) = (k, z[k] - ubv[v], false))
        end
        if p == 0       # primal feasible, so optimal
            status = fill(0x03, nv)
            for v in 1:nv; at_upper[v] && (status[v] = 0x02); end
            for v in basic; status[v] = 0x01; end
            return status
        end
        fill!(e_p, 0.0); e_p[p] = 1.0
        rho, y = F' \ e_p, F' \ cB
        mul!(alphaA, A', rho); mul!(yA, A', y)
        # Entering variable (ratio test). z_p changes by -theta alpha_v when v moves by theta (> 0 at its lower, < 0 at its upper bound).
        q, best, bestalpha = 0, Inf, 0.0
        for v in 1:nv
            (is_basic[v] || lbv[v] == ubv[v]) && continue
            av, dv = v <= nc ? (alphaA[v], cost[v] - yA[v]) : (-rho[v - nc], y[v - nc])
            abs(av) <= 1e-12 && continue
            # z_p must increase (below its lower bound) or decrease (above its upper bound)
            (below ? (at_upper[v] ? av > 0 : av < 0) : (at_upper[v] ? av < 0 : av > 0)) || continue
            ratio = abs(dv / av)
            if ratio < best || (ratio == best && abs(av) > abs(bestalpha))
                q, best, bestalpha = v, ratio, av
            end
        end
        q == 0 && return nothing        # (LP infeasible: cannot happen for the weight LPs)
        leaving = basic[p]
        is_basic[leaving], at_upper[leaving] = false, !below
        basic[p] = q
        is_basic[q], at_upper[q] = true, false
    end
    return nothing
end

"""
Like solve_with_basis!, but returns how far the basic solution of the cached basis C for b is from being feasible (the total bound
violation; 0 if it is optimal for b). The basic values are left in z.
"""
function basis_infeasibility!(rhs::Vector{Float64}, z::Vector{Float64}, L::WeightLP, C::CachedBasis, b::DiscreteHashedBelief)
    k = 0
    for (p, c) in zip(b.probs, L.covered)
        c && (k += 1; rhs[k] = p)
    end
    for i in L.neq+1:length(rhs); rhs[i] = 0.0; end
    for j in C.upper_cols, e in L.colstart[j]+1:L.colstart[j+1]
        rhs[L.rowidx[e]+1] -= L.vals[e] * L.colub[j]
    end
    viol = 0.0
    nb = length(C.basic_cols)
    if C.basic_state_rows   # (rare: a basic state row activity must still equal b(s))
        bstate = rhs[1:L.neq]
        for i in C.basic_rows; rhs[i] = 0.0; end
        lu_solve!(z, C.LU, C.perm, rhs)
        for (k, i) in enumerate(C.basic_rows); i <= L.neq && (viol += abs(z[nb+k] - bstate[i])); end
    else
        for i in C.basic_rows; rhs[i] = 0.0; end
        lu_solve!(z, C.LU, C.perm, rhs)
    end
    for (k, j) in enumerate(C.basic_cols)
        viol += max(0.0, L.collb[j] - z[k]) + max(0.0, z[k] - L.colub[j])
    end
    for (k, i) in enumerate(C.basic_rows)
        i > L.neq && (viol += max(0.0, -z[nb+k]))
    end
    return viol
end

const BASIS_CACHE_TRIES = 1000  # how many cached bases are tried before calling Clp

"""
Returns all beliefs of B and Bbao (id <= nb for B), their candidate beliefs (overlaps), and the ids of the beliefs that need
weights (all of Bbao, and the beliefs in B that are also in Bbao), grouped by support (in order of first appearance).
The candidates of a belief only depend on its support.
"""
function support_classes(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    nb = length(Data.B)
    beliefs = [Data.B; Bbao_data.Bbao]
    overlaps = [Bbao_data.B_overlap; Bbao_data.Bbao_overlap]
    class_of = Dict{Vector{S},Int}()
    classes = Vector{Int}[]
    for id in eachindex(beliefs)
        id <= nb && !Bbao_data.B_in_Bbao[id] && continue
        c = get!(class_of, beliefs[id].state_list, length(classes) + 1)
        c > length(classes) && push!(classes, Int[])
        push!(classes[c], id)
    end
    return beliefs, overlaps, classes
end

"""
Computes the LP weights of all beliefs in Bbao (and of the beliefs in B that are also in Bbao). The candidate beliefs (and hence
the LP, apart from its right-hand side) of a belief only depend on its support, and most supports are shared by hundreds of beliefs.
For each support we keep the (verified) optimal bases found so far, and first try those (most recently used first) before calling
Clp: this avoids most LP solves. Solutions are computed from the basis, so they satisfy the constraints up to rounding (Clp's own
solutions can miss them by up to the tolerance of its scaled problem). LP_data(cands) returns the objective and Qs for the
candidates, and keep(w) the weights that are kept.
"""
function get_all_LP_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}, LP_data, keep; maximize=false) where S
    nb = length(Data.B)
    beliefs, overlaps, classes = support_classes(Data, Bbao_data)
    idxs, weights = Vector{Vector{Int}}(undef, length(beliefs)), Vector{Vector{Float64}}(undef, length(beliefs))
    for ids in classes
        cands = overlaps[ids[1]]
        obj, Qs = LP_data(cands)
        L = WeightLP(beliefs[ids[1]], cands, Data, obj; Qs=Qs, maximize=maximize)
        A = dense_matrix(L)
        rhs, z = zeros(nrows(L)), zeros(nrows(L))
        cache = CachedBasis[]
        for id in ids
            b, x = beliefs[id], nothing
            closest, closest_viol = 0, Inf
            for (k, C) in enumerate(cache)
                k > BASIS_CACHE_TRIES && break
                viol = basis_infeasibility!(rhs, z, L, C, b)
                viol < closest_viol && ((closest, closest_viol) = (k, viol))
                viol <= LP_PRIMAL_TOLERANCE || continue
                x = solve_with_basis!(rhs, z, L, C, b)
                if !isnothing(x)
                    k > 1 && (deleteat!(cache, k); pushfirst!(cache, C))     # most recently used first
                    break
                end
            end
            if isnothing(x) && closest > 0 && nrows(L) <= MAX_DUAL_SIMPLEX_ROWS
                # Dual simplex from the cached basis that is closest to being feasible
                status = dual_simplex(L, A, cache[closest], b)
                C = isnothing(status) ? nothing : CachedBasis(L, A, status)
                if !isnothing(C)
                    x = solve_with_basis!(rhs, z, L, C, b)
                    isnothing(x) || pushfirst!(cache, C)
                end
            end
            if isnothing(x)
                x, _, status = clp_solve(L, b)
                C = CachedBasis(L, A, status)
                if !isnothing(C)
                    pushfirst!(cache, C)
                    xC = solve_with_basis!(rhs, z, L, C, b)
                    isnothing(xC) || (x = xC)
                end
            end
            w = x[1:L.n]
            k = keep(w)
            idxs[id], weights[id] = cands[k], w[k]
        end
    end
    return Weights_Data(idxs[1:nb], weights[1:nb], idxs[nb+1:end], weights[nb+1:end])
end

########## Max-min optimal weights (OATIB) ##########

const MAXMIN_TOLERANCE = 1e-9   # (relative) actions whose values are this close count as tied, and solutions this close as equal

"""
OATIB's weights. Set 1 holds the optimal weights w* of get_all_optimal_weights, which solve min_w max_a sum_j w_j Q_j(a). For the
beliefs where w* does not also solve max_a min_w sum_j w_j Q_j(a), the later sets hold the extra weights needed for that (see
maxmin_extra_weights); for all other beliefs they are left #undef. Combined as max_a min_k over the sets (MultiWeightedColumns{true}),
this gives the max-min value, which is at most the min-max value and still an upper bound, since every set represents the belief.
"""
function get_all_maxmin_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    # Set 1: the min-max optimal weights of all beliefs (as for OTIB)
    W1 = get_all_optimal_weights(Data, Bbao_data)
    nb, nbao = length(W1.B_idxs), length(W1.Bbao_idxs)
    sets, nfailed = [W1], 0

    # Go through all beliefs that have weights in set 1. For each, we need its candidate beliefs (overlaps) and where to store its weights in a Weights_Data.
    for (inB, beliefs, overlaps, idxs, weights) in ((true, Data.B, Bbao_data.B_overlap, W1.B_idxs, W1.B_weights),
                                                    (false, Bbao_data.Bbao, Bbao_data.Bbao_overlap, W1.Bbao_idxs, W1.Bbao_weights))
        for i in eachindex(idxs)
            isassigned(idxs, i) || continue
            # Find the extra weights this belief needs for the max-min value (none for most beliefs)
            extra, failed = maxmin_extra_weights(beliefs[i], overlaps[i], Data, idxs[i], weights[i])
            nfailed += failed
            # Store the k-th extra weights of the belief in set k+1.
            for (k, (ik, wk)) in enumerate(extra)
                if k + 1 > length(sets)
                    push!(sets, Weights_Data(Vector{Vector{Int}}(undef, nb), Vector{Vector{Float64}}(undef, nb),
                                             Vector{Vector{Int}}(undef, nbao), Vector{Vector{Float64}}(undef, nbao)))
                end
                W = sets[k+1]
                inB ? (W.B_idxs[i] = ik; W.B_weights[i] = wk) : (W.Bbao_idxs[i] = ik; W.Bbao_weights[i] = wk)
            end
        end
    end
    # A failed LP only means we may miss some extra weights: the weights we have still give a valid (if looser) bound
    nfailed > 0 && @warn "OATIB: $nfailed per-action weight LPs failed; the weights found so far were kept (valid, but maybe not max-min)"
    return sets
end

"""
The extra weights (as (idxs, weights) pairs) that are needed, together with the optimal weights w* (idxs, weights) of belief b, to get
max_a min_w sum_j w_j Q_j(a) over the candidates Bidxs, and the number of LPs that failed. With u[a] = sum_j w*_j Q_j(a):
1. If only one action a attains max_a u[a], then near w* the maximum over actions equals u[a], so w* also minimizes the (linear) value of
   a: max-min equals min-max, and no extra weights are needed.
2. Otherwise, the LP min_w sum_j w_j Q_j(a) is solved for the actions in order of decreasing u[a] (an upper bound on its value), until
   u[a] is at most the best value found. The weights of an action are kept if they improve on u[a]: even if they don't lower the
   max-min value for the current Q, they can for the Q-values of later iterations.
"""
function maxmin_extra_weights(b::DiscreteHashedBelief, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data, idxs, weights)
    na = Data.constants.na
    extra = Tuple{Vector{Int},Vector{Float64}}[]

    # Value of each action under w* (with the same clipped Q-values as the LP, see optimal_Qs)
    u = zeros(na)
    for (j, w) in zip(idxs, weights), a in 1:na
        u[a] += w * max(MIN_Q_VALUE, Data.Q[j,a])
    end
    v = maximum(u)      # the min-max value
    tol = MAXMIN_TOLERANCE * (1 + abs(v))

    # Case 1: a single best action certifies that max-min equals min-max, we're done
    count(>(v - tol), u) == 1 && return extra, 0

    # Case 2: no such action exists. Then, we need to find worst-case weights for all actions that obtain a higher value
    Qs = optimal_Qs(Bidxs, Data)
    best, failed = -Inf, 0
    for a in sortperm(u; rev=true)
        u[a] <= best + tol && break
        # Minimize the value of action a over all weights that represent b
        obj = [abs(q) < MIN_LP_COEF ? 0.0 : q for q in view(Qs, :, a)]     # (see WeightLP on tiny coefficients)
        wa, va = try
            solve_weight_LP(b, Bidxs, Data, obj)
        catch e
            # Clp occasionally fails on these LPs: skip the action. The other sets still bound its value, so this stays valid.
            e isa ErrorException || rethrow()
            failed += 1
            continue
        end
        best = max(best, va)
        # Keep the weights if they lower the value of a (dropping zero weights, as in get_single_optimal_weights)
        if va < u[a] - tol
            nonzero = findall(!iszero, wa)
            push!(extra, (Bidxs[nonzero], wa[nonzero]))
        end
    end
    return extra, failed
end

"""The max-min value max_a min_w sum_j w_j Q_j(a) of belief b over the candidates Bidxs (for OATIB's policy, see maxmin_extra_weights)."""
function get_single_maxmin_value(b::DiscreteHashedBelief{S}, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data{S}) where S
    # Min-max optimal weights, and the value of each action under them
    idxs, weights = get_single_optimal_weights(b, Bidxs, Data)
    Qa = vec(sum(weights .* Data.Q[idxs,:], dims=1))
    # For each action, take the minimum over all sets of weights (as target_value does for MultiWeightedColumns{true})
    for (ik, wk) in first(maxmin_extra_weights(b, Bidxs, Data, idxs, weights))
        Qa = min.(Qa, vec(sum(wk .* Data.Q[ik,:], dims=1)))
    end
    return maximum(Qa)
end

########## Sawtooth belief weights ##########

function get_all_sawtooth_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}; Qs=nothing) where S
    return get_all_weights(Data.B, Bbao_data, Data, get_single_sawtooth_weights; values = Qs)
end

function get_single_sawtooth_weights(b::DiscreteHashedBelief{S}, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data{S}; model=nothing, values=nothing, return_value=false) where S
    
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

function get_single_iterative_sawtooth_weights(b::DiscreteHashedBelief{S}, Bidxs::AbstractVector{<:Integer}, Data::TIB_Data{S}; model=nothing, values=nothing, return_value=false) where S
    
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