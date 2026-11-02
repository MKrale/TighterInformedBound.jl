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

function get_all_iterative_closeness_weights(Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S
    return get_all_weights(Data.B, Bbao_data, Data, get_single_iterative_closeness_weights)
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

const BASIS_CACHE_TRIES = 1000  # how many cached bases are tried before calling Clp

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
    beliefs = [Data.B; Bbao_data.Bbao]
    overlaps = [Bbao_data.B_overlap; Bbao_data.Bbao_overlap]
    # Group the beliefs (B: only those also in Bbao) by support, in order of first appearance
    class_of = Dict{Vector{S},Int}()
    classes = Vector{Int}[]
    for id in eachindex(beliefs)
        id <= nb && !Bbao_data.B_in_Bbao[id] && continue
        c = get!(class_of, beliefs[id].state_list, length(classes) + 1)
        c > length(classes) && push!(classes, Int[])
        push!(classes[c], id)
    end
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
            for (k, C) in enumerate(cache)
                k > BASIS_CACHE_TRIES && break
                x = solve_with_basis!(rhs, z, L, C, b)
                if !isnothing(x)
                    k > 1 && (deleteat!(cache, k); pushfirst!(cache, C))     # most recently used first
                    break
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