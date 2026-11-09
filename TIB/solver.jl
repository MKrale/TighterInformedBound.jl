# Code for the (E/O)TIB solvers and policies.

#########################################
#               Solver:
#########################################

verbose = false

abstract type TIBSolver <: Solver end

@kwdef struct STIBSolver <: TIBSolver
    max_iterations::Int64   = 250       # maximum iterations taken by solver
    max_time::Float64       = 3600      # maximum time spent solving
    precision::Float64      = 1e-4      # precision at which iterations is stopped
    precomp_solver          = FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600) # Solver used for precomputing Q
end
@kwdef struct ETIBSolver <: TIBSolver
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600)) end
@kwdef struct OTIBSolver <: TIBSolver
    dynamic_recompute       = true
    dynamic_precision       = 1e-4
    max_recomputes          = 50
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
 end
 @kwdef struct SawTIBSolver <: TIBSolver
    dynamic_recompute       = true
    dynamic_precision       = 1e-4
    max_recomputes          = 50
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
 end
  @kwdef struct ISawTIBSolver <: TIBSolver
    dynamic_recompute       = true
    dynamic_precision       = 1e-4
    max_recomputes          = 50
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
 end
 @kwdef struct CTIBSolver <: TIBSolver
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
 end
@kwdef struct ICTIBSolver <: TIBSolver
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
end
@kwdef struct MultiTIBSolver <: TIBSolver
    dynamic_recompute       = true
    dynamic_precision       = 1e-4
    max_recomputes          = 50
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600))
end

struct TIB_solver_info
    time::Float64
    time_init::Float64
    time_init_Qs::Float64
    time_init_weights::Float64
    time_iterating::Float64
    nmbr_iterations::Int
    nmbr_recomputes::Int
    precesion::Float64
end

function compute_weights(solver::X, Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where X<:TIBSolver where S
    if solver isa ETIBSolver
        entropies = map(b -> get_entropy(b), Data.B)
        Weights = get_all_entropy_weights(Data, Bbao_data; entropies=entropies)
    elseif solver isa CTIBSolver
        Weights = get_all_closeness_weights(Data, Bbao_data)
    elseif solver isa ICTIBSolver
        Weights = get_all_iterative_closeness_weights(Data, Bbao_data)
    elseif solver isa OTIBSolver
        Weights = get_all_optimal_weights(Data, Bbao_data)
    elseif solver isa SawTIBSolver
        Weights = get_all_sawtooth_weights(Data, Bbao_data)
    elseif solver isa ISawTIBSolver
        Weights = get_all_iterative_sawtooth_weights(Data, Bbao_data)
    elseif solver isa MultiTIBSolver
        W_closeness = get_all_iterative_closeness_weights(Data, Bbao_data)
        W_optimal = get_all_optimal_weights(Data, Bbao_data)
        entropies = map(b -> get_entropy(b), Data.B)
        W_entropy = get_all_entropy_weights(Data, Bbao_data; entropies=entropies)
        Weights = [W_closeness, W_optimal, W_entropy]
    end
    return Weights 
end


SOLVERS_REQUIRING_BBAO = [ETIBSolver, CTIBSolver, ICTIBSolver, OTIBSolver, SawTIBSolver, ISawTIBSolver, MultiTIBSolver]
POMDPs.solve(solver::X, model::POMDP) where X<: TIBSolver = solve_info(solver, model; Data=nothing)[1]
POMDPTools.solve_info(solver::X, model::POMDP) where X<: TIBSolver = solve_info(solver, model; Data=nothing)

function solve(solver::X, model::POMDP{S,A,O}; Data::Union{TIB_Data{S},Nothing}=nothing) where X<:TIBSolver where S where A where O
    return solve_info(solver, model; Data)[1]
end

"""Computes policy for TIB-style policies"""
function solve_info(solver::X, model::POMDP{S,A,O}; Data::Union{TIB_Data{S},Nothing}=nothing) where X<:TIBSolver where S where A where O
    t0 = time()

    # 1 : Cash all relevant model data
    if isnothing(Data) # This is the default case (only skipped when initializing with another TIB-style policy)
        Data::TIB_Data = get_TIB_Data(model)
    end
    
    # 3 : If OTIB or ETIB, precompute all beliefs after 2 steps
    if any(map(solvertype -> solver isa solvertype, SOLVERS_REQUIRING_BBAO))
        Bbao_data = get_Bbao(model, Data)
    end
    time_init = time() - t0

    # 2 : Compute Q-values bsoa beliefs
    Qs = precompute_Qs(model, Data, solver.precomp_solver)    # ∀ b ∈ B, contains QTIB value (initialized using QMDP)
    Data = TIB_Data{S}(Qs, Data)
    time_Qs = time() - t0 - time_init

    verbose && printdb("general init time:", (time() - t0))

    # 4 : If using ETIB, pre-compute entropy weights
    
    # Weights::Union{Weights_Data, Nothing} = nothing
    if !(solver isa STIBSolver)
        Weights = compute_weights(solver, Data, Bbao_data)
    end

    time_weights = time() - t0 - time_init - time_Qs
    verbose && printdb("weights calculation time:", time_weights)

    # Lets be overly fancy! Define a function for computing Q, depending on the specific solver
    get_Q, args = identity, []
    if solver isa STIBSolver
        pol, get_Q, args = STIBPolicy, get_QTIB_Beliefset, (Data, tib_backup_table(Data))
    elseif solver isa OTIBSolver
        pol, get_Q, args = OTIBPolicy, Get_Q_weights_Beliefset, weighted_args(Data, Bbao_data, Weights)
    elseif solver isa SawTIBSolver
        pol, get_Q, args = SawTIBPolicy, Get_Q_weights_Beliefset, weighted_args(Data, Bbao_data, Weights)
    elseif solver isa ISawTIBSolver
        pol, get_Q, args = ISawTIBPolicy, Get_Q_weights_Beliefset, weighted_args(Data, Bbao_data, Weights)
    elseif solver isa ETIBSolver
        pol, get_Q, args = ETIBPolicy, Get_Q_weights_Beliefset, weighted_args(Data, Bbao_data, Weights)
    elseif solver isa CTIBSolver
        pol, get_Q, args = CTIBPolicy, Get_Q_weights_Beliefset, weighted_args(Data, Bbao_data, Weights)
    elseif solver isa ICTIBSolver
        pol, get_Q, args = ICTIBPolicy, Get_Q_weights_Beliefset, weighted_args(Data, Bbao_data, Weights)
    elseif solver isa MultiTIBSolver
        pol, get_Q, args = MultiTIBPolicy, Get_Q_weights_Beliefset, weighted_args(Data, Bbao_data, Weights)
    else
        throw("Solver type not recognized!")
    end
    
    # 5 : Now iterate:
    it = 0
    total_it = 0
    recomputes = 0
    factor = discount(model) / (1-discount(model))
    max_dif = Inf
    while true
        time_left = solver.max_time-(time()-t0)
        verbose && printdb("starting iteration $it (precision = $(factor * max_dif))")
        Qs, max_dif = get_Q(model, Qs,time_left, args...)
        Data = TIB_Data{S}(Qs, Data)

        it += 1

        ### Normal convergence check
        if !(isdefined(solver, :dynamic_recompute)) || !(solver.dynamic_recompute)
            if it > solver.max_iterations || factor * max_dif < solver.precision || time()-t0 > solver.max_time
                total_it = it
                break
            end

        ### Convergence & recomputation checks for OTIB
        elseif solver.dynamic_recompute
            ### Break if 1) fully converged, 2) Max iterations is reached, or 3) timeout is reached
            if ((factor * max_dif < solver.dynamic_precision && it == 1)
                || (it >  solver.max_iterations && recomputes >= solver.max_recomputes)
                || time()-t0 > solver.max_time)
                total_it += it
                break
            ### Compute new weights if 1) converged for current weights, or 2) reached max iterations for weights
            elseif ( factor * max_dif < solver.dynamic_precision 
                    || (it > solver.max_iterations && recomputes < solver.max_recomputes))             
                verbose && printdb("Recomputing weights (precision = $(factor * max_dif)")
                Weights = compute_weights(solver, Data, Bbao_data)
                args = weighted_args(Data, Bbao_data, Weights)
                max_dif = Inf
                it = 0
                recomputes += 1
                total_it += it
            end
        
        elseif mod(i, solver.recompute_iterations) == 0
            verbose && printdb("Recomputing weights")
            Weights = get_all_optimal_weights(Data, Bbao_data)
            args = weighted_args(Data, Bbao_data, Weights)
        end

    end
    time_iterations = time() - t0 - time_init - time_Qs - time_weights
    verbose && printdb("Converged with precision $(max_dif * factor)")
    verbose && printdb("iteration time $time_iterations (avg over $total_it iterations: $(time_iterations/total_it))")

    info = TIB_solver_info(
        time() - t0,
        time_init,
        time_Qs,
        time_weights,
        time_iterations,
        total_it,
        recomputes,
        max_dif * factor
    )

    return pol(model, TIB_Data{S}(Qs,Data)), info
end

#########################################
#            Value Computations:
#########################################

############ Q-value Precomputations ###################

"""Initializes Q-values for all belies in Data.B using solver."""
function precompute_Qs(model::POMDP{S}, Data::TIB_Data{S}, solver::X) where X<:Union{QMDPSolver_alt, FIBSolver_alt} where S
    π = solve(solver, model; Data=Data)
    B, constants = Data.B, Data.constants
    Qs = zeros(Float64, length(B), constants.na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            for (s, p) in weighted_iterator(b)      # DONE: do weigthed iterator loop
                si = Data.S_dict[s]
                Qs[b_idx, ai] += p * π.Data.Q[si,ai]
            end
        end
    end
    return Qs
end

"""Initializes Q-values for all belies in Data.B using solver."""
function precompute_Qs(model::POMDP{S}, Data::TIB_Data{S}, solver::X) where X<: TIBSolver where S
	π = solve(solver, model; Data=Data)
    B, constants = Data.B, Data.constants
    Qs = zeros(Float64, length(B), constants.na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            Qs[b_idx, ai] = π.Data.Q[b_idx,ai]
        end
    end
    return Qs
end

############ TIB ###################

"""
The BackupTable for TIB backups. For each belief-action pair (b,a) and each observation o in BAOs[b,a] (in that order), the columns
of its group are the next beliefs b' = B_idx[s,a,o] with weights b(s) * P(o|s,a) for all s in supp(b) (in the order of supp(b)).
Since B is fixed, this saves looking these up again in every sweep.
"""
function tib_backup_table(Data::TIB_Data)
    nb, na, no = length(Data.B), Data.constants.na, Data.constants.no
    ba_start, obs_start, next, weight = Int[1], Int[1], Int32[], Float64[]
    group_next, group_weight = [Int32[] for _ in 1:no], [Float64[] for _ in 1:no]
    for (bi, b) in enumerate(Data.B), ai in 1:na
        for (s, ps) in weighted_iterator(b)
            si = Data.S_dict[s]
            for oi in Data.SAOs[si,ai]
                push!(group_next[oi], Data.B_idx[si,ai,oi])
                push!(group_weight[oi], ps * Data.SAO_probs[oi,si,ai])
            end
        end
        for oi in Data.BAOs[bi,ai]
            isempty(group_next[oi]) && error("observation $oi is possible after (b,a) = ($bi,$ai), but not after any state in supp(b)")
            append!(next, group_next[oi]); append!(weight, group_weight[oi])
            push!(obs_start, length(next) + 1)
        end
        foreach(empty!, group_next); foreach(empty!, group_weight)
        push!(ba_start, length(obs_start))
    end
    return BackupTable(ba_start, UnitProbs(), Base.OneTo(length(obs_start)-1), WeightedColumns(obs_start, next, weight))
end

"""Performs one iteration of TIB"""
function get_QTIB_Beliefset(model::POMDP{S}, Q::Qtype, timeleft::Float64, Data::TIB_Data{S}, T::BackupTable) where S where Qtype <: AbstractArray{Float64,2}
    t0 = time()
    Qs_new = zero(Q) # TODO: this may be inefficient?
    gamma = Float64(discount(model))
    Qt = permutedims(Q)     # ai, bi -> Q: the Q-values of a belief are contiguous
    for b_idx in eachindex(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        QTIB_backup_b!(Qs_new, Val(size(Q,2)), b_idx, gamma, Qt, Data.Br, T)
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

"""TIB backups of all actions of belief b_idx. The number of actions NA is a type parameter, so that the loops over actions are unrolled."""
function QTIB_backup_b!(Qs_new::Matrix{Float64}, ::Val{NA}, b_idx::Int, gamma::Float64, Qt::Matrix{Float64}, Br::Matrix{Float64}, T::BackupTable) where NA
    @inbounds for ai in 1:NA
        k = (b_idx-1)*NA + ai
        Qs_new[b_idx,ai] = backup_action(Br[b_idx,ai], T, gamma, Qt, k, Val(NA))
    end
    return Qs_new
end

"""Performs one iteration of TIB for the given belief-action pair"""
function get_QTIB_ba(model::POMDP{S,A,O},b::DiscreteHashedBelief{S},a::A,Qs::Qtype, Data::TIB_Data{S}; ai=nothing, bi=nothing) where S where A where O where Qtype <: AbstractArray{Float64,2}
    isnothing(ai) && ( ai=findfirst(==(a), Data.constants.A) )
    if isnothing(bi)
        r = breward(model,b,a)
        possible_os = get_possible_obs(b,ai,Data)
    else
        r = Data.Br[bi,ai]
        possible_os = Data.BAOs[bi,ai]
    end
    buffers = TIB_Buffers(size(Qs,2), Data.constants.no)
    return QTIB_backup!(buffers, Float64(discount(model)), b, ai, Float64(r), possible_os, transpose(Qs), Data)
end

"""Work arrays for TIB backups, so that sweeps do not allocate"""
struct TIB_Buffers
    Qo::Matrix{Float64}     # ai, oi -> Q-value of the next beliefs given oi
    seen::Vector{Bool}      # oi -> is Qo[:,oi] initialized?
end
TIB_Buffers(na::Int, no::Int) = TIB_Buffers(zeros(na, no), zeros(Bool, no))

"""
TIB backup of (b,ai) with expected reward r, where possible_os are the observations possible after (b,ai), and Qt[a,bi] the
current Q-values. Loops over the possible observations of each state once, and sums over observations in the order of possible_os.
"""
function QTIB_backup!(buffers::TIB_Buffers, gamma::Float64, b::DiscreteHashedBelief{S}, ai::Int, r::Float64, possible_os, Qt::AbstractMatrix{Float64}, Data::TIB_Data{S}) where S
    Qo, seen = buffers.Qo, buffers.seen
    na = size(Qt, 1)
    for (s, ps) in weighted_iterator(b)
        si = Data.S_dict[s]
        for oi in Data.SAOs[si,ai]
            if !seen[oi]
                seen[oi] = true
                @inbounds for a in 1:na; Qo[a,oi] = 0.0; end
            end
            p = ps * Data.SAO_probs[oi,si,ai]
            bp_idx = Data.B_idx[si,ai,oi]
            @inbounds for a in 1:na; Qo[a,oi] += p * Qt[a,bp_idx]; end
        end
    end
    Q = r
    for oi in possible_os
        Qmax = Qo[1,oi]
        @inbounds for a in 2:na; Qmax = max(Qmax, Qo[a,oi]); end
        Q += gamma * Qmax
        seen[oi] = false
    end
    return Q
end

# Some different function definitions for convenience:
get_QTIB_ba(model::POMDP{S},b::DiscreteHashedBelief{S},a,D::TIB_Data{S}; bi=nothing, ai=nothing) where S = get_QTIB_ba(model,b,a, D.Q, D; bi=bi, ai=ai)

########## Heuristics with pre-computed weights (OTIB, ETIB, CTIB) ###########

"""
The BackupTable for ETIB, CTIB, OTIB and MultiTIB. For each belief-action pair (b,a) and each observation o (in the order of
Bbao_idx[b,a]), it stores P(o|b,a) and the next belief b': bi for beliefs in B, nb + baoi for beliefs only in Bbao. The columns are the
beliefs and weights that represent each b', with one set of columns for each set of weights (MultiTIB uses several).
It is rebuilt whenever the weights change.
"""
function weighted_backup_table(Data::TIB_Data, Bbao_data::BBAO_Data, Weights::Union{Weights_Data, Vector{Weights_Data}})
    nb, na = length(Data.B), Data.constants.na
    ba_start, prob, target = Int[1], Float64[], Int[]
    for bi in 1:nb, ai in 1:na
        for oi in get_possible_obs((true,bi), ai, Data, Bbao_data)
            in_B, baoi = Bbao_data.Bbao_idx[bi,ai][oi]
            push!(prob, Bbao_data.BAO_probs[oi,bi,ai])
            push!(target, in_B ? baoi : nb + baoi)
        end
        push!(ba_start, length(prob) + 1)
    end
    columns = Weights isa Weights_Data ? weighted_columns(Weights) : map(weighted_columns, Weights)
    return BackupTable(ba_start, prob, target, columns)
end

"""The columns representing all beliefs for one set of weights: bi for beliefs in B, nb + baoi for beliefs only in Bbao."""
function weighted_columns(W::Weights_Data)
    start, idx, val = Int[1], Int32[], Float64[]
    for (idxs, ws) in ((W.B_idxs, W.B_weights), (W.Bbao_idxs, W.Bbao_weights)), i in eachindex(idxs)
        isassigned(idxs, i) && (append!(idx, idxs[i]); append!(val, ws[i]))     # (B_idxs is only set for beliefs in Bbao)
        push!(start, length(idx) + 1)
    end
    return WeightedColumns(start, idx, val)
end

weighted_args(Data::TIB_Data, Bbao_data::BBAO_Data, Weights) = (Data, Bbao_data, weighted_backup_table(Data, Bbao_data, Weights))

"""Performs one iteration of ETIB"""
function Get_Q_weights_Beliefset(model::POMDP{S}, Q::Qtype, timeleft, Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}, W::BackupTable) where S where Qtype <: AbstractArray{Float64,2}
    Qs_new = zero(Q)
    t0 = time()
    gamma = Float64(discount(model))
    Qt = permutedims(Q)     # ai, bi -> Q: the Q-values of a belief are contiguous
    for b_idx in eachindex(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        weighted_backup_b!(Qs_new, Q, Val(size(Q,2)), b_idx, gamma, Qt, Data.Br, W)
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

"""
Backups of all actions of belief bi, using precomputed weights (for MultiTIB, the minimum over its sets of weights).
The number of actions NA is a type parameter, so that the loops over actions are unrolled.
"""
function weighted_backup_b!(Qs_new::Matrix{Float64}, Q::Matrix{Float64}, ::Val{NA}, bi::Int, gamma::Float64, Qt::Matrix{Float64}, Br::Matrix{Float64}, W::BackupTable) where NA
    @inbounds for ai in 1:NA
        k = (bi-1)*NA + ai
        Qs_new[bi,ai] = min(Q[bi,ai], backup_action(Br[bi,ai], W, gamma, Qt, k, Val(NA)))
    end
    return Qs_new
end

"""Performs one iteration of ETIB for the given belief-action pair, computing weights for b on the spot"""
function get_Q_weights_ba(model::POMDP{S,A,O}, b::DiscreteHashedBelief{S}, a, Data::TIB_Data{S}; weight_function = get_single_entropy_weights) where S where A where O
    ai = actionindex(model, a)
    Q = breward(model,b,a)
    for (oi, po) in get_possible_obs_probs(b,ai,Data)
        o = Data.constants.O[oi]
        bao = update(DiscreteHashedBeliefUpdater{S,A,O}(model),b,a,o)
        _relevant_Bs, relevant_Bis = get_overlapping_beliefs(bao, Data.B)
        idxs, weights = weight_function(bao, relevant_Bis, Data)
        Qo = maximum( sum(weights .* Data.Q[idxs,:], dims=1) )
        Q += po * discount(model) * Qo
    end
    return Q
end

########## TIB3 ###########

function get_QTIB3_Beliefset(model::POMDP{S}, Q::Qtype, timeleft::Float64, Data::TIB_Data{S}, Bbao_data::BBAO_Data{S}) where S where Qtype <: AbstractArray{Float64,2}
    t0 = time()
    Qs_new = zero(Q)
    for (b_idx,b) in enumerate(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[b_idx,ai] = get_QTIB_ba(model,b,a, Q, Data; bi=b_idx, ai=ai)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

#########################################
#      Policies, actions, values:
#########################################

### Policy definitions ###

abstract type TIBPolicy <: Policy end
POMDPs.updater(π::X) where X<:TIBPolicy = DiscreteHashedBeliefUpdater(π.model)

struct STIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end
struct OTIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end
struct SawTIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end
struct ISawTIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end
struct ETIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end
struct CTIBPolicy <: TIBPolicy
    model::POMDP 
    Data::TIB_Data
end
struct ICTIBPolicy <: TIBPolicy
    model::POMDP 
    Data::TIB_Data
end
struct MultiTIBPolicy <: TIBPolicy
    model::POMDP
    Data::TIB_Data
end

### Actions & value functions ###

# Since finding the best action also finds it's value, we define one function that does both, and define:
POMDPs.action(π::X, b) where X<: TIBPolicy = first(action_value(π, b))
POMDPs.value(π::X, b) where X<: TIBPolicy = last(action_value(π,b))

"""Computes the optimal action and the corresponding expected value for a policy"""
function action_value end

action_value(π::X, b::SparseCat{Vector{S}}) where X <:TIBPolicy where S = action_value(π, DiscreteHashedBelief{S}(b))
function action_value(π::STIBPolicy, b::DiscreteHashedBelief{S}) where S
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        Qa = get_QTIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

# ETIB and CTIB essentially do the same, except with different ways of finding weights. 
# Thus, we generalize as follows:
action_value(π::ETIBPolicy, b::DiscreteHashedBelief{S}) where S = action_value_preweights(π, b; weight_function=get_single_entropy_weights)
action_value(π::CTIBPolicy, b::DiscreteHashedBelief{S}) where S = action_value_preweights(π, b; weight_function=get_single_closeness_weights)
action_value(π::ICTIBPolicy, b::DiscreteHashedBelief{S}) where S = action_value_preweights(π, b; weight_function=get_single_iterative_closeness_weights)
action_value(π::OTIBPolicy, b::DiscreteHashedBelief{S}) where S = action_value_preweights(π, b; weight_function=get_single_optimal_weights)
action_value(π::SawTIBPolicy, b::DiscreteHashedBelief{S}) where S = action_value_preweights(π, b; weight_function=get_single_sawtooth_weights)
action_value(π::ISawTIBPolicy, b::DiscreteHashedBelief{S}) where S = action_value_preweights(π, b; weight_function=get_single_iterative_sawtooth_weights)
action_value(π::MultiTIBPolicy, b::DiscreteHashedBelief{S}) where S = action_value_preweights(π, b; weight_function=get_single_optimal_weights)
function action_value_preweights(π::X, b::DiscreteHashedBelief{S}; weight_function = get_single_entropy_weights) where X<:TIBPolicy where S
    trace = stacktrace()
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        # If we want to use this policy only, computing weights might be too expensive. 
        # Instead, we could use the TIB approach online, but use the Q-table computed using our weights
        # However, we currently just use the more expensive variant
        if true
            Qa = get_Q_weights_ba(model::POMDP, b, a, π.Data; weight_function=weight_function)
        else
            Qa = get_QTIB_ba(model, b, a, π.Data; ai=ai)
        end
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

"""
Returns:

* A vector with the values of the exerior beliefs (ordered via POMDP.stateindex);
* A vector of additional beliefs for which a value is known;
* A vector with these corresponding values.
"""
function get_heuristic_pointset(policy::X; get_only_Bs=false) where X<:TIBPolicy
    B_heuristic, V_heuristic = Vector{Float64}[], Float64[]
    ns = policy.Data.constants.ns
    S_dict = policy.Data.S_dict

    badBs = []

    for (b_idx, b) in enumerate(policy.Data.B)
        b_svector = spzeros(ns)
        for (s, p) in weighted_iterator(b)
            b_svector[stateindex(policy.model, s)] = p 
        end
        push!(B_heuristic, b_svector)
        push!(V_heuristic, maximum(policy.Data.Q[b_idx,:]))
    end
    corner_values = V_heuristic[1:ns]
    # get_only_Bs && return corner_values
    return corner_values, B_heuristic[ns:end], V_heuristic[ns:end]
end
