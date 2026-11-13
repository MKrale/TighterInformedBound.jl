#########################################
#          Backup Tables
#########################################

"""Compressed weighted columns: for each target t, the columns of Qt and their weights that together represent t."""
struct WeightedColumns
    start::Vector{Int}      # t -> first entry of t in idx and val
    idx::Vector{Int32}      # entry -> index of a column of Qt
    val::Vector{Float64}    # entry -> its weight
end

"""
Compressed weighted columns with several sets of weights per target: the sets of target t are start[t]:start[t+1]-1, and the
entries of set k are set_start[k]:set_start[k+1]-1. The number of sets may differ per target (MultiTIB).
"""
struct MultiWeightedColumns
    start::Vector{Int}      # t -> first set of t
    set_start::Vector{Int}  # set -> first entry of the set in idx and val
    idx::Vector{Int32}      # entry -> index of a column of Qt
    val::Vector{Float64}    # entry -> its weight
end

"""The probability of every group in a table whose probabilities are already part of the column weights (TIB)."""
struct UnitProbs end
Base.getindex(::UnitProbs, ::Int) = 1.0

"""
Precomputed backups, shared by FIB, TIB and the TIB variants with precomputed weights. Each pair (i,a) of a state or belief i
and an action has a list of groups (one for each possible observation). The value of a group is its probability times the
maximum over actions of the weighted sum of the columns of its target. See fib_backup_table, tib_backup_table and weighted_backup_table.
"""
struct BackupTable{P,T,W}
    start::Vector{Int}      # (i-1)*na + ai -> first group of (i,a)
    prob::P                 # group -> probability: a Vector{Float64}, or UnitProbs for TIB
    target::T               # group -> target in columns: a Vector{Int}, or Base.OneTo for TIB
    columns::W              # WeightedColumns, or MultiWeightedColumns for MultiTIB
end

"""Returns Qo .+ w .* Qt[:,bp] (a separate function, so that Qo is not captured by a closure in the loop of target_value)"""
@inline add_weighted_column(Qo::NTuple{NA,Float64}, w::Float64, Qt::Matrix{Float64}, bp::Int32) where NA =
    ntuple(a -> @inbounds(Qo[a] + w * Qt[a,bp]), Val(NA))

"""The maximum over actions of the weighted sum of the columns of target t. NA is a type parameter, so that the loops over actions are unrolled."""
@inline function target_value(W::WeightedColumns, t::Int, Qt::Matrix{Float64}, ::Val{NA}) where NA
    Qo = ntuple(_ -> 0.0, Val(NA))
    @inbounds for e in W.start[t]:W.start[t+1]-1
        Qo = add_weighted_column(Qo, W.val[e], Qt, W.idx[e])
    end
    return maximum(Qo)
end

"""The minimum over the sets of weights of target t of the maximum over actions of their weighted sum (MultiTIB)."""
@inline function target_value(W::MultiWeightedColumns, t::Int, Qt::Matrix{Float64}, ::Val{NA}) where NA
    value = Inf
    @inbounds for k in W.start[t]:W.start[t+1]-1
        Qo = ntuple(_ -> 0.0, Val(NA))
        for e in W.set_start[k]:W.set_start[k+1]-1
            Qo = add_weighted_column(Qo, W.val[e], Qt, W.idx[e])
        end
        value = min(value, maximum(Qo))
    end
    return value
end

"""Returns r plus the discounted value of all groups of k = (i-1)*na + ai, where Qt[a,c] holds the current Q-values of column c."""
@inline function backup_action(r::Float64, T::BackupTable, gamma::Float64, Qt::Matrix{Float64}, k::Int, v::Val)
    Q = r
    @inbounds for g in T.start[k]:T.start[k+1]-1
        Q += gamma * T.prob[g] * target_value(T.columns, T.target[g], Qt, v)
    end
    return Q
end
