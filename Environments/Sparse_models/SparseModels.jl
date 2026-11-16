module SparseModels

using POMDPs, POMDPModels, POMDPTools, Random, SparseArrays

include("Hallway1.jl")
include("Hallway2.jl")
include("TigerGrid.jl")
include("aloha10.jl")
include("aloha30.jl")
include("baseball.jl")
include("cit.jl")
include("fourth.jl")
include("mit.jl")
include("pentagon.jl")
include("sunysb.jl")
include("iff.jl")
include("grid_columns.jl")

"""
A categorical distribution over the stored entries of one sparse matrix column, without copying them.
vals must be sorted, so that pdf can use binary search.
"""
struct SparseColumnDistribution{V<:AbstractVector{Int},P<:AbstractVector{Float64}}
    vals::V
    probs::P
end

function SparseColumnDistribution(M::SparseMatrixCSC{Float64,Int}, col::Int)
    r = nzrange(M, col)
    return SparseColumnDistribution(view(rowvals(M), r), view(nonzeros(M), r))
end

POMDPs.support(d::SparseColumnDistribution) = d.vals
POMDPTools.weighted_iterator(d::SparseColumnDistribution) = zip(d.vals, d.probs)
function POMDPs.pdf(d::SparseColumnDistribution, x::Integer)
    k = searchsortedfirst(d.vals, x)
    return (k <= length(d.vals) && d.vals[k] == x) ? d.probs[k] : 0.0
end
Base.rand(rng::AbstractRNG, s::Random.SamplerTrivial{<:SparseColumnDistribution}) = rand(rng, SparseCat(s[].vals, s[].probs))

"""
A SparseTabularPOMDP with fast transition and observation distributions.
POMDPTools' versions slice a row out of a column-major sparse matrix (a search through all columns) and allocate a
SparseCat. Here, the transposed matrices give the same values in the same order as a view of a single column.
"""
struct QuickSparseTabularPOMDP <: POMDP{Int,Int,Int}
    pomdp::SparseTabularPOMDP
    Tt::Vector{SparseMatrixCSC{Float64,Int}}    # Tt[a][sp, s] = T(sp | s, a)
    Ot::Vector{SparseMatrixCSC{Float64,Int}}    # Ot[a][o, sp] = O(o | a, sp)

    function QuickSparseTabularPOMDP(pomdp::SparseTabularPOMDP)
        return new(pomdp, map(T -> sparse(transpose(T)), pomdp.T), map(O -> sparse(transpose(O)), pomdp.O))
    end
end

QuickSparseTabularPOMDP(pomdp::POMDP) = QuickSparseTabularPOMDP(SparseTabularPOMDP(pomdp))
QuickSparseTabularPOMDP(T, R, O, initial_probs, terminal_states, discount) = QuickSparseTabularPOMDP(SparseTabularPOMDP(T, R, O, initial_probs, terminal_states, discount))

POMDPs.transition(m::QuickSparseTabularPOMDP, s::Int, a::Int) = SparseColumnDistribution(m.Tt[a], s)
POMDPs.observation(m::QuickSparseTabularPOMDP, a::Int, sp::Int) = SparseColumnDistribution(m.Ot[a], sp)

POMDPs.states(m::QuickSparseTabularPOMDP) = states(m.pomdp)
POMDPs.actions(m::QuickSparseTabularPOMDP) = actions(m.pomdp)
POMDPs.actions(m::QuickSparseTabularPOMDP, s::Int) = actions(m.pomdp, s)
POMDPs.observations(m::QuickSparseTabularPOMDP) = observations(m.pomdp)
POMDPs.stateindex(m::QuickSparseTabularPOMDP, s::Int) = stateindex(m.pomdp, s)
POMDPs.actionindex(m::QuickSparseTabularPOMDP, a::Int) = actionindex(m.pomdp, a)
POMDPs.obsindex(m::QuickSparseTabularPOMDP, o::Int) = obsindex(m.pomdp, o)
POMDPs.discount(m::QuickSparseTabularPOMDP) = discount(m.pomdp)
POMDPs.reward(m::QuickSparseTabularPOMDP, s::Int, a::Int) = reward(m.pomdp, s, a)
POMDPs.initialstate(m::QuickSparseTabularPOMDP) = initialstate(m.pomdp)
POMDPs.isterminal(m::QuickSparseTabularPOMDP, s::Int) = isterminal(m.pomdp, s)

export
QuickSparseTabularPOMDP,
SparseHallway1, SparseHallway2, SparseTigerGrid, Sparse_aloha10, Sparse_aloha30, Sparse_baseball, Sparse_cit, Sparse_fourth, Sparse_mit, Sparse_pentagon, Sparse_sunysb, Sparse_Grid, Sparse_iff

end
