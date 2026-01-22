module SparseModels

using POMDPs, POMDPModels, POMDPTools, SparseArrays

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


export 
SparseHallway1, SparseHallway2, SparseTigerGrid, Sparse_aloha10, Sparse_aloha30, Sparse_baseball, Sparse_cit, Sparse_fourth, Sparse_mit, Sparse_pentagon, Sparse_sunysb, Sparse_Grid, Sparse_iff

end
