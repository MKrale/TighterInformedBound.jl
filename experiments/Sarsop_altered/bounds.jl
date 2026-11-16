init_root_value(tree::SARSOPTree, b::AbstractVector) = dot(tree.Vs_upper, b)

@inline function min_ratio(v1::AbstractVector, v2::AbstractSparseVector)
    min_ratio = Inf
    I,V = v2.nzind, v2.nzval
    @inbounds for _i ∈ eachindex(I)
        i = I[_i]
        ratio = v1[i] / V[_i] # calling getindex on sparsevector -> NOT GOOD
        ratio < min_ratio && (min_ratio = ratio)
    end
    return min_ratio
end

function min_ratio(x::AbstractSparseVector, y::AbstractSparseVector)
    xnzind = SparseArrays.nonzeroinds(x)
    xnzval = nonzeros(x)
    ynzind = SparseArrays.nonzeroinds(y)
    ynzval = nonzeros(y)
    mx = length(xnzind)
    my = length(ynzind)
    return _sparse_min_ratio(mx, my, xnzind, xnzval, ynzind, ynzval)
end

@inline function _sparse_min_ratio(mx::Int, my::Int, xnzind, xnzval, ynzind, ynzval)
    ir = 0; ix = 1; iy = 1
    min_ratio = Inf
    @inbounds while ix ≤ mx && iy ≤ my
        jx = xnzind[ix]
        jy = ynzind[iy]

        if jx == jy
            v = xnzval[ix]/ynzval[iy]
            v < min_ratio && (min_ratio = v)
            ix += 1; iy += 1
        elseif jx < jy # x has nonzero value where y has zero value
            ix += 1
        else
            return zero(eltype(ynzval))
        end
    end
    return ix ≥ mx && iy ≤ my ? zero(eltype(ynzval)) : min_ratio
end

function upper_value(tree::SARSOPTree, b_idx::Int)
    
    v̂_min = Inf
    b= tree.b[b_idx]

    ### Method 1) if real, just use values of children.
    # if tree.is_real[b_idx]
    #     fill_populated!(tree, b_idx)
    #     return tree.V_upper[b_idx]
    # end

    ### Method 2) Point set approximation (standard)
    α_corner = tree.Vs_upper
    V_corner = dot(b, α_corner)
    V_upper = tree.V_upper

    for b_idx in tree.real 
        (tree.b_pruned[b_idx] || tree.is_terminal[b_idx]) && continue
        # tree.b_pruned[b_idx] && continue
        bint = tree.b[b_idx]
        vint = V_upper[b_idx]
        ϕ = min_ratio(b, bint)
        v̂ = V_corner + ϕ * (vint - dot(bint, α_corner))
        v̂ < v̂_min && (v̂_min = v̂)
    end

    # badB = spzeros(length(tree.S))
    # badB[5]=0.5; badB[9]=0.5
    # b == badB && println(b, v̂_min)
    ## 
    for (bpsi, bps) in enumerate(tree.B_heuristic)
        vint = tree.V_heuristic[bpsi]
        ϕ = min_ratio(b, bps)
        v̂ = V_corner + ϕ * (vint - dot(bps, α_corner))
        # v̂ < v̂_min && ( println("---");println(b);println(bps); println(vint); println(v̂, " ", v̂_min); v̂_min = v̂;)
        v̂ < v̂_min && (v̂_min = v̂)
    end

    # ### Method 3) TIB
    # if !(tree.pol_heuristic isa Nothing)
    #     idxs, vals = findnz(b)
    #     ss = map(si -> tree.S[si], idxs)
    #     b_sparsecat = DiscreteHashedBelief(ss, vals, UInt(0))
    #     v̂_min = min(v̂_min, value(tree.pol_heuristic, b_sparsecat))
    # end
    ##

    return v̂_min
end

function lower_value(tree::SARSOPTree, b::AbstractVector)
    MAX_VAL = -Inf
    for α in tree.Γ
        new_val = dot(α, b)
        if new_val > MAX_VAL
            MAX_VAL = new_val
        end
    end
    return MAX_VAL
end

root_diff(tree) = tree.V_upper[1] - tree.V_lower[1]

function root_diff_normalized(tree)
    if tree.V_lower[1] != 0
        return abs( root_diff(tree) / tree.V_lower[1] )
    end
    return abs( root_diff(tree) / (tree.V_lower[1]+1e-10) )
end

