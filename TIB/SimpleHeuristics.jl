# Code for computing QMDP and FIB policies.

"""Abstract type for policies that use Q-tables (of size |S|) to choose actions"""
abstract type QS_table_policy <: Policy end

#########################################
#               QMDP:
#########################################

"""QMDP Solver, reimplemented for efficiency"""
@kwdef struct QMDPSolver_alt <: Solver
    precision::AbstractFloat    = 1e-3      # (Approximate) precision at which the algorithm terminates
    max_time::Float64           = 600       # Timeout time for computations 
    max_iterations::Int         = 5_000     # Timeout nmbr. of iterations
end

"""QMDP Policy, reimplemented for efficiency"""
struct QMDPPlanner_alt <: QS_table_policy
    model::POMDP
    Data::Simple_Data
end

"""Returns max_{s,a} R(s,a)"""
get_max_r(m::POMDP{S}) where S = get_max_r(m,states(m), actions(m))
function get_max_r(m,S, A)
    maxr = 0.0
    for s in S
        for a in A
            maxr = max(maxr, reward(m,s,a))
        end
    end
    return maxr 
end

POMDPs.solve(sol::QMDPSolver_alt, m::POMDP{S}) where S = solve(sol,m)

"""Computes a QMDP policy using value iteration"""
function solve(sol::QMDPSolver_alt, m::POMDP{S,A,O}; C=nothing, S_dict=nothing) where S where A where O
    # Intialization
    t0 = time()
    C isa Nothing && (C = get_constants(m))
    S_dict isa Nothing && (S_dict = Dict{eltype(C.S),Int}( zip(C.S, 1:C.ns)))

    Q = zeros(Float64, C.ns,C.na)
    Qmax = zeros(Float64, C.ns)
    max_r = get_max_r(m,C.S, C.A)
    maxQ = max_r / (1-discount(m))
    Q[:,:] .= maxQ
    Qmax[:] .= maxQ
    factor = discount(m) / (1-discount(m))

    # Precompute T & R (in case they have slow implementations)
    T = Array{DiscreteHashedBelief{S}}(undef,C.na,C.ns)
    R = zeros(Float64, C.na, C.ns)
    for (si, s) in enumerate(C.S)
        for (ai, a) in enumerate(C.A)
            T[ai,si] = DiscreteHashedBelief{S}(transition(m,s,a))
            R[ai,si] = reward(m,s,a)
        end
    end
    # Lets iterate!
    i=0
    largest_change = Inf

    while (factor * largest_change > sol.precision) && (i < sol.max_iterations)
        i+=1
        largest_change = 0.0
        for (si,s) in enumerate(C.S)
            for (ai,a) in enumerate(C.A)
                Qnext = 0.0
                for (sp, psp) in weighted_iterator(T[ai,si])
                    Qnext += psp * Qmax[S_dict[sp]]
                end
                Qnext = R[ai,si] + discount(m) * Qnext 
                largest_change = max(largest_change, abs((Qnext - Q[si,ai]) / (Q[si,ai]+1e-10) ))
                Q[si,ai] = Qnext
            end
            Qmax[si] = maximum(Q[si,:])
        end
        time()-t0 > sol.max_time && break
    end
    
    Data = Simple_Data(Q,Qmax,S_dict,C)
    return QMDPPlanner_alt(m,Data)
end

#########################################
#               FIB:
#########################################

"""FIB Solver, reimplemented for efficiency"""
@kwdef struct FIBSolver_alt <: Solver
    precision::AbstractFloat    = 1e-4
    max_time::Float64           = 600
    max_iterations::Int         = 250
end

"""FIB Policy, reimplemented for efficiency"""
struct FIBPlanner_alt <: QS_table_policy
    model::POMDP 
    Data::Simple_Data
end


POMDPs.solve(sol::FIBSolver_alt, m::POMDP{S}) where S = solve(sol,m;Data=nothing)

function solve(sol::FIBSolver_alt, m::POMDP{S,A,O}; Data::Union{TIB_Data{S}, Nothing} = nothing) where S where A where O
    # Initialization & cachin
    t0 = time()
    if Data isa Nothing
        C = get_constants(m)
        S_dict = Dict{S,Int}( zip(C.S, 1:C.ns))

        SAO_probs, SAOs = get_all_obs_probs(m, C)

        B, B_idx = get_belief_set(m, SAOs, C)
        Q = solve(QMDPSolver_alt(precision=sol.precision, max_iterations=sol.max_iterations*10), m; C=C, S_dict=S_dict).Data.Q
    else
        C, S_dict, SAO_probs, SAOs = Data.constants, Data.S_dict, Data.SAO_probs, Data.SAOs
        B, B_idx, Q = Data.B, Data.B_idx,  Data.Q
        Q isa Nothing && (Q = solve(QMDPSolver_alt(precision=sol.precision, max_iterations=sol.max_iterations*10), m; C=C, S_dict=S_dict).Data.Q)
    end
    γ = discount(m)
    factor = discount(m) / (1-discount(m))

    # Lets iterate!
    largest_change = Inf
    i=0
    while true
        i+=1
        largest_change = 0
        for (si,s) in enumerate(C.S)
            for (ai,a) in enumerate(C.A)
                thisQ = reward(m,s,a)
                for oi in SAOs[si, ai]
                    bnext_idx = B_idx[si,ai,oi]
                    bnext = B[bnext_idx]
                    Qo = zeros(C.na)
                    for (s, ps) in weighted_iterator(bnext)
                        Qo = Qo .+ ( ps .* Q[S_dict[s], :])
                    end
                    thisQ += γ * SAO_probs[oi,si,ai] * maximum(Qo)
                end
                largest_change = max(largest_change, abs((thisQ - Q[si,ai]) / (Q[si,ai]+1e-5) ))
                Q[si,ai] = thisQ
            end
        end
        if (time()-t0 > sol.max_time) || (factor * largest_change < sol.precision) || (i >= sol.max_iterations)
            break
        end
    end
    Data = Simple_Data(Q,vec(maximum(Q, dims=2)),S_dict,C)
    return FIBPlanner_alt(m,Data)
end

#########################################
#            Values & actions:
#########################################

"""Computes the optimal action and the corresponding expected value for a policy"""
function action_value(π::X,b) where X<: QS_table_policy
    M = π.model
    thisQ = zeros(π.Data.constants.na)
    for ai in 1:π.Data.constants.na
        for (s, ps) in weighted_iterator(b)
            si = π.Data.S_dict[s]
            thisQ[ai] += ps * π.Data.Q[si,ai]
        end
    end 
    aimax = argmax(thisQ)
    return (π.Data.constants.A[aimax], thisQ[aimax])
end

POMDPs.action(π::X,b) where X<: QS_table_policy = first(action_value(π,b))
POMDPs.value(π::X,b) where X<: QS_table_policy = last(action_value(π,b))

function get_heuristic_pointset(policy::X) where X<:QS_table_policy
    return policy.Data.V, [], []
end