# Code for the (E/O)TIB solvers and policies.

#########################################
#               Solver:
#########################################

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
    recompute_iterations    = Inf
    max_iterations::Int64   = 250
    max_time::Float64       = 3600
    precision::Float64      = 1e-4
    precomp_solver          = ETIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=STIBSolver(precision=1e-4, max_iterations=250, max_time=3600, precomp_solver=FIBSolver_alt(precision=1e-4, max_iterations=1000, max_time=3600)))
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

SOLVERS_REQUIRING_BBAO = [ETIBSolver, CTIBSolver, ICTIBSolver, OTIBSolver]

verbose = false
POMDPs.solve(solver::X, model::POMDP) where X<: TIBSolver = solve(solver, model; Data=nothing)
"""Computes policy for TIB-style policies"""
function solve(solver::X, model::POMDP; Data::Union{TIB_Data,Nothing}=nothing) where X<:TIBSolver
    t0 = time()

    # 1 : Cash all relevant model data
    if isnothing(Data) # This is the default case (only skipped when initializing with another TIB-style policy)
        Data::TIB_Data = get_TIB_Data(model)
    end

    # 2 : Compute Q-values bsoa beliefs
    Qs = precompute_Qs(model, Data, solver.precomp_solver)    # ∀ b ∈ B, contains QTIB value (initialized using QMDP)

    # 3 : If OTIB or ETIB, precompute all beliefs after 2 steps
    # Bbao_data::Union{BBAO_Data, Nothing} = nothing
    if any(map(solvertype -> solver isa solvertype, SOLVERS_REQUIRING_BBAO))
        Bbao_data = get_Bbao(model, Data, Data.constants)
    end
    t_init = time() - t0
    verbose && printdb("general init time:", t_init)

    # 4 : If using ETIB, pre-compute entropy weights
    
    # Weights::Union{Weights_Data, Nothing} = nothing
    if solver isa ETIBSolver
        entropies = map(b -> get_entropy(b), Data.B)
        Weights = get_all_entropy_weights(Data, Bbao_data; entropies=entropies)
    elseif solver isa CTIBSolver
        Weights = get_all_closeness_weights(Data, Bbao_data)
    elseif solver isa ICTIBSolver
        Weights = get_all_iterative_closeness_weights(Data, Bbao_data)
    # elseif solver isa LOTIB_Solver                                          
    #     Weights = get_optimal_weights_all(Data.B, Bbao_data, Data)
    end

    t_w = time() - t0 - t_init
    verbose && printdb("weights calculation time:", t_w)

    # Lets be overly fancy! Define a function for computing Q, depending on the specific solver
    get_Q, args = identity, []
    if solver isa STIBSolver
        pol, get_Q, args = STIBPolicy, get_QTIB_Beliefset, (Data,)
    elseif solver isa OTIBSolver
        pol, get_Q, args = OTIBPolicy, get_QOTIB_Beliefset, (Data, Bbao_data)
    elseif solver isa ETIBSolver
        pol, get_Q, args = ETIBPolicy, get_QETIB_Beliefset, (Data, Bbao_data, Weights)
    elseif solver isa CTIBSolver
        pol, get_Q, args = CTIBPolicy, get_QETIB_Beliefset, (Data, Bbao_data, Weights)
    elseif solver isa ICTIBSolver
        pol, get_Q, args = ICTIBPolicy, get_QETIB_Beliefset, (Data, Bbao_data, Weights)
    else
        throw("Solver type not recognized!")
    end
    
    # 5 : Now iterate:
    it = 0
    factor = discount(model) / (1-discount(model))
    for i=1:solver.max_iterations
        time_left = solver.max_time-(time()-t0)
        verbose && printdb("starting iteration $it")
        Qs, max_dif = get_Q(model, Qs,time_left, args...)
        if factor * max_dif < solver.precision || time()-t0 > solver.max_time
            break
        end
        it = i
    end
    t_it = time()- t0 - t_init - t_w
    verbose && printdb("iteration time $t_it (avg over $it iterations: $(t_it/it))")

    return pol(model, TIB_Data(Qs,Data))
end

#########################################
#            Value Computations:
#########################################

############ Q-value Precomputations ###################

"""Initializes Q-values for all belies in Data.B using solver."""
function precompute_Qs(model::POMDP, Data::TIB_Data, solver::X ; getdata=false) where X<:Union{QMDPSolver_alt, FIBSolver_alt}
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
    getdata ? (return TIB_Data(Qs, Data)) : (return Qs)
end

"""Initializes Q-values for all belies in Data.B using solver."""
function precompute_Qs(model::POMDP, Data::TIB_Data, solver::X ; getdata=false) where X<: TIBSolver
	π = solve(solver, model; Data=Data)
    B, constants = Data.B, Data.constants
    Qs = zeros(Float64, length(B), constants.na)
    for (b_idx, b) in enumerate(B)
        for (ai, a) in enumerate(constants.A)
            Qs[b_idx, ai] = π.Data.Q[b_idx,ai]
        end
    end
    getdata ? (return TIB_Data(Qs, Data)) : (return Qs)
end

############ TIB ###################

"""Performs one iteration of TIB"""
function get_QTIB_Beliefset(model::POMDP, Q, timeleft, Data::TIB_Data)
    t0 = time()
    Qs_new = zero(Q) # TODO: this may be inefficient?
    for (b_idx,b) in enumerate(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[b_idx,ai] = get_QTIB_ba(model,b,a, Q, Data; bi=b_idx, ai=ai)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

"""Performs one iteration of TIB for the given belief-action pair"""
function get_QTIB_ba(model::POMDP,b,a,Qs,B_idx,Br,SAO_probs, SAOs, constants::C; ai=nothing, bi=nothing, S_dict=nothing)
    isnothing(ai) && ( ai=findfirst(==(a), constants.A) )
    isnothing(bi) ? (Q = breward(model,b,a)) : (Q = Br[bi,ai])
    for oi in get_possible_obs(b,ai,SAOs, S_dict)
        o = constants.O[oi]
        Qo = zeros(Float64,constants.na)
        for (s, ps) in weighted_iterator(b)
            si = S_dict[s]
            if oi in SAOs[si,ai]
                p = ps * SAO_probs[oi,si,ai]
                bp_idx = B_idx[si,ai,oi]
                @views Qo .+= p .* Qs[bp_idx,:]
            end
        end
        Q += discount(model) * maximum(Qo)
    end
    return Q
end

# Some different function definitions for convenience:
get_QTIB_ba(model::POMDP,b,a,Q,D::TIB_Data; bi=nothing, ai=nothing) = get_QTIB_ba(model,b,a, Q, D.B_idx, D.Br, D.SAO_probs, D.SAOs, D.constants; bi=bi, ai=ai, S_dict=D.S_dict)
get_QTIB_ba(model::POMDP,b,a,D::TIB_Data; bi=nothing, ai=nothing) = get_QTIB_ba(model,b,a, D.Q, D.B_idx, D.Br, D.SAO_probs, D.SAOs, D.constants; bi=bi, ai=ai, S_dict=D.S_dict)

########## OTIB ###########

"""Performs one iteration of OTIB"""
function get_QOTIB_Beliefset(model::POMDP, Q, timeleft, Data::TIB_Data, Bbao_data::BBAO_Data)
    t0 = time()
    Qs_new = zero(Q)
    for (bi,b) in enumerate(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[bi,ai] = get_QOTIB_ba(model,b,a, Q, Data; ai=ai, Bbao_data=Bbao_data, bi=bi)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

get_QOTIB_ba(model::POMDP, b,a,Q, D::TIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QOTIB_ba(model, b, a, Q, D.B, D.Br, D.SAOs, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)
get_QOTIB_ba(model::POMDP, b,a,D::TIB_Data; ai=nothing, Bbao_data=nothing, bi=nothing) = get_QOTIB_ba(model, b, a, D.Q, D.B, D.Br, D.SAOs, D.SAO_probs, D.constants; ai=ai, S_dict=D.S_dict, Bbao_data=Bbao_data, bi=bi)

"""Performs one iteration of TIB for the given belief-action pair"""
function get_QOTIB_ba(model::POMDP,b,a,Qs,B,Br, SAOs, SAO_probs, constants::C; ai=nothing, Bbao_data=nothing, bi=nothing, S_dict=nothing)
    
    opt_model = Model(Clp.Optimizer; add_bridges=false)
    # opt_model = direct_generic_model(Float64, HiGHS.Optimizer())
    set_silent(opt_model)
    set_string_names_on_creation(opt_model, false)
    
    isnothing(bi) ? (Q = breward(model,b,a)) : (Q = Br[bi,ai])

    for oi in get_possible_obs(b,ai,SAOs, S_dict)
        Qo = -Inf
        if !(Bbao_data isa Nothing) && !(bi isa Nothing)
            bao = get_bao(Bbao_data, bi, ai, oi, B)
            overlap_idxs = get_overlap(Bbao_data, bi, ai, oi)
            if length(overlap_idxs) == 1
                Qo = maximum(Qs[overlap_idxs,:])
            elseif support(bao) == 1
                Qo = Qs[overlap_idxs[1]]
            else
                thisBs, thisQs = B[overlap_idxs], Qs[overlap_idxs,:]
                empty!(opt_model)
                Qo = get_QLP(bao, thisQs, thisBs, opt_model)
            end
        else
            o = constants.O[oi]
            bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
            B_rel, Bidx_rel = get_overlapping_beliefs(bao,B)
            if length(Bidx_rel) == 1
                Qo = maximum(Qs[Bidx_rel,:])
            else
                empty!(opt_model)
                Qo = get_QLP(bao, Qs[Bidx_rel,:], B_rel, opt_model)
            end
        end
        p = 0
        for (s, ps) in weighted_iterator(b)
            p += ps * SAO_probs[oi,S_dict[s],ai]
        end
        Q += p * discount(model) * Qo
    end
    return Q
end

""" Uses a point-set B with value estimates Qs to estimate the value of a belief b."""
function get_QLP(b,Qs,B, model; return_weights = false)
    na = length(Qs[1,:])
    if model isa Nothing
        # model = Model(HiGHS.Optimizer)
        model = Model(Clp.Optimizer; add_bridges=false)
        set_silent(model)
        set_string_names_on_creation(opt_model, false)
    end

    
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
    for ai in 1:na
        @constraint(model, Qmax >= sum(Qs[:,ai] .* b_ps))
    end

    @objective(model, Min, 1.0 * Qmax)
    optimize!(model)

    return_weights && (return JuMP.value.(b_ps), objective_value(model))
    return objective_value(model) 
end

########## ETIB & CTIB ###########

"""Performs one iteration of ETIB"""
function get_QETIB_Beliefset(model::POMDP,Q, timeleft, Data::TIB_Data, Bbao_data, Weights)
    Qs_new = zero(Q) # TODO: this may be inefficient?
    t0 = time()
    for (b_idx,b) in enumerate(Data.B)
        timeleft+t0-time() < 0 && (return Q, 0)
        for (ai, a) in enumerate(Data.constants.A)
            Qs_new[b_idx,ai] = get_QETIB_ba(model, b_idx, ai, Q, Data,  Bbao_data, Weights)
        end
    end
    max_dif = maximum(map(abs, (Qs_new .- Q) ./ (Q.+1e-10)))
    return Qs_new, max_dif
end

"""Performs one iteration of ETIB for the given belief-action pair, using pre-computed weights"""
function get_QETIB_ba(model::POMDP,bi, ai, Qs, Data::TIB_Data, Bbao_data::BBAO_Data, Weights_data::Weights_Data)
    # SAOs, constants, S_dict = Data.SAOs, Data.constants, Data.S_dict
    Q = Data.Br[bi,ai]
    Os = get_possible_obs( (true,bi) ,ai,Data,Bbao_data)
    for oi in Os
        o = Data.constants.O[oi]
        bao = get_bao(Bbao_data, bi, ai, oi, Data.B)
        w = discount(model) * Bbao_data.BAO_probs[oi,bi,ai]
        idxs, weights = get_weights(Bbao_data, Weights_data, bi, ai, oi)
        Q += w * maximum( sum( weights .* Qs[idxs,:], dims=1) )
    end
    return Q
end

"""Performs one iteration of ETIB for the given belief-action pair, computing weights for b on the spot"""
function get_QETIB_ba(model::POMDP, b, a, Data::TIB_Data; weight_function = get_single_entropy_weights)
    ai = actionindex(model, a)
    Q = breward(model,b,a)
    for (oi, po) in get_possible_obs_probs(b,ai,Data.SAOs,Data.SAO_probs,Data.S_dict)
        o = Data.constants.O[oi]
        bao = update(DiscreteHashedBeliefUpdater(model),b,a,o)
        _relevant_Bs, relevant_Bis = get_overlapping_beliefs(bao, Data.B)
        idxs, weights = weight_function(bao, relevant_Bis, Data)
        Qo = maximum( sum(weights .* Data.Q[idxs,:], dims=1) )
        Q += po * discount(model) * Qo
    end
    return Q
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

### Actions & value functions ###

# Since finding the best action also finds it's value, we define one function that does both, and define:
POMDPs.action(π::X, b) where X<: TIBPolicy = first(action_value(π, b))
POMDPs.value(π::X, b) where X<: TIBPolicy = last(action_value(π,b))

"""Computes the optimal action and the corresponding expected value for a policy"""
function action_value end

function action_value(π::STIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        Qa = get_QTIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

function action_value(π::OTIBPolicy, b)
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        Qa = get_QOTIB_ba(model, b, a, π.Data; ai=ai)
        Qa > bestQ && ((bestQ, bestA) = (Qa, a))
    end
    return (bestA, bestQ)
end

# ETIB and CTIB essentially do the same, except with different ways of finding weights. 
# Thus, we generalize as follows:
action_value(π::ETIBPolicy, b) = action_value_preweights(π, b; weight_function=get_single_entropy_weights)
action_value(π::CTIBPolicy, b) = action_value_preweights(π, b; weight_function=get_single_closeness_weights)
action_value(π::ICTIBPolicy, b) = action_value_preweights(π, b; weight_function=get_single_iterative_closeness_weights)
function action_value_preweights(π::X, b; weight_function = get_single_entropy_weights) where X <: Union{ETIBPolicy, CTIBPolicy, ICTIBPolicy}
    trace = stacktrace()
    b = DiscreteHashedBelief(b)
    model = π.model
    bestQ, bestA = -Inf, nothing
    for (ai,a) in enumerate(π.Data.constants.A)
        # If we want to use this policy only, computing weights might be too expensive. 
        # Instead, we could use the TIB approach online, but use the Q-table computed using our weights
        # However, we currently just use the more expensive variant
        if true
            Qa = get_QETIB_ba(model::POMDP, b, a, π.Data; weight_function=weight_function)
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
