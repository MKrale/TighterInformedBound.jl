# Tighter Informed Bound for POMDPs

Implementation of the Tighter Informed Bound (TIB) (and variants) for POMDPs. A description of the bounds is provided in the following paper:

> **Tighter Value-Function Approximations for POMDPs** \
> Merlijn Krale, Wietze Koops, Sebastian Junges, Thiago D. Simão, Nils Jansen \
> AAMAS 2025, Detroit

## Example Usage

```julia
# Preamble
using TighterInformedBound
using POMDPs, POMDPModels

# Setup
pomdp = TigerPOMDP()
solver = STIBSolver()           # Standard TIB solver

# Obtain a policy
policy = solve(solver, pomdp)

# Obtain upper bounds for exterior points (and interior points used by solver)
Vs, _Bsao, _Vsao = get_heuristic_pointset(policy)
```

Implemented bounds include ```TIB, OTIB, ETIB, CTIB, MultiTIB```, as well as optimized version of ```FIB``` and ```QMDP```. For solving Linear Programs, we use [Clp.jl](https://github.com/jump-dev/Clp.jl), which is freely accessible and performs well on small LPs. Our solvers are compatible with any discrete POMDPs following the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework.

