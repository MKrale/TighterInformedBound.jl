# Experiments

This folder contains the code and data used for the experiments in the paper. It is a separate Julia environment that uses the package from this repository. Its contents are:

  - **RunFiles**        : Scripts for running single experiments (```run_upperbound.jl``` and ```run_sarsop.jl```).
  - **Sarsop_altered**  : A copy of the NativeSARSOP solver, with alterations as explained in the paper.
  - **Environments**    : All benchmarks in *POMDPs.jl* format that are not publicly available elsewhere, including the environments collected from pomdp.org (**Sparse_models**).
  - **Data**            : All data gathered for the experiments.
  - **RunAll.sh**       : Runs all experiments used in the paper.
  - **Plotting.ipynb**  : Notebook used for data analysis & plotting.

All commands below should be run from within this folder. To install all packages, run:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
To run all experiments from the paper at once, run:
```bash
bash ./RunAll.sh
```
To run a single experiment, run either ```RunFiles/run_upperbound.jl``` or ```RunFiles/run_sarsop.jl``` using ```--project=.```. To see the available options, use flag ```-h```. For example:
```bash
julia --project=. RunFiles/run_upperbound.jl --env RockSample5 --solvers TIB --discount 0.95 --precompile true
```

