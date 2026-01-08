# Tighter POMDP Bounds Repository

Repository containing code, as well as gathered data, as used for the following submission

> **Tighter Value-Function Approximations for POMDPs** \
> Merlijn Krale, Wietze Koops, Sebastian Junges, Thiago D. Simão, Nils Jansen \
> AAMAS 2025, Detroit

## Contents

This repository contains the following files:

Folders:

  - **TIB**                 : Contains all code related to our novel bounds TIB, ETIB and OTIB (here denoted as TIB, ETIB and OTIB). The most important files within are:
    - **solver.jl**         : Contains the algorithms for computing the different novel bounds, implemented using the *POMDPs.jl* framework.
    - **SimpleHeuristics.jl**: Contains a custom implementation of FIB and QMDP.
    - **Caching.jl**        : Contains code for precomputing beliefs and probabilities.
  - **Sarsop_altered**      : Contains a copy of the NativeSARSOP solver, with alterations as explained in the paper.
  - **Data**                : Contains all data used for our experiments
  - **Environments**        : Contains all benchmarks in *POMDPs.jl* format that are not publically available elsewhere. It includes:
    - **Sparse_models**      : Contains all environments collected form pomdps.org

Relevant scripts:

  - **RunAll.sh**             : Automatically runs all experiments used in the paper.
  - **run\*.jl**              : Scripts for running single experiments.
  - **plotting_python.ipynb** : Notebook used for data collection & plotting.


## Getting started

After cloning this repository, run the followig commands to install all packages:
```bash
julia --project=.
]
instantiate
```

## How to run

To run all experiments from the paper at once, run the following:
```bash
bash ./Runall.sh
```
To run a single test, run either ```run_upperbound.jl``` or ```run_sarsoptest.jl``` using ```--project=.```
To see the available options, us flag ```-h```.
For example, a possible command may look as follows:
```bash
julia --project=. run_upperbound.jl --env RockSample5 --solvers TIB --discount 0.95 --precompile true
```




## TODOs
* TIB:
  * Represent this explicitely as computing weight based on dynamics -> same as ETIB then
  * Optimalisation: multiple b_{s,a,o}'s might be equal, and can thus all be represented using different weights: use the one that is best.
* CTIB optimalizations:
  * Precompute x = max_s(pdf(b,s)) forall b \in B: we can then use x / x' as an upper bound for min ratio.
  * Try sorting beliefs according to highest pdf: might lead to earlier cutoffs (this requires new belief representation: skip if you don't think it'll work)
* 3TIB and 4TIB:
  * Think about if/how to define a generic 
  