#!/bin/bash

# File to run all experiments used in the paper. For analysing the created data, see Plotting.ipynb.
# Model sizes are collected separately, see RunModelSizes.sh.
#
# All runs are first collected in `processes`, and then executed using GNU parallel with `njobs` jobs at a time.
# For the final experiments, keep njobs=1 (sequential), so that runs do not interfere with each other's timings.
# To rerun only part of the experiments, comment out the blocks you do not need.

njobs=1
logfile="RunAll.log"
discount="0.95"

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Solvers used in the upper bound experiments (run_upperbound.jl) and SARSOP experiments (run_sarsop.jl)
ub_solvers=("QMDP" "FIB" "TIB" "ETIB" "CTIB" "OTIB_pre" "OTIB" "OATIB_pre" "OATIB" "MultiTIB_pre" "MultiTIB" "SARSOP")
sarsop_solvers=("standard" "TIB" "ETIB" "CTIB" "OTIB_pre" "OTIB" "OATIB_pre" "OATIB" "MultiTIB_pre" "MultiTIB")

processes=()

#### Normal experiments #####

## Small, UB
for env in "ABC" "RockSample5" "Tiger" "K-out-of-N2" "HeavenOrHell" "grid" "DroneSurveilance" "iff" "Tag" # QUICK, using precompile to minimize variance
do
  for solver in "${ub_solvers[@]}"
  do
    processes+=("julia --project=. RunFiles/run_upperbound.jl --env $env --discount $discount --solvers $solver --precompile true")
  done
done

## Large, UB
for env in "RockSample7" "SparseHallway1" "SparseHallway2" "K-out-of-N3" "SparseTigerGrid" "baseball" # LONG, not using precompile to save on computational cost
do
  for solver in "${ub_solvers[@]}"
  do
    processes+=("julia --project=. RunFiles/run_upperbound.jl --env $env --discount $discount --solvers $solver --precompile false")
  done
done

## Extra large, UB
for env in "pentagon" "aloha30" "fourth" # LONGER
do
  for solver in "${ub_solvers[@]}"
  do
    processes+=("julia --project=. RunFiles/run_upperbound.jl --env $env --discount $discount --solvers $solver --precompile false")
  done
done

## Small, SARSOP
for env in "ABC" "RockSample5" "Tiger" "K-out-of-N2" "HeavenOrHell" "grid" "DroneSurveilance" "iff" "Tag" # QUICK
do
  for solver in "${sarsop_solvers[@]}"
  do
    processes+=("julia --project=. RunFiles/run_sarsop.jl --env $env --discount $discount --solvers $solver --onlyBs true --precompile true")
  done
done

## Large, SARSOP
for env in "RockSample7" "SparseHallway1" "SparseHallway2" "K-out-of-N3" "SparseTigerGrid" "baseball" # LONG
do
  for solver in "${sarsop_solvers[@]}"
  do
    processes+=("julia --project=. RunFiles/run_sarsop.jl --env $env --discount $discount --solvers $solver --onlyBs true --precompile false")
  done
done

## Extra large, SARSOP
for env in "pentagon" "aloha30" "fourth" # LONGER
do
  for solver in "${sarsop_solvers[@]}"
  do
    processes+=("julia --project=. RunFiles/run_sarsop.jl --env $env --discount $discount --solvers $solver --onlyBs true --precompile false")
  done
done

## Run all collected processes ($njobs at a time)
if command -v parallel > /dev/null; then
  printf "%s\n" "${processes[@]}" | parallel -j "$njobs" 2>&1 | tee -a "$logfile"
else
  echo "GNU parallel not found: running all processes sequentially." | tee -a "$logfile"
  for process in "${processes[@]}"; do
    $process 2>&1 | tee -a "$logfile"
  done
fi
wait

##### Discount experiments (not used in the current version of the paper) #####
# Runs sequentially, since the sweep for a heuristic stops as soon as a run takes longer than 3600s.

# folder_path="Data/DiscountTest/"
# for env in "Tiger" "RockSample5" "K-out-of-N2"
# do
#   for heuristic in "standard" "TIB" "ETIB" "OTIB" "OATIB"
#   do
#     for discount in $(seq 0.95 0.001 0.999)
#     do
#       start_time=$(date +%s)
#       julia --project=. RunFiles/run_sarsop.jl --env $env --precompile true --path $folder_path --solvers $heuristic --discount $discount 2>&1 | tee -a "$logfile"
#       end_time=$(date +%s)
#       elapsed_time=$((end_time - start_time))
#       if [ $elapsed_time -gt 3600 ]; then
#         echo "Run took ${elapsed_time}s (longer than 3600s): stopping discount sweep for $heuristic on $env." | tee -a "$logfile"
#         break
#       fi
#     done
#   done
# done
# wait

echo -e "\n\n============= RUNS COMPLETED =============\n\n" | tee -a "$logfile"
