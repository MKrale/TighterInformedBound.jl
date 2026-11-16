#!/bin/bash

# Collects the sizes of all environments (|S|, |A|, |O|, |B1|, |B2|, |B3|) into Data/ModelSizes/.
# This is kept separate from RunAll.sh, since computing |B3| is expensive for the larger environments.
# Runs are executed sequentially.

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

logfile="RunModelSizes.log"
discount="0.95"

for env in "ABC" "RockSample5" "Tiger" "K-out-of-N2" "HeavenOrHell" "grid" "DroneSurveilance" "iff" "Tag" \
           "RockSample7" "SparseHallway1" "SparseHallway2" "K-out-of-N3" "SparseTigerGrid" "baseball" \
           "pentagon" "aloha30" "fourth"
do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] model sizes: $env" | tee -a "$logfile"
  julia --project=. RunFiles/run_modelsizes.jl --env "$env" --discount "$discount" 2>&1 | tee -a "$logfile"
done

echo -e "\n\n============= RUNS COMPLETED =============\n\n" | tee -a "$logfile"
