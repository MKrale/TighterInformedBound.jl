#!/bin/bash

# Quick smoke test: runs every solver of RunAll.sh (and the model size script) on three small environments with short
# timeouts, writing all output to a temporary folder. Use this to check the setup before starting RunAll.sh.

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

outdir="$(mktemp -d)/"
echo "Writing test output to $outdir"
failed=()

check() {
  echo ">> $*"
  "$@" || failed+=("$*")
}

ub_solvers=("QMDP" "FIB" "TIB" "ETIB" "CTIB" "OTIB_pre" "OTIB" "OATIB_pre" "OATIB" "MultiTIB_pre" "MultiTIB" "SARSOP")
sarsop_solvers=("standard" "TIB" "ETIB" "CTIB" "OTIB_pre" "OTIB" "OATIB_pre" "OATIB" "MultiTIB_pre" "MultiTIB")

for env in "ABC" "Tiger" "HeavenOrHell"; do
  for solver in "${ub_solvers[@]}"; do
    check julia --project=. RunFiles/run_upperbound.jl --env "$env" --solvers "$solver" --timeout 60 --path "$outdir"
  done
  for solver in "${sarsop_solvers[@]}"; do
    check julia --project=. RunFiles/run_sarsop.jl --env "$env" --solvers "$solver" --timeout 60 --onlyBs true --path "$outdir"
  done
  check julia --project=. RunFiles/run_modelsizes.jl --env "$env" --path "$outdir"
done

echo -e "\nOutput files: $(ls "$outdir" | wc -l) in $outdir"
if [ ${#failed[@]} -eq 0 ]; then
  echo "All runs succeeded."
else
  echo "FAILED runs:"; printf '  %s\n' "${failed[@]}"; exit 1
fi
