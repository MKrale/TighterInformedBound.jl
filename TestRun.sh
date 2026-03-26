processes=()
discount="0.95"
outfile="model_sizes_3.txt"
# for env in "ABC" "RockSample5" "Tiger" "K-out-of-N2" "HeavenOrHell" "grid" "DroneSurveilance" "iff" "Tag" "RockSample7" "SparseHallway1" "SparseHallway2" "K-out-of-N3" "SparseTigerGrid" "baseball" "pentagon" "aloha30" 
for env in "fourth"
do
    thisrun="julia --project=. RunFiles/run_upperbound.jl --env $env --discount $discount --solvers QMDP"
    processes+=("$thisrun")
done
printf "%s\n" "${processes[@]}" | parallel -j1 >> "$outfile" 2>&1
wait
echo -e "\n\n============= RUNS COMPLETED =============\n\n"