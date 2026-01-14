processes=()
processes+=("julia --project=. run_upperbound.jl --env grid --discount 0.95 --solvers TIB")
processes+=("julia --project=. run_upperbound.jl --env grid --discount 0.95 --solvers CTIB")
processes+=("julia --project=. run_upperbound.jl --env grid --discount 0.95 --solvers ETIB")
processes+=("julia --project=. run_upperbound.jl --env grid --discount 0.95 --solvers OTIB")
# processes+=("julia --project=. run_upperbound.jl --env Tag --discount 0.95 --solvers TIB")
# processes+=("julia --project=. run_upperbound.jl --env Tag --discount 0.95 --solvers CTIB")
# processes+=("julia --project=. run_upperbound.jl --env Tag --discount 0.95 --solvers ETIB")
printf "%s\n" "${processes[@]}" | parallel -j4
wait
echo -e "\n\n============= RUNS COMPLETED =============\n\n"