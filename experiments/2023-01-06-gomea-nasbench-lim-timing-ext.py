# Generate the runs for this experiment
# Despite the proc-limited label, this script runs both with and without limit.

from pathlib import Path
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--append", action='store_true')
args = parser.parse_args()

output = Path("./runs.txt")

# increased from 4h, 8h
time_limit = "16h"

num_runs = 10

current_uid = 0

experiment_name = "timing-gomea-nasbench-ext"
# ls = [25, 50, 100]
base_seed = 52
seeds = range(base_seed, base_seed + num_runs)
replacement_strategies = [0] # Note: unused
# limiteds = (False, True)
limiteds = (True,)
# target value, 4.7273 is the best reliably obtained error for GOMEA 
# (running for 8h wall time) required a median of 54156.5 evaluations
# slight increase to account for rounding errors in the obtained value.
# This is not an issue knowing that the next best error reported during
# the runs is 4.7280.
# It should be noted that better error values are possible, especially
# with even greater budgets, however, these values are not obtained
# reliably. Indicating their difficulty (or potential occurence due
# to the noisy nature of the originating problem)
vtr = 100.0 - 4.7274

algorithm_types = ["gomea-sync", "gomea-async"] # , "kernel-gomea-async"
tournament_sizes = [4] # Note: unused

with output.open("a" if args.append else "w") as f:
    # Nasbench
    for (
        seed,
        replacement_strategy,
        algorithm_type,
        tournament_size,
        limited,
    ) in product(
        seeds,
        replacement_strategies,
        algorithm_types,
        tournament_sizes,
        limiteds,
    ):
        f.write(
            (
                f"timeout {time_limit} "
                f"python3 run_approach_nasbench_limited_timing.py "
                f"{experiment_name}{'_lim' if limited else ''} "
                f"{current_uid} "
                f"{seed} "
                f"{replacement_strategy} "
                f"{algorithm_type} "
                f"{tournament_size} "
                f"{vtr}"
                "\n"
            )
        )
        current_uid += 1
