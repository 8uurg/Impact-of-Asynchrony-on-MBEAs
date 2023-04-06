# Generate the runs for this experiment
# Despite the proc-limited label, this script runs both with and without limit.

from pathlib import Path
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--append", action='store_true')
args = parser.parse_args()

output = Path("./runs.txt")

time_limit = "1h"

num_runs = 100

current_uid = 0

experiment_name = "timing-ga-trap"
# ls = [25, 50, 100]
seeds = range(42, 42 + num_runs)
replacement_strategies = [5, 6]
runtime_types = ["cheap-ones-100", "cheap-ones-10", "cheap-ones", "constant", "expensive-ones", "expensive-ones-10", "expensive-ones-100"]
limiteds = (True,)

algorithm_types = ["ga-uniform-sync", "ga-twopoint-sync", "ga-subfunction-sync", "ga-uniform-async", "ga-twopoint-async", "ga-subfunction-async"]
tournament_sizes = [4]

ls_and_instance = [
    (25, "instances/trap__l_25__k_5.txt"),
    (50, "instances/trap__l_50__k_5.txt"),
    (100, "instances/trap__l_100__k_5.txt"),
    # (200, "instances/trap__l_200__k_5.txt"),
]

with output.open("a" if args.append else "w") as f:
    # NKL
    for (
        (l, instance),
        seed,
        replacement_strategy,
        runtime_type,
        algorithm_type,
        tournament_size,
        limited,
    ) in product(
        ls_and_instance,
        seeds,
        replacement_strategies,
        runtime_types,
        algorithm_types,
        tournament_sizes,
        limiteds,
    ):
        vtr = l

        f.write(
            (
                f"timeout {time_limit} "
                f"python3 run_approach_trap_dps_cl_bht.py "
                f"{experiment_name}{'_lim' if limited else ''} "
                f"{current_uid} "
                f"{l} "
                f"{instance} "
                f"{seed} "
                f"{replacement_strategy} "
                f"{runtime_type} "
                f"{algorithm_type} "
                f"{tournament_size} "
                f"{vtr}"
                "\n"
            )
        )
        current_uid += 1
