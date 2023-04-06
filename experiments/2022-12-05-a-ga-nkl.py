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

experiment_name = "scalability-ga-nkl-re"
# ls = [25, 50, 100]
seeds = range(42, 42 + num_runs)
replacement_strategies = [5, 6]
runtime_types = ["cheap-optimum-100", "cheap-optimum-10", "cheap-optimum", "constant", "expensive-optimum", "expensive-optimum-10", "expensive-optimum-100"]
limiteds = [False] # (False, True)

algorithm_types = ["ga-uniform-sync", "ga-twopoint-sync", "ga-subfunction-sync", "ga-uniform-async", "ga-twopoint-async", "ga-subfunction-async"]
tournament_sizes = [4]

ls_and_instance = [
    # (20, "instances/nk/instances/n5_s2/L20/1.txt"),
    (40, "instances/nk/instances/n5_s2/L40/1.txt"),
    # (80, "instances/nk/instances/n5_s2/L80/1.txt"),
    # (160, "instances/nk/instances/n5_s2/L20/1.txt"),
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
        instance_path = Path(f"../EALib/{instance}")
        vtr = 0.0
        with instance_path.with_name(f"{instance_path.stem}_vtr{instance_path.suffix}").open('r') as fx:
            vtr = fx.readlines()[0].strip()

        f.write(
            (
                f"timeout {time_limit} "
                f"python3 run_ecga_nkl_dps{'_cl' if limited else ''}.py "
                f"{experiment_name}{'_lim' if limited else ''} "
                f"{current_uid} "
                f"{l} "
                f"{instance_path} "
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
