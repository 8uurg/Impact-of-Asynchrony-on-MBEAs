#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

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

experiment_name = "scalability-ecga-nkl-re"
# ls = [25, 50, 100]
seeds = range(42, 42 + num_runs)
replacement_strategies = [5, 6] # [0, 1, 2, 3, 4, 5]
runtime_types = ["cheap-optimum-10", "cheap-optimum", "constant", "expensive-optimum", "expensive-optimum-10", "expensive-optimum-100", "cheap-optimum-100"] # , "expensive-optimum-100", "cheap-optimum-100"
algorithm_types = ["async-throttled", "sync"]
tournament_sizes = [4] # [4, 8]
# limiteds = (False, True)
limiteds = [False]

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
