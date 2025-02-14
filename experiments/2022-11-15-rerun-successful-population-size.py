#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# We have performed quite a few runs, but we lack the evaluations from many configurations
# here we - for each approach, find a configuration that was successful and use the re-running script.
import pandas as pd

vtr = 95.2727

# Load data
data_nolim = pd.concat(
    [
        pd.read_csv(
            "./results/2022-11-07-long-scalability-ecga-nasbench-1042df6-archive.csv.gz"
        ),
        pd.read_csv(
            "./results/2022-11-09-long-scalability-ecga-nasbench-8e78f7b-archive.csv.gz"
        ),
        pd.read_csv(
            "./results/2022-11-07-long-scalability-ga-nasbench-1042df6-archive.csv.gz"
        ),
        pd.read_csv(
            "./results/2022-11-07-long-scalability-gomea-nasbench-1042df6-archive.csv.gz"
        ),
    ]
)
data_nolim["n_procs"] = data_nolim["population_size"]
data_nolim["lim"] = False
data_lim = pd.concat(
    [
        pd.read_csv(
            "./results/2022-11-14-timing-ecga-nasbench_lim-8e78f7b-archive.csv.gz"
        ),
        pd.read_csv(
            "./results/2022-11-15-timing-ecga-nasbench-6_lim-30ae8d0-archive.csv.gz"
        ),
        pd.read_csv(
            "./results/2022-11-14-timing-ga-nasbench_lim-8e78f7b-archive.csv.gz"
        ),
        pd.read_csv(
            "./results/2022-11-14-timing-gomea-nasbench_lim-8e78f7b-archive.csv.gz"
        ),
    ]
)
data_lim["n_procs"] = 64
data_lim["lim"] = True
data_both = pd.concat([data_nolim, data_lim])

configuration_columns = [
    "replacement_strategy",
    "tournament_size",
    "algorithm_type",
    "lim",
]
problem_columns = ["problem"]
run_columns = ["seed"]

# Get endpoint of each run
data_both = (
    data_both.sort_values("#evaluations")
    .groupby(configuration_columns + run_columns)
    .last()
    .reset_index()
)

# Filter configurations
hit_vtr = -data_both["objectives"] >= vtr
data_both = data_both[hit_vtr]

# Grab one successful run per configuration
data_both = data_both.groupby(configuration_columns).last().reset_index()

def get_run_cmd(
    idx: int,
    seed: int,
    replacement_strategy: int,
    algorithm_type: str,
    tournament_size: int,
    population_size: int,
    num_processors: int,
):
    return (
        "python run_approach_nasbench_specific.py "
        "nasbench-samples "
        f"{idx} "
        f"{seed} "
        f"{replacement_strategy} "
        f"{algorithm_type} "
        f"{tournament_size} "
        f"95.2726 "
        f"{int(population_size)} "
        f"{int(num_processors)}"
    )


with open("runs.txt", 'w') as f:
    for idx, (_, r) in enumerate(data_both.iterrows()):
        run_cmd = get_run_cmd(
            idx,
            r["seed"],
            r["replacement_strategy"],
            r["algorithm_type"],
            r["tournament_size"],
            r["population_size"],
            r["n_procs"],
        )
        f.write(f"timeout 16h {run_cmd}\n")
