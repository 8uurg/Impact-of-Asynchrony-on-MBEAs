import os
# Limit the number of threads used by numpy & co.
os.environ['OMP_NUM_THREADS'] = "1"

import ealib
import datetime
import nasbench.nasbench_wrapper as nbw
from pathlib import Path
import subprocess
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("seed", type=int)

parsed = argparser.parse_args()

seed = parsed.seed

# Get git hash
# from https://stackoverflow.com/a/21901260/4224646
def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

output_dir_run = Path(f"./results/nasbench-reachability-serial-{get_git_revision_short_hash()}/{seed}")
output_dir_run.mkdir(parents=True, exist_ok=True)

init = ealib.CategoricalUniformInitializer()
problem, l = nbw.get_ealib_problem(simulated_runtime=False, with_performance_noise=False, with_evaluation_time_noise=False)
problem_limited = ealib.Limiter(problem, 20000, datetime.timedelta(minutes=10))
if "shuffled-fos" in algorithm_type:
    foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.Random)
    algorithm_type = algorithm_type.replace("-shuffled-fos", "")
else:
    foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.AsIs)
criterion = ealib.SingleObjectiveAcceptanceCriterion()
base_archive = ealib.BruteforceArchive([0])
archive_logitem = ealib.ArchivedLogger()
logger = ealib.CSVLogger(
    output_dir_run / "archive.csv",
    ealib.SequencedItemLogger(
        [
            ealib.NumEvaluationsLogger(problem_limited),
            ealib.WallTimeLogger(problem_limited),
            ealib.ObjectiveLogger(),
            archive_logitem,
            ealib.GenotypeCategoricalLogger(),
        ]
    ),
)
archive = ealib.LoggingArchive(base_archive, logger, archive_logitem)
stepper = ealib.InterleavedMultistartScheme((lambda p : ealib.GOMEA(p, init, foslearner, criterion, archive)), ealib.AverageFitnessComparator())

problem_monitored = ealib.ElitistMonitor(problem_limited, criterion)
f = ealib.SimpleConfigurator(problem_monitored, stepper, seed)
f.run()

pop = f.getPopulation()
elitist = archive.get_archived()[0]
# print(f"elitist: {elitist}")
gt = np.array(pop.getData(ealib.GENOTYPECATEGORICAL, elitist), copy=False)
fnts = np.array(pop.getData(ealib.OBJECTIVE, elitist), copy=False)
print((f"spent {problem_limited.get_num_evaluations()} evaluations and {problem_limited.get_time_spent_ms()}ms to\n"
       f"obtain solution {gt.view(np.byte)}\n"
       f"with fitnesses {fnts}"))
