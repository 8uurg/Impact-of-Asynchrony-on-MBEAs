#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from pathlib import Path
import ealib
import numpy as np
import datetime
import argparse
import json
import subprocess

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

def path_that_exists(p):
    r = Path(p)
    if not r.exists():
        raise Exception("Path provided does not exist")
    return r


argparser = argparse.ArgumentParser()
argparser.add_argument("experiment_name", type=str)
argparser.add_argument("uid", type=int)
argparser.add_argument("l", type=int)
argparser.add_argument("instance_path", type=path_that_exists)
argparser.add_argument("seed", type=int)
argparser.add_argument("replacement_strategy", type=int)
argparser.add_argument("population_size", type=int)
argparser.add_argument("runtime_type", type=str)
argparser.add_argument("algorithm_type", type=str)
argparser.add_argument("tournament_size", type=int, default=2)
max_generations = 100  # note: for async this gets multiplied by population size!

parsed = argparser.parse_args()
experiment_name = parsed.experiment_name
uid = parsed.uid
l = parsed.l
instance = parsed.instance_path
seed = parsed.seed
replacement_strategy = parsed.replacement_strategy
population_size = parsed.population_size
tournament_size = parsed.tournament_size
runtime_type = parsed.runtime_type
algorithm_type = parsed.algorithm_type

vtr = -l

approach_name = "gomea" if "gomea" in algorithm_type else "ecga"

output_dir = Path(
    f"results/{experiment_name}-{get_git_revision_short_hash()}/constant_{approach_name}_single_spans_{uid}"
)
output_dir.mkdir(parents=True, exist_ok=True)

# Data dict
metadata = dict(
    problem="deceptive trap",
    seed=parsed.seed,
    l=parsed.l,
    # vtr=parsed.vtr,
    replacement_strategy=parsed.replacement_strategy,
    tournament_size=parsed.tournament_size,
    runtime_type=parsed.runtime_type,
    algorithm_type=parsed.algorithm_type,
)

# Initial dump!
def dump_metadata():
    with (output_dir / "configuration.json").open("w") as f:
        json.dump(metadata, f)


dump_metadata()

initializer = ealib.CategoricalUniformInitializer()
criterion = ealib.SingleObjectiveAcceptanceCriterion()
base_archive = ealib.BruteforceArchive([0])


problem = ealib.BestOfTraps(instance)

limiter = ealib.Limiter(problem, 100_000)
problem_monitored = ealib.ElitistMonitor(limiter, criterion)

if runtime_type == "constant":

    def get_runtime(population, individual):
        return 1.0

elif runtime_type == "expensive-ones":

    def get_runtime(population, individual):
        a = np.array(
            population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
        )
        return 1.0 + a.sum() / a.shape[0]

elif runtime_type == "expensive-ones-10":

    def get_runtime(population, individual):
        a = np.array(
            population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
        )
        return 1.0 + 9 * a.sum() / a.shape[0]

elif runtime_type == "expensive-ones-100":

    def get_runtime(population, individual):
        a = np.array(
            population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
        )
        return 1.0 + 99 * a.sum() / a.shape[0]

else:
    raise Exception("No runtime specifier specified")

problem_simulated_runtime = ealib.SimulatedFunctionRuntimeObjectiveFunction(
    limiter, get_runtime
)
evaluator_problem = problem_simulated_runtime

update_pop_every = 1
update_mpm_every = 1
if algorithm_type == "async-throttled-mpm":
    # Note, learn() is called quite a bit more often!
    # Throttled version reduces the impact of this.
    update_mpm_every = population_size
elif algorithm_type == "async-throttled":
    update_pop_every = population_size

if not "gomea" in algorithm_type:
    issd = ealib.ECGAGreedyMarginalProduct(update_pop_every, update_mpm_every)
    selection = ealib.OrderedTournamentSelection(
        tournament_size, 1, ealib.ShuffledSequentialSelection(), criterion
    )

# ws = ealib.WritingSimulator(output_dir / "events.jsonl")
ws = ealib.Simulator()
sim = ealib.SimulatorParameters(ws, population_size, False)

archive_logitem = ealib.ArchivedLogger()
logger = ealib.CSVLogger(
    output_dir / "archive.csv",
    ealib.SequencedItemLogger(
        [
            ealib.NumEvaluationsLogger(limiter),
            ealib.SimulationTimeLogger(ws),
            ealib.ObjectiveLogger(),
            archive_logitem,
            ealib.GenotypeCategoricalLogger(),
        ]
    ),
)
if "shuffled-fos" in algorithm_type:
    foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.Random)
    algorithm_type = algorithm_type.replace("-shuffled-fos", "")
else:
    foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.AsIs)
archive = ealib.LoggingArchive(base_archive, logger, archive_logitem)

spanlogger_itemlogger = ealib.SpanItemLogger()
spanlogger_timelogger = ealib.SimulationTimeLogger(ws)
spanlogger_logger = ealib.CSVLogger(
    output_dir / "spans.csv",
    ealib.SequencedItemLogger(
        [
            spanlogger_itemlogger,
            spanlogger_timelogger,
            ealib.TimeSpentItemLogger()
        ]
    ),
)
spanlogger = ealib.SpanLogger(spanlogger_logger, spanlogger_itemlogger)

evalspanlogging_problem = ealib.SimulatedEvaluationSpanLoggerWrapper(
    evaluator_problem, spanlogger, spanlogger_timelogger
)
evaluator_problem = evalspanlogging_problem

if algorithm_type == "sync":
    approach = ealib.SynchronousSimulatedECGA(
        replacement_strategy,
        criterion,
        sim,
        population_size,
        issd,
        initializer,
        selection,
        archive,
        spanlogger,
    )
elif (
    algorithm_type == "async"
    or algorithm_type == "async-throttled"
    or algorithm_type == "async-throttled-mpm"
):
    # each step acts upon a single solution
    # unlike a normal ea, which acts upon each solution.
    max_generations = max_generations * population_size
    approach = ealib.AsynchronousSimulatedECGA(
        replacement_strategy,
        criterion,
        sim,
        population_size,
        issd,
        initializer,
        selection,
        archive,
        spanlogger,
    )
elif algorithm_type == "gomea-sync":
    approach = ealib.SimParallelSynchronousGOMEA(
        sim, population_size, initializer, foslearner, criterion, archive, spanlogger
    )
elif algorithm_type == "kernel-gomea-async":
    num_clusters = 0
    indices = [0]
    approach = ealib.SimParallelAsynchronousKernelGOMEA(
        sim,
        population_size,
        num_clusters,
        indices,
        initializer,
        foslearner,
        criterion,
        archive,
        spanlogger,
    )
else:
    raise Exception("Unknown algorithm type")

stepper = ealib.TerminationStepper((lambda: approach), max_generations)
f = ealib.SimpleConfigurator(evaluator_problem, stepper, seed)
f.run()

# pop = f.getPopulation()
# print(limiter.get_time_spent_ms())
# print(limiter.get_num_evaluations())
# elitist = archive.get_archived()[0]
# print(f"elitist: {elitist}")
# print(np.array(pop.getData(ealib.GENOTYPECATEGORICAL, elitist), copy=False))
# print(np.array(pop.getData(ealib.OBJECTIVE, elitist), copy=False))
