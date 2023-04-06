# Run experiment & determine required population size.

# But first, limit the number of threads used by numpy & co.
# It doesn't actually seem to speed things up, but does utilize additional
# computational resources.
import os
os.environ['OMP_NUM_THREADS'] = "1"

import ealib
from pathlib import Path
import numpy as np
import datetime
import argparse
import subprocess
import json
import re
import nasbench.nasbench_wrapper as nbw


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
argparser.add_argument("seed", type=int)
argparser.add_argument("replacement_strategy", type=int)
argparser.add_argument("algorithm_type", type=str)
argparser.add_argument("tournament_size", type=int, default=2)
argparser.add_argument("vtr", type=float)

parsed = argparser.parse_args()
experiment_name = parsed.experiment_name
uid = parsed.uid
seed = parsed.seed
replacement_strategy = parsed.replacement_strategy
tournament_size = parsed.tournament_size
algorithm_type = parsed.algorithm_type
vtr = parsed.vtr
num_processors = 64
do_log = True

max_generations = 100  # note: for async this gets multiplied by population size!

output_dir = Path(f"results/{experiment_name}-{get_git_revision_short_hash()}/nasbench_bisect_{uid}")
output_dir.mkdir(parents=True, exist_ok=True)

# Data dict
metadata = dict(
    problem="nasbench301",
    seed=seed,
    vtr=vtr,
    replacement_strategy=replacement_strategy,
    tournament_size=tournament_size,
    num_processors=num_processors,
    algorithm_type=algorithm_type,
)

# Initial dump!
def dump_metadata():
    with (output_dir / "configuration.json").open("w") as f:
        json.dump(metadata, f)


dump_metadata()

print("Loading problem...", flush=True)

ga_approach_pattern = re.compile("ga-([a-z]+)-(a?sync)")
problem, l = nbw.get_ealib_problem(simulated_runtime=True, with_performance_noise=False, with_evaluation_time_noise=False)

def run(population_size: int):
    global parsed
    global output_dir
    global max_generations
    global algorithm_type

    initializer = ealib.CategoricalUniformInitializer()
    criterion = ealib.SingleObjectiveAcceptanceCriterion()
    base_archive = ealib.BruteforceArchive([0])

    limited_problem = ealib.Limiter(problem, int(1e7))

    output_dir_run = output_dir / f"{population_size}"
    output_dir_run.mkdir(parents=True, exist_ok=True)

    # evaluator_problem = problem_simulated_runtime
    vtr_monitored = ealib.ObjectiveValuesToReachDetector(limited_problem, [[-vtr]], True)
    evaluator_problem = vtr_monitored
    
    update_pop_every = 1
    update_mpm_every = 1
    if algorithm_type == "async-throttled-mpm":
        # Note, learn() is called quite a bit more often!
        # Throttled version reduces the impact of this.
        update_mpm_every = population_size
    elif algorithm_type == "async-throttled":
        update_pop_every = population_size

    issd = ealib.ECGAGreedyMarginalProduct(update_pop_every, update_mpm_every)
    selection = ealib.OrderedTournamentSelection(
        tournament_size, 1, ealib.ShuffledSequentialSelection(), criterion
    )

    # ws = ealib.WritingSimulator(output_dir_run / "events.jsonl", 1000.0)
    ws = ealib.Simulator()
    sim = ealib.SimulatorParameters(ws, num_processors, False)

    foslearner = ealib.CategoricalLinkageTree(ealib.NMI())

    archive_logitem = ealib.ArchivedLogger()
    logger = ealib.CSVLogger(
        output_dir_run / "archive.csv",
        ealib.SequencedItemLogger(
            [
                ealib.NumEvaluationsLogger(limited_problem),
                ealib.SimulationTimeLogger(ws),
                ealib.ObjectiveLogger(),
                archive_logitem,
                ealib.GenotypeCategoricalLogger(),
            ]
        ),
    )
    ita = ealib.ImprovementTrackingArchive(base_archive, lambda: limited_problem.get_num_evaluations(), population_size * 10, 2)
    base_archive = ita
    archive = ealib.LoggingArchive(base_archive, logger, archive_logitem)

    if algorithm_type == "sync":
        if replacement_strategy == 6:
            generational_selection = ealib.OrderedTournamentSelection(
                2, 1, ealib.ShuffledSequentialSelection(), criterion
            )
            approach = ealib.SynchronousSimulatedECGA(
                generational_selection,
                True,
                sim,
                population_size,
                issd,
                initializer,
                selection,
                archive,
            )
        else:
            approach = ealib.SynchronousSimulatedECGA(
                replacement_strategy,
                criterion,
                sim,
                population_size,
                issd,
                initializer,
                selection,
                archive,
            )
    elif algorithm_type == "async" or algorithm_type == "async-throttled" or algorithm_type == "async-throttled-mpm":
        if replacement_strategy == 6:
            generational_selection = ealib.OrderedTournamentSelection(
                2, 1, ealib.ShuffledSequentialSelection(), criterion
            )
            approach = ealib.AsynchronousSimulatedECGA(
                generational_selection,
                True,
                sim,
                population_size,
                issd,
                initializer,
                selection,
                archive,
            )
        else:
            approach = ealib.AsynchronousSimulatedECGA(
                replacement_strategy,
                criterion,
                sim,
                population_size,
                issd,
                initializer,
                selection,
                archive,
            )
    elif algorithm_type == "gomea-sync":
        approach = ealib.SimParallelSynchronousGOMEA(
            sim, population_size, initializer, foslearner, criterion, archive
        )
    elif algorithm_type == "kernel-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive
        )
    elif algorithm_type == "nis-kernel-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            neighborhood_learner = ealib.NISDoublingHammingKernel(),
            perform_fi_upon_no_change = False,
        )
    elif algorithm_type == "random-power-kernel-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            neighborhood_learner = ealib.RandomPowerHammingKernel(),
            perform_fi_upon_no_change = False,
        )
    elif algorithm_type == "preserving-random-power-kernel-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            neighborhood_learner = ealib.PreservingRandomPowerHammingKernel(),
            perform_fi_upon_no_change = False,
        )
    elif algorithm_type == "nis-kernel-asym-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            neighborhood_learner = ealib.NISDoublingHammingKernel(symmetric=False),
            perform_fi_upon_no_change = False,
        )
    elif algorithm_type == "random-power-asym-kernel-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            neighborhood_learner = ealib.RandomPowerHammingKernel(symmetric=False),
            perform_fi_upon_no_change = False,
        )
    elif algorithm_type == "preserving-random-power-asym-kernel-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            neighborhood_learner = ealib.PreservingRandomPowerHammingKernel(symmetric=False),
            perform_fi_upon_no_change = False,
        )
    elif algorithm_type == "gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive
        )
    elif (matchy := ga_approach_pattern.match(algorithm_type)) != None:
        # matchy = ga_approach_pattern.match(algorithm_type)
        cx_name = matchy.group(1)
        sync_or_async = matchy.group(2)

        selection_per_tournament = 1
        # Recombinative approaches may need more generations
        # to mix everything...
        max_generations = 250

        parent_selection = ealib.ShuffledSequentialSelection()
        
        if cx_name == "uniform":
            crossover = ealib.UniformCrossover()
        elif cx_name == "twopoint":
            crossover = ealib.KPointCrossover(2)
        elif cx_name == "subfunction":
            crossover = ealib.SubfunctionCrossover()

        mutation = ealib.PerVariableInAlphabetMutation(1 / l)

        ppluso = True

        if replacement_strategy == 6:
            generational_selection = ealib.OrderedTournamentSelection(tournament_size, selection_per_tournament, ealib.ShuffledSequentialSelection(), criterion)

            if sync_or_async == "sync":
                approach = ealib.SimulatedSynchronousSimpleGA(sim,
                    population_size, population_size, initializer, crossover, mutation, parent_selection, criterion, archive, ppluso, generational_selection)
            else:
                approach = ealib.SimulatedAsynchronousSimpleGA(sim,
                    population_size, population_size, initializer, crossover, mutation, parent_selection, criterion, archive, ppluso, generational_selection)

        else:
            if sync_or_async == "sync":
                approach = ealib.SimulatedSynchronousSimpleGA(sim,
                    population_size, population_size, replacement_strategy, initializer, crossover, mutation, parent_selection, criterion, archive)
            else:
                approach = ealib.SimulatedAsynchronousSimpleGA(sim,
                    population_size, population_size, replacement_strategy, initializer, crossover, mutation, parent_selection, criterion, archive)
    
    else:
        raise Exception("Unknown algorithm type")

    stepper = ealib.TerminationStepper((lambda: approach), max_generations)
    f = ealib.SimpleConfigurator(evaluator_problem, stepper, seed)
    f.run()

    pop = f.getPopulation()
    
    # print(limiter.get_time_spent_ms())
    # print(limiter.get_num_evaluations())
    elitist = archive.get_archived()[0]
    # print(f"elitist: {elitist}")
    # print(np.array(pop.getData(ealib.GENOTYPECATEGORICAL, elitist), copy=False))
    # print(np.array(pop.getData(ealib.OBJECTIVE, elitist), copy=False))
    objectives = np.array(pop.getData(ealib.OBJECTIVE, elitist), copy=False)

    # Return if run was successful
    # success meaning that the fitness was better than the value to reach.
    successful = objectives[0] < -vtr

    if do_log:
        print(f"Completed for population size {population_size}. Success: {successful}")
        print(f"Obtained objectives: {objectives}", flush=True)
    return successful


print("Starting population size search", flush=True)

start_population_size = 8
max_population_size = 2**20
highest_failing_population_size = start_population_size
population_size = start_population_size
failed = False

# Exponential part
print(f"Range: [{population_size}, inf)", flush=True)
while not run(population_size):
    highest_failing_population_size = population_size
    population_size = population_size * 2
    print(f"Range: ({highest_failing_population_size}, inf)", flush=True)

    if population_size > max_population_size:
        failed = True
        break

# If not failed, output upper population size
if not failed:
    metadata["population_size"] = population_size
    dump_metadata()

print(f"Range: [{highest_failing_population_size}, {population_size}]", flush=True)
# Binary Search
margin = 8  # stop when the difference is this small
while not failed and population_size - highest_failing_population_size >= margin:
    p = (highest_failing_population_size + population_size) // 2
    if run(p):
        population_size = p
        # Update to smallest working population size
        metadata["population_size"] = population_size
        dump_metadata()
    else:
        highest_failing_population_size = p
    if do_log:
        print(f"Range: ({highest_failing_population_size}, {population_size}]", flush=True)

# If not failed, output incl best population size
if not failed:
    print(f"Completed, best population size: {population_size}", flush=True)
    metadata["population_size"] = population_size
    dump_metadata()
else:
    print("No run completed successfully", flush=True)
