# Run experiment & determine required population size.

# But first, limit the number of threads used by numpy & co.
# It doesn't actually seem to speed things up, but does utilize additional
# computational resources.
import os
from typing import Optional
os.environ['OMP_NUM_THREADS'] = "1"

import ealib
from pathlib import Path
import numpy as np
import pandas as pd
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

def run(population_size: int, target_hitting_time: Optional[float] = None):
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

    if "shuffled-fos" in algorithm_type:
        foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.Random)
        algorithm_type = algorithm_type.replace("-shuffled-fos", "")
    else:
        foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.AsIs)

    archive_file = output_dir_run / "archive.csv"
    archive_logitem = ealib.ArchivedLogger()
    logger = ealib.CSVLogger(
        archive_file,
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
    ita = ealib.ImprovementTrackingArchive(base_archive, lambda: limited_problem.get_num_evaluations(), population_size * 20, 2)
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
    elif algorithm_type == "gomea-immidiate-sync":
        # Note: update_solution_at_end_of_gom = False means do not defer change until end.
        approach = ealib.SimParallelSynchronousGOMEA(
            sim, population_size, initializer, foslearner, criterion, archive,
            update_solution_at_end_of_gom = False
        )
    elif algorithm_type == "kernel-gomea-async":
        num_clusters = 0
        indices = [0]
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive
        )
    elif algorithm_type == "kernel-gomea-immidiate-async":
        num_clusters = 0
        indices = [0]
        # Note: update_solution_at_end_of_gom = False means do not defer change until end.
        approach = ealib.SimParallelAsynchronousKernelGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            update_solution_at_end_of_gom = False
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
    elif algorithm_type == "gomea-immidiate-async":
        num_clusters = 0
        indices = [0]
        # Note: update_solution_at_end_of_gom = False means do not defer change until end.
        approach = ealib.SimParallelAsynchronousGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            update_solution_at_end_of_gom = False
        )
    elif algorithm_type == "gomea-steady-state-async":
        num_clusters = 0
        indices = [0]
        # Note: update_solution_at_end_of_gom = False means do not defer change until end.
        approach = ealib.SimParallelAsynchronousGOMEA(
            sim, population_size, num_clusters, indices, initializer, foslearner, criterion, archive,
            update_solution_at_end_of_gom = False,
            copy_population_for_kernels = False,
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

    stepper = ealib.TerminationStepper((lambda: approach), max_generations, verbose=True)
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

    if successful:
            global last_hitting_time
            last_hitting_time = pd.read_csv(archive_file)["simulation time (s)"].max()
            if target_hitting_time:
                improved = last_hitting_time < target_hitting_time
                if do_log:
                    print(f"Improved?: {last_hitting_time} (current) < {target_hitting_time} (best) = {improved}")
                # We are successful if we improve over the previous best.
                return improved

    return successful


def trisect():
    start_population_size = 8
    max_population_size = 2**20
    highest_failing_population_size = start_population_size
    population_size = start_population_size
    failed = False

    # Exponential probing until we hit vtr
    while not run(population_size):
        highest_failing_population_size = population_size
        population_size = population_size * 2

        if population_size > max_population_size:
            failed = True
            break

    # Exit if we hit a maximum population size.
    if failed:
        return
    print("Completed finding valid point.")
    # Just in case! We can find a solution within this time!
    metadata["population_size"] = population_size
    dump_metadata()

    # we now know that within (highest_failing_population_size, inf) there is a point with minimum evaluation time
    # and population_size has a particular evaluation time corresponding to it.
    # First, before we can apply Tenary search, we need to find a finite upper bound.
    # (sidenote: if resources are not limited, this could explode off into very high numbers!)
    best_hitting_time = last_hitting_time

    # If the first step fails - we have our upper bound (population_size * 2).
    # # and the lower bound should be one greater that the highest failing population size.
    population_size_lb = highest_failing_population_size + 1
    lb_time = np.inf
    current_time = last_hitting_time
    # as long as this returns true we keep finding improvements.
    # once this returns false we have found our upper bound
    while run(population_size * 2, best_hitting_time):
        best_hitting_time = last_hitting_time
        # note that if the function is convex, everything with a lower population should be worse
        # lb is hence the current population size.
        population_size_lb = population_size
        lb_time = current_time
        population_size = population_size * 2
        current_time = last_hitting_time
    population_size_ub = population_size * 2
    ub_time = last_hitting_time
    print(f"Range: ({population_size_lb}, {population_size_ub}) Best: {population_size} ({current_time})")

    print("Completed finite range finding.")
    # Tenary Search
    # at this point we have obtained an upper and lower bound - as well as the time for a point in between.
    # we have enough to perform a tenary search. but standard tenary search does not use the in-between point
    # and samples 2 points per iteration, rather than 1.
    # this is a modified variant that tries to remedy this particularity.

    # in part, we sample within the area which is the largest (or if equal, flip!)
    side = population_size_ub - population_size > population_size - population_size_lb

    margin = 3  # stop when the difference between bounds is this small
    while population_size_ub - population_size_lb >= margin:
        delta = (population_size_ub - population_size_lb) // 3
        p = population_size_ub - delta if side else population_size_lb + delta
        if run(p, current_time):
            # store for update
            op = population_size
            # if this solution improves other the other, the best population size has changed.
            population_size = p
            # Update to smallest working population size
            metadata["population_size"] = population_size
            dump_metadata()
            # also, the current best time has changed
            current_time = last_hitting_time
            # since our new point is better - either the lower bound (side = True) or upper bound
            # (side = False) should change to become the original population size.
            if side:
                # new point is on the right and better.
                # original solution should become lower bound
                population_size_lb = op
            else:
                # new point is on the left and better
                # original solution should become upper bound
                population_size_ub = op
        else:
            # No improvement to be seen here!
            # since our new point is better - either the lower bound (side = True) or upper bound
            # (side = False) should change to become the original population size.
            if side:
                # new point is on the right and worse.
                # new solution should become upper bound
                population_size_ub = p
            else:
                # new point is on the left and worse.
                # new solution should become lower bound
                population_size_lb = p
        
        if population_size_ub - population_size == population_size - population_size_lb:
            # if equal size, flip (otherwise we may continue sampling...)
            side = not side
        else:
            # otherwise, pick the largest side such that we can decrease the range the most.
            # (also, since I am assuming the point to be in the right position...)
            side = population_size_ub - population_size > population_size - population_size_lb
        if do_log:
            print(f"Range: ({population_size_lb}, {population_size_ub}) Best: {population_size} ({current_time})")

    print("Completed tenary search.")
    # If not failed, output incl best population size
    if not failed:
        print(f"Completed, best population size: {population_size}")
        metadata["population_size"] = population_size
        dump_metadata()
    else:
        print("No run completed successfully")

trisect()
