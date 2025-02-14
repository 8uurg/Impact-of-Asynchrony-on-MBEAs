#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# Run ecga experiment & determine required population size.

from pathlib import Path
from tracemalloc import start
import ealib
import numpy as np
import datetime
import argparse
import subprocess
import json
import re


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
argparser.add_argument("runtime_type", type=str)
argparser.add_argument("algorithm_type", type=str)
argparser.add_argument("tournament_size", type=int, default=2)
argparser.add_argument("vtr", type=float)

parsed = argparser.parse_args()
experiment_name = parsed.experiment_name
uid = parsed.uid
seed = parsed.seed
l = parsed.l
instance = parsed.instance_path
replacement_strategy = parsed.replacement_strategy
tournament_size = parsed.tournament_size
runtime_type = parsed.runtime_type
algorithm_type = parsed.algorithm_type
vtr = parsed.vtr
num_processors = 64
do_log = True

max_generations = 100  # note: for async this gets multiplied by population size!

output_dir = Path(f"results/{experiment_name}-{get_git_revision_short_hash()}/nkl_bisect_{uid}")
output_dir.mkdir(parents=True, exist_ok=True)

# Data dict
metadata = dict(
    problem="nk-landscape",
    seed=seed,
    l=l,
    vtr=vtr,
    instance=str(instance),
    replacement_strategy=replacement_strategy,
    tournament_size=tournament_size,
    runtime_type=runtime_type,
    algorithm_type=algorithm_type,
    num_procs=num_processors,
)

# Initial dump!
def dump_metadata():
    with (output_dir / "configuration.json").open("w") as f:
        json.dump(metadata, f)


dump_metadata()

ga_approach_pattern = re.compile("ga-([a-z]+)-(a?sync)")

def run(population_size: int):
    global parsed
    global output_dir
    global max_generations
    global algorithm_type

    initializer = ealib.CategoricalUniformInitializer()
    criterion = ealib.SingleObjectiveAcceptanceCriterion()
    base_archive = ealib.BruteforceArchive([0])

    problem = ealib.NKLandscape(instance)
    limiter = ealib.Limiter(problem, int(1e10))

    output_dir_run = output_dir / f"{population_size}"
    output_dir_run.mkdir(parents=True, exist_ok=True)

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

    elif "optimum" in runtime_type:
        optimum = np.ones((l,))
        with instance.with_suffix(".txt.opt").open() as f:
            cline = f.readline().split('\t')
            read_optimum = np.array([int(a) for a in cline[1].split()])
            optimum[:len(read_optimum)] = read_optimum    

        if runtime_type == "expensive-optimum":
            def get_runtime(population, individual):
                a = np.array(
                    population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
                )
                return 1.0 + (a == optimum).sum() / a.shape[0]
        elif runtime_type == "cheap-optimum":
            def get_runtime(population, individual):
                a = np.array(
                    population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
                )
                return 2.0 - (a == optimum).sum() / a.shape[0]
        elif runtime_type == "expensive-optimum-10":
            def get_runtime(population, individual):
                a = np.array(
                    population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
                )
                return 1.0 + 9 * (a == optimum).sum() / a.shape[0]
        elif runtime_type == "cheap-optimum-10":
            def get_runtime(population, individual):
                a = np.array(
                    population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
                )
                return 10.0 - 9 * (a == optimum).sum() / a.shape[0]
        elif runtime_type == "expensive-optimum-100":
            def get_runtime(population, individual):
                a = np.array(
                    population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
                )
                return 1.0 + 99 * (a == optimum).sum() / a.shape[0]
        elif runtime_type == "cheap-optimum-100":
            def get_runtime(population, individual):
                a = np.array(
                    population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
                )
                return 100.0 - 99 * (a == optimum).sum() / a.shape[0]
        else:
            raise Exception("No runtime specifier specified")

    else:
        raise Exception("No runtime specifier specified")

    problem_simulated_runtime = ealib.SimulatedFunctionRuntimeObjectiveFunction(
        limiter, get_runtime
    )
    # evaluator_problem = problem_simulated_runtime
    vtr_monitored = ealib.ObjectiveValuesToReachDetector(problem_simulated_runtime, [[-vtr]], False)
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

    archive_logitem = ealib.ArchivedLogger()
    logger = ealib.CSVLogger(
        output_dir_run / "archive.csv",
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
        max_generations = 250

        parent_selection = ealib.ShuffledSequentialSelection()
        
        if cx_name == "uniform":
            crossover = ealib.UniformCrossover()
        elif cx_name == "twopoint":
            crossover = ealib.KPointCrossover(2)
        elif cx_name == "subfunction":
            crossover = ealib.SubfunctionCrossover()

        mutation = ealib.PerVariableBitFlipMutation(1 / l)

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
    # Note: onemax is negated (as library assumes lower = better), so flip around the vtr
    successful = np.isclose(objectives[0], -vtr)

    if do_log:
        print(f"Completed for population size {population_size}. Success: {successful}")
    return successful


start_population_size = 8
max_population_size = 2**20
highest_failing_population_size = start_population_size
population_size = start_population_size
failed = False

# Exponential probing
while not run(population_size):
    highest_failing_population_size = population_size
    population_size = population_size * 2

    if population_size > max_population_size:
        failed = True
        break

# If not failed, output upper population size
if not failed:
    metadata["population_size"] = population_size
    dump_metadata()

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
        print(f"Range: ({highest_failing_population_size}, {population_size}]")

# If not failed, output incl best population size
if not failed:
    print(f"Completed, best population size: {population_size}")
    metadata["population_size"] = population_size
    dump_metadata()
else:
    print("No run completed successfully")
