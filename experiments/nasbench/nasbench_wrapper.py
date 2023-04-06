# This file contains the prerequisite utilties to use the NasBench101 search space
# within code used above.

from pathlib import Path
import numpy as np

#
import ealib

# NasBench
import os
import ConfigSpace as cs_
from ConfigSpace.read_and_write import json as cs_json
import ConfigSpace.configuration_space as cs_cs
from ConfigSpace.util import deactivate_inactive_hyperparameters
import nasbench301 as nb


folder_path = Path(__file__).parent


class DiscreteGenotypeConfigSpace:
    """
    Utility for transforming a ConfigurationSpace into a discrete genotype,
    inferring the alphabet size & characterization of this genotype as well as
    be the utility that converts the resulting genotype into configurations.
    """

    def __init__(self, cs: cs_.ConfigurationSpace):
        self.cs = cs
        self.alphabet_size = [hp.get_size() for hp in cs.get_hyperparameters()]
        self.l = len(self.alphabet_size)

    def convert(self, v) -> cs_cs.Configuration:
        # convert to numpy array
        v = np.array(v)
        if v.dtype.itemsize == 1:
            # no copy has taken place, lucky for us :)
            # Convert to int and then to float - not allowed to use chars here.
            v = v.view(np.byte)
        elif v.dtype.itemsize == 4:
            v = v.view(np.uint32)
        elif v.dtype.itemsize == 8:
            v = v.view(np.uint64)
        else:
            assert False, "Unexpectedly received an item size other that 1 or 4."
        v = v.astype(float)
        # While the active/inactive map could be a useful information source,
        # with an approach that does not account for this, searching with the requirement that
        # inactive parameters are set as such makes the search space more complicated - with many
        # solutions/offspring violating the constraint.
        # Deactivate them here instead, so that the check does not fail :)
        decoded_config = deactivate_inactive_hyperparameters(None, configuration_space=self.cs, vector=v)
        return decoded_config

def get_nasbench_301_configspace() -> cs_.ConfigurationSpace:
    return cs_json.read(open(folder_path / "configspace_simplified.json").read())

def get_ealib_problem(simulated_runtime=True, with_performance_noise=False, with_evaluation_time_noise=False):
    # Benchmark configuration - following nasbench301/example.py
    version = '0.9'
    # Notes:
    # - with_performance_noise is currently defaulting to False, in part
    #   because none of the approaches have been implemented & tested with
    #   noise (though: it should probably work?)
    #   In any case: it is an additional complication out of scope for this
    #   particular work, that may be worth revisiting at some point.
    #   (as noise could make particular effects more likely to happen, maybe?)
    # - with_evaluation_time_noise doesn't seem to do anything.

    #
    performance_model_type = 'xgb'
    runtime_model_type = 'lgb_runtime'

    # 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

    # First, download model files
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=version, delete_zip=True,
                        download_dir=current_dir)

    config_space = get_nasbench_301_configspace()

    dgcs = DiscreteGenotypeConfigSpace(config_space)

    l = dgcs.l

    # Load performance model
    ensemble_dir_performance = model_paths[performance_model_type]
    performance_model = nb.load_ensemble(ensemble_dir_performance)

    def evaluate_obj(genotype):
        # First convert the genotype to the ConfigSpace representation NasBench301 uses.
        config_i = dgcs.convert(genotype)
        # Note: Performance model predicts validation accuracy, which should be maximized.
        objective = performance_model.predict(config=config_i, representation="configspace", with_noise=with_performance_noise)
        # As a convention the library always minimizes, so flip the sign.
        return -objective

    # Convert into a ealib problem
    problem = ealib.DiscreteObjectiveFunction(
        evaluate_obj,
        dgcs.l,
        [chr(s) for s in dgcs.alphabet_size],
    )

    if simulated_runtime:
        # Load runtime model
        ensemble_dir_runtime = model_paths[runtime_model_type]
        runtime_model = nb.load_ensemble(ensemble_dir_runtime)
        def get_runtime(population, individual):
            # again, convert genotype to config.
            # we could share this computation, but that makes implementation more difficult as
            # we have to attach this data to the solution & would be dependent on ordering.
            genotype = np.array(
                population.getData(ealib.GENOTYPECATEGORICAL, individual), copy=False
            )
            config_i = dgcs.convert(genotype)
            # 
            predicted_runtime = runtime_model.predict(config_i, representation="configspace", with_noise=with_evaluation_time_noise)
            # guard against <= 0 runtimes, probably not going to happen, but just in case
            determined_runtime = max(1e-5, predicted_runtime)
            return determined_runtime

        problem = ealib.SimulatedFunctionRuntimeObjectiveFunction(
            problem, get_runtime
        )

    return problem, l

def get_ealib_similar_problem(simulated_runtime=True, with_performance_noise=False, with_evaluation_time_noise=False):
    # Benchmark configuration - following nasbench301/example.py
    version = '0.9'
    # Notes:
    # - with_performance_noise is currently defaulting to False, in part
    #   because none of the approaches have been implemented & tested with
    #   noise (though: it should probably work?)
    #   In any case: it is an additional complication out of scope for this
    #   particular work, that may be worth revisiting at some point.
    #   (as noise could make particular effects more likely to happen, maybe?)
    # - with_evaluation_time_noise doesn't seem to do anything.

    #
    performance_model_type = 'xgb'
    runtime_model_type = 'lgb_runtime'

    # 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

    # First, download model files
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=version, delete_zip=True,
                        download_dir=current_dir)

    config_space = get_nasbench_301_configspace()

    dgcs = DiscreteGenotypeConfigSpace(config_space)

    l = dgcs.l

    # Load performance model
    ensemble_dir_performance = model_paths[performance_model_type]
    performance_model = nb.load_ensemble(ensemble_dir_performance)

    def evaluate_obj(genotype):
        # Do not actually evaluate, just a sum!
        objective = np.array(genotype).view(np.byte).astype(int).sum()
        return -objective

    # Convert into a ealib problem
    problem = ealib.DiscreteObjectiveFunction(
        evaluate_obj,
        dgcs.l,
        [chr(s) for s in dgcs.alphabet_size],
    )

    if simulated_runtime:
        # Load runtime model
        ensemble_dir_runtime = model_paths[runtime_model_type]
        runtime_model = nb.load_ensemble(ensemble_dir_runtime)
        def get_runtime(population, individual):
            # Do not actually predict a runtime. Just give a fixed value.
            determined_runtime = 1.1
            return determined_runtime

        problem = ealib.SimulatedFunctionRuntimeObjectiveFunction(
            problem, get_runtime
        )

    return problem, l