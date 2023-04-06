import argparse
import os
from pathlib import Path

import nasbench301 as nb
import numpy as np
import pandas as pd

import nasbench.nasbench_wrapper as nbw

def existing_path(s):
    p = Path(s)
    assert p.exists(), "path should exist"
    return p

argparser = argparse.ArgumentParser()
subparsers = argparser.add_subparsers(dest="mode", required=True)
args_random = subparsers.add_parser("random")
args_random.add_argument("seed", type=int)
args_random.add_argument("count", type=int)
args_archive = subparsers.add_parser("archive")
args_archive.add_argument("path", type=existing_path)
parsed = argparser.parse_args()

folder_path = Path(__file__).parent / "nasbench"
out_path = Path(".") / "sampling-nasbench-points"


# Partially copied from nasbench wrapper.
def get_l_and_problem_fns(with_performance_noise=False, with_evaluation_time_noise=False):
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
    models_0_9_dir = os.path.join(folder_path, 'nb_models_0.9')
    model_paths_0_9 = {
        model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    models_1_0_dir = os.path.join(folder_path, 'nb_models_1.0')
    model_paths_1_0 = {
        model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
        for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
    }
    model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

    # First, download model files
    if not all(os.path.exists(model) for model in model_paths.values()):
        nb.download_models(version=version, delete_zip=True,
                        download_dir=folder_path)

    config_space = nbw.get_nasbench_301_configspace()

    dgcs = nbw.DiscreteGenotypeConfigSpace(config_space)

    l = dgcs.l

    # Load performance model
    ensemble_dir_performance = model_paths[performance_model_type]
    performance_model = nb.load_ensemble(ensemble_dir_performance)

    # Load runtime model
    ensemble_dir_runtime = model_paths[runtime_model_type]
    runtime_model = nb.load_ensemble(ensemble_dir_runtime)

    def evaluate(genotype):
        # First convert the genotype to the ConfigSpace representation NasBench301 uses.
        config_i = dgcs.convert(genotype)
        # Note: Performance model predicts validation accuracy, which should be maximized.
        predicted_objective = performance_model.predict(config=config_i, representation="configspace", with_noise=with_performance_noise)
        # Predict runtime.
        predicted_runtime = runtime_model.predict(config_i, representation="configspace", with_noise=with_evaluation_time_noise)

        return predicted_objective, predicted_runtime

    # return evaluation function, l and alphabet size.
    return evaluate, l, dgcs.alphabet_size

eval_fn, l, alphabet_size = get_l_and_problem_fns()

out_path.mkdir(parents=True, exist_ok=True)

if parsed.mode == "random":
    seed = parsed.seed
    count = parsed.count

    csv_path = out_path / f"{seed}.csv"
    source_name = f"random_seed_{seed}"
    
    rng = np.random.default_rng(seed=seed)
    def s_source():
        for i in range(count):
            s = rng.integers(0, np.array(alphabet_size))
            yield s
elif parsed.mode == "archive":
    csv_path = out_path / f"{parsed.path.name}.csv"
    source_name = f"archive_{parsed.path.name}"

    archive = pd.read_csv(parsed.path)
    archive = archive[np.isfinite(archive["population_size"])]
    print(archive)
    def s_source():
        for idx, row in archive.iterrows():
            gt_str = row["genotype (categorical)"]
            s = np.array([int(x) for x in gt_str.split()])
            yield s

with csv_path.open("w") as f:
    f.write(f"idx,source,solution,objective,evaluation_time\n")
    seq = s_source()
    for i, s in enumerate(seq):
        s_str = " ".join(str(x) for x in s)
        o, t = eval_fn(s)
        f.write(f"{i},{source_name},{s_str},{o},{t}\n")
