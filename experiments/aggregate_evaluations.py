from pathlib import Path
import pandas as pd
from tqdm import tqdm

results_folder = Path("results")

def is_aggregated(p: Path):
    return (p / "evaluations.csv.gz").exists()

def aggregate_archive(p: Path):

    def merge(run: Path):
        try:
            configuration = pd.read_json(run / "configuration.json", lines=True)
            best_population_size = None
            if "population_size" in configuration.columns:
                best_population_size = configuration.iloc[0]["population_size"]
            elif "population_size_chosen" in configuration.columns:
                # fallback for fixed-size runs
                best_population_size = configuration.iloc[0]["population_size_chosen"]
            else:
                # i.e. insert null run
                return configuration
             
            archive = pd.read_csv(run / f"{best_population_size}" / "evaluations.csv")
            combined = pd.merge(archive, configuration, how="cross")
            return combined 
        except:
            print(f"Unable to process run {run}")
    
    try:
        concatenated = pd.concat((merge(run) for run in tqdm(p.iterdir())))
        concatenated.to_csv(p / "evaluations.csv.gz")
    except:
        pass
        
        

def aggregate(p: Path):
    aggregate_archive(p)

for subfolder in (s for s in results_folder.iterdir() if s.is_dir()):
    # Skip over runs that have already been aggregated.
    if is_aggregated(subfolder):
        continue

    # Otherwise, collect data
    aggregate(subfolder)
