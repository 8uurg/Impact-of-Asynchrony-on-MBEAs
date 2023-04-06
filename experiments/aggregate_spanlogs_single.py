from pathlib import Path
import pandas as pd
from tqdm import tqdm

results_folder = Path("results")

def is_aggregated(p: Path):
    return (p / "spans.csv.gz").exists()
        
def aggregate_events(p: Path):

    def merge(run: Path):
        try:
            configuration = pd.read_json(run / "configuration.json", lines=True)
            path = run / "spans.csv"
            if not path.exists():
                # Skip if not exists.
                return None

            archive = pd.read_csv(path)
            combined = pd.merge(archive, configuration, how="cross")
            return combined 
        except Exception as e:
            print(f"Unable to process run {run}: {e}")
    
    try:
        concatenated = pd.concat((merge(run) for run in tqdm(p.iterdir())))
        concatenated.to_csv(p / "spans.csv.gz")
    except Exception as e:
        # If we error out (i.e. because all None)
        # do not write the file to output.
        print(f"Aggregating for {p} failed: {e}")
        pass

def aggregate(p: Path):
    aggregate_events(p)

for subfolder in (s for s in results_folder.iterdir() if s.is_dir()):
    # Skip over runs that have already been aggregated.
    if is_aggregated(subfolder):
        continue

    # Otherwise, collect data
    aggregate(subfolder)
