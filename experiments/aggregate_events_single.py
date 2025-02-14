#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from pathlib import Path
import pandas as pd
from tqdm import tqdm

results_folder = Path("results")

def is_aggregated(p: Path):
    return (p / "events.csv.gz").exists()
        
def aggregate_events(p: Path):

    def merge(run: Path):
        try:
            configuration = pd.read_json(run / "configuration.json", lines=True)
            path = run / "events.jsonl"
            if not path.exists():
                # Skip if not exists.
                return None

            archive = pd.read_json(path, orient="records", lines=True)
            combined = pd.merge(archive, configuration, how="cross")
            return combined 
        except Exception as e:
            print(f"Unable to process run {run}: {e}")
    
    try:
        concatenated = pd.concat((merge(run) for run in tqdm(p.iterdir())))
        concatenated.to_csv(p / "events.csv.gz")
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
