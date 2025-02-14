#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# Quick script to tarball a set of csv.gz files from particular runs.
# Runs in this case is a name (or glob thereof) of a experiment in the results directory.

import argparse
from pathlib import Path
from datetime import date
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("output")
parser.add_argument("input", nargs='+')
parsed = parser.parse_args()

today = date.today()

base_path = Path("results")
suffixes = ["archive.csv.gz"]
date_prefix = f"{today.year:04}-{today.month:02}-{today.day:02}"
target_path = base_path / f"{date_prefix}-{parsed.output}.tar.gz"

input_paths = []

for in_glob in parsed.input:
    for path in base_path.glob(in_glob):
        for suffix in suffixes:
            suffixed_path = path / suffix
            if suffixed_path.exists():
                # Shallowified name
                relativeified = suffixed_path.relative_to(base_path)
                shallow_name = "-".join(relativeified.parts)
                flattened_path = base_path / f"{date_prefix}-{shallow_name}"
                print(f"copying {suffixed_path} to {flattened_path}")
                subprocess.check_output(["cp", suffixed_path, flattened_path])
                input_paths.append(flattened_path)

joiner_str = "`,`"
print(f"gathering `{joiner_str.join(str(a) for a in input_paths)}` into `{target_path}`")
subprocess.check_output(["tar", "-czf", target_path] + input_paths)

