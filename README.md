# The Impact of Asynchrony on MBEAs - Source Code

This repository contains the source code for 'The Impact of Asynchrony on MBEAs'.
- General implementation in C++ can be found under EALib
- Scripts related to running, setting up & processing data can be found under experiments.
- The content of `experiments/results` can be found in a zip file under releases.

## Usage
```bash
cd experiments
# create environment - assuming poetry (python-poetry.org) is installed (i.e., using pip install poetry).
poetry install
# install python extension
poetry shell
# in the newly opened shell - assuming meson is installed (i.e., using pip install meson)
../EALib/install.sh
# pick an experiment script, if you wish to run multiple, use --append on the next runs.
# this fills up `runs.txt` with the commands to be ran for the experiments.
<run experiment script(s) here>
# Finally, run these commands using (assuming pueue, https://github.com/Nukesor/pueue, is installed)
# still within the environment (!)
# Alternative methods for running this list of commands should work too. Do ensure that the
# environment variable OMP_NUM_THREADS is set to 1.
./run_experiments.sh
```

# Credit
 DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability

This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.

Project leaders: Peter A.N. Bosman, Tanja Alderliesten
Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
Main code developer: Arthur Guijt
