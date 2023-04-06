import subprocess
import os

# Ensure the project is set-up and configured correctly
# and the library is installed into the current environment.
subprocess.check_output(["bash", "../EALib/install.sh"])

# Force OpenMP to restrict itself to one thread.
# Otherwise it will use more than one core, and become somewhat unpredictable.
os.environ["OMP_NUM_THREADS"] = "1"

failures = []
# For each line in runs.txt, add it to pueue. 
with open("runs.txt", "r") as f:
    for cmd in f:
        cmd_s = cmd.strip()
        try:
            subprocess.check_output(["pueue", "add", cmd_s])
        except:
            failures.append(["pueue", "add", cmd_s])


while len(failures) > 0:
    retrying = failures
    failures = []
    for failed_cmd in failures:
        try:
            subprocess.check_output(failed_cmd)
        except:
            failures.append(failed_cmd)

print("All commands have been sent to pueue.")