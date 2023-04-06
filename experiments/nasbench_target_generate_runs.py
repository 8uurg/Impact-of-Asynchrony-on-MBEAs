# nasbench_target_generate_runs
f = open("runs.txt", "w")

for i in range(10):
    seed = i + 42
    f.write(f"python ./run_nasbench_serial_gomea_ims.py {seed}\n")