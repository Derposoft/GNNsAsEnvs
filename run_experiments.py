"""
experiments must be defined in configs/experiments.json in the following format:

{
    "experiment1": {
        "flag1 (e.g. fixed_start or init_health)": 2
    },
    "experiment2": {
        "flag1": 1,
        "flag2": 2
    },
    ...
}
"""

import json
import multiprocessing
import subprocess
import shlex
from multiprocessing.pool import ThreadPool


N_PROCS = 2     #multiprocessing.cpu_count() // 2
N_SEEDS = 2
START_SEED = 0

# read experiments.json
with open("configs/experiments/experiments.json", "r") as f:
    experiments = json.load(f)

# turn experiments in json into cmdline commands
experiment_cmds = []
for experiment_name in experiments:
    flags = experiments[experiment_name]
    for i in range(N_SEEDS):
        flags["name"] = experiment_name + f"_SEED{i+START_SEED}"
        experiment_cmds.append(
            "python3 train.py"
            + "".join([f" --{flag}={flags[flag]}" for flag in flags])
            + f" --seed {i+START_SEED}"
        )

# https://stackoverflow.com/questions/25120363/multiprocessing-execute-external-command-and-wait-before-proceeding
def call_proc(cmd):
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)
print(f"running on {multiprocessing.cpu_count()} cpus")
print(f"choosing to run {N_PROCS} processes")
pool = ThreadPool(N_PROCS)
results = []
for cmd in experiment_cmds:
    print(f"starting exp parametrized by: {cmd}")
    results.append(pool.apply_async(call_proc, (cmd,)))
pool.close()
pool.join()
