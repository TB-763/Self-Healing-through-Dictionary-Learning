import os
import re
import sys
import random
import subprocess
from copy import deepcopy
from itertools import product
from datetime import datetime, timedelta
from tkinter import ALL

# ------------------------------------------------------------
# 1) RUN SETTINGS
# ------------------------------------------------------------

DRY_RUN = False  # True = only print the plan, don't touch config.py

SELECTED_DATASETS = ["MNIST"]   # CIFAR10 | MNIST | GTSRB | "ALL"
SELECTED_ARCHITECTURES = ["RESNET18"]    # FTROJAN | DENSENET121 | EFFICIENTNET_B0 | RESNET18 | "ALL"
EXPERIMENTS_TO_RUN = []    # ["clean", "self_healing", "cdict", "pdict"] | "ALL"

# How many seeds to run per model, experiment, sweep-combination.
NUM_SEEDS = 5
RANDOM_SEEDS = True

# ------------------------------------------------------------
# PARAMETER SWEEP
# ------------------------------------------------------------
# Multiple parameters:
#     SWEEP_PARAMS = {"DICT_L1_LAMBDA": [0.04, 0.05], "LR": [0.01, 0.05]}
#     This creates every combination:
#     0.04 + 0.01, 0.04 + 0.05, 0.05 + 0.01, 0.05 + 0.05

SWEEP_PARAMS = {
   "DICT_L1_LAMBDA": [0.06]#, 0.1],
   #"DICT_NUM_ATOMS": [256]#, 256],
   #"DICT_PATCH_SIZE": [16, 4] # 4
}

# ------------------------------------------------------------
# GLOBAL PARAMETERS
# ------------------------------------------------------------
# Applied to every run, None means: "Do not modify this value in config.py."
GLOBAL_PARAMS = {
    "OPTIMIZER": None,
    "LR": None,
    "WEIGHT_DECAY": None,
    "USE_SCHEDULER": None,
    "DICT_EPOCHS": None,
    "DICT_USE_MEAN_CENTERING": None,
}

MIN_MINUTES_PER_RUN = 15
MAX_MINUTES_PER_RUN = 50

# ------------------------------------------------------------
# 2) DATASET / MODEL CATALOG
# ------------------------------------------------------------
DATASETS = {
    "CIFAR10": {"OPTIMIZATION_STEPS": 15065, "DEFENSE_STEPS": 14935,
                "ARCHITECTURES": ["FTROJAN", "DENSENET121", "RESNET18", "EFFICIENTNET_B0"]},
    "MNIST":   {"OPTIMIZATION_STEPS": 18055, "DEFENSE_STEPS": 17898,
                "ARCHITECTURES": ["FTROJAN", "DENSENET121", "RESNET18", "EFFICIENTNET_B0"]},
    "GTSRB":   {"OPTIMIZATION_STEPS": 8050, "DEFENSE_STEPS": 7980,
                "ARCHITECTURES": ["FTROJAN", "DENSENET121", "RESNET18", "EFFICIENTNET_B0"]},
}
# ------------------------------------------------------------
# 3) EXPERIMENT LIBRARY
# ------------------------------------------------------------
_DICT_DEFAULTS = {
    "DICT_MODE": "image", "DICT_OUTPUT": "reconstruction",
    "DICT_L1_LAMBDA": 0.02, "DICT_NUM_ATOMS": 512,
    "DICT_ISTA_ITERS": 5, "DICT_PATCH_SIZE": 8,
}

EXPERIMENT_LIBRARY = {
    "clean": {
        "PIXEL_ATTACK": False, "FTROJAN_ATTACK": False,
        "DICT_MODE": "none", "DICT_OUTPUT": "reconstruction", "DICT_POISON": False,
        "CONTINUE_CLEAN_TRAINING": False, "_uses_defense_steps": False,
    },
    "self_healing": {
        "PIXEL_ATTACK": False, "FTROJAN_ATTACK": True,
        "DICT_MODE": "none", "DICT_OUTPUT": "reconstruction", "DICT_POISON": False,
        "CONTINUE_CLEAN_TRAINING": True,
    },
    "cdict": {
        "PIXEL_ATTACK": False, "FTROJAN_ATTACK": True, **_DICT_DEFAULTS,
        "DICT_POISON": False, "CONTINUE_CLEAN_TRAINING": True,
    },
    "pdict": {
        "PIXEL_ATTACK": False, "FTROJAN_ATTACK": True, **_DICT_DEFAULTS,
        "DICT_POISON": True, "CONTINUE_CLEAN_TRAINING": True,
    },
}

# ------------------------------------------------------------
# 4) PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CONFIG_FILE = os.path.join(BASE_DIR, "config.py")
MAIN_FILE = os.path.join(BASE_DIR, "main.py")

# ------------------------------------------------------------
# 5) HELPERS
# ------------------------------------------------------------
def resolve_selected_datasets():
    if SELECTED_DATASETS == "ALL":
        return list(DATASETS.keys())
    unknown = [d for d in SELECTED_DATASETS if d not in DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s) {unknown}. "
            f"Available: {list(DATASETS)}"
        )
    return list(SELECTED_DATASETS)

def build_model_list():
    datasets = resolve_selected_datasets()
    if SELECTED_ARCHITECTURES == "ALL":
        requested = None
    else:
        requested = list(SELECTED_ARCHITECTURES)
        available_anywhere = {a for d in datasets for a in DATASETS[d]["ARCHITECTURES"]}
        unknown = [a for a in requested if a not in available_anywhere]
        if unknown:
            raise ValueError(
                f"Architecture(s) {unknown} not available for "
                f"any selected dataset {datasets}. "
                f"Available across those datasets: "
                f"{sorted(available_anywhere)}"
            )

    models = []

    for dataset in datasets:
        available = DATASETS[dataset]["ARCHITECTURES"]
        if requested is None:
            archs = available
        else:
            archs = [
                a for a in requested
                if a in available
            ]
        if not archs:
            print(f"[WARN] No selected architecture applies to dataset '{dataset}' (available there: {available}); skipping it.")
            continue
        for arch in archs:
            models.append({
                "name": f"{dataset}_{arch}",
                "dataset": dataset,
                "architecture": arch,
            })
    if not models:
        raise ValueError(
            "No models resolved - check "
            "SELECTED_DATASETS / SELECTED_ARCHITECTURES."
        )
    return models

def generate_seed_pool(total_seeds):
    if RANDOM_SEEDS:
        rng = random.SystemRandom()
        seeds = set()
        while len(seeds) < total_seeds:
            seeds.add(
                rng.randrange(1, 2**31 - 1)
            )
        seeds = list(seeds)
        rng.shuffle(seeds)
        return seeds
    return list(range(total_seeds))

def resolve_experiments():
    if EXPERIMENTS_TO_RUN is None:
        return [
            ("none", {})
        ]

    if isinstance(EXPERIMENTS_TO_RUN, str):

        mode = EXPERIMENTS_TO_RUN.upper()

        if mode in ("NONE", "SWEEPS"):
            return [
                ("none", {})
            ]

        if mode == "ALL":
            names = list(EXPERIMENT_LIBRARY.keys())

        else:
            raise ValueError(
                f"Unknown EXPERIMENTS_TO_RUN mode "
                f"'{EXPERIMENTS_TO_RUN}'. "
                f"Use 'NONE', 'SWEEPS', 'ALL', "
                f"or an explicit list such as ['cdict']."
            )

    else:
        names = list(EXPERIMENTS_TO_RUN)
        if not names:
            return [
                ("none", {})
            ]

    for n in names:
        if n not in EXPERIMENT_LIBRARY:
            raise ValueError(
                f"Unknown experiment '{n}'. "
                f"Available: {list(EXPERIMENT_LIBRARY)}"
            )

    return [
        (n, EXPERIMENT_LIBRARY[n])
        for n in names
    ]

def sweep_combinations():
    if not SWEEP_PARAMS:
        return [{}]

    keys = list(SWEEP_PARAMS.keys())

    value_lists = [
        SWEEP_PARAMS[k]
        for k in keys
    ]

    # Validate that every sweep has at least one value.
    for key, values in SWEEP_PARAMS.items():
        if not values:
            raise ValueError(
                f"Sweep parameter '{key}' has no values."
            )

    return [
        dict(zip(keys, combo))
        for combo in product(*value_lists)
    ]

def change_config(parameter, value):
    if value is None:
        return

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = rf"^{re.escape(parameter)}\s*=.*$"
    replacement = (
        f"{parameter} = {repr(value)}"
    )

    new_content, count = re.subn(
        pattern,
        replacement,
        content,
        flags=re.MULTILINE,
    )

    if count == 0:
        raise ValueError(
            f"Parameter '{parameter}' was not found "
            f"in {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)


def apply_config(run):
    change_config("MODEL", run["model"])
    change_config("SEED", run["seed"])

    for param, value in GLOBAL_PARAMS.items():
        change_config(param, value)

    for param, value in run["params"].items():
        if param.startswith("_"):
            continue

        change_config(param, value)

def run_main():
    result = subprocess.run(
        [sys.executable, MAIN_FILE],
        cwd=PROJECT_ROOT,
    )

    return result.returncode == 0

def format_minutes(minutes):
    hours = minutes / 60

    return (
        f"{minutes:,.0f} min "
        f"(~{hours:,.1f} h)"
    )

# ------------------------------------------------------------
# 6) BUILD THE RUN QUEUE
# ------------------------------------------------------------
def build_run_queue(models):
    experiments = resolve_experiments()
    combos = sweep_combinations()
    total_slots = (len(models) * len(experiments) * len(combos) * NUM_SEEDS)
    seed_pool = iter(generate_seed_pool(total_slots))
    queue = []
    for model in models:
        steps = DATASETS[model["dataset"]]
        for exp_name, exp_params in experiments:
            base = deepcopy(exp_params)
            base["OPTIMIZATION_STEPS"] = (steps["OPTIMIZATION_STEPS"])
            base["DEFENSE_STEPS"] = (steps["DEFENSE_STEPS"])
            for combo in combos:
                params = deepcopy(base)
                params.update(combo)
                for seed_idx in range(1, NUM_SEEDS + 1):
                    queue.append({
                        "model": model["name"],
                        "dataset": model["dataset"],
                        "architecture": model["architecture"],
                        "experiment": exp_name,
                        "sweep": combo,
                        "seed_index": seed_idx,
                        "seed": next(seed_pool),
                        "params": params,
                    })
    return queue

def describe_run(run):
    parts = [
        f"model={run['model']}",
        f"experiment={run['experiment']}",
    ]
    for k, v in run["sweep"].items():
        parts.append(
            f"{k}={v}"
        )
    parts.append(
        f"seed={run['seed']} "
        f"(#{run['seed_index']}/{NUM_SEEDS})"
    )
    return " | ".join(parts)

# ------------------------------------------------------------
# 7) MAIN
# ------------------------------------------------------------
def main():
    models = build_model_list()
    queue = build_run_queue(models)
    total = len(queue)
    unique_seeds_used = len({
        run["seed"]
        for run in queue
    })
    resolved_experiments = resolve_experiments()
    print("=" * 75)
    print("RUN PLAN")
    print("=" * 75)
    print(f'Datasets selected:   {resolve_selected_datasets()}')
    print(f'Architectures:       {SELECTED_ARCHITECTURES}')
    print(f"Models resolved:     {[m['name'] for m in models]}")
    print(f'Experiments:         {[e for e, _ in resolved_experiments]}')
    if EXPERIMENTS_TO_RUN in (
        None,
        "NONE",
        "SWEEPS",
    ) or not EXPERIMENTS_TO_RUN:
        print('Experiment mode:     SWEEP-ONLY / no experiment preset')
    print(f"Sweep parameters:    {(SWEEP_PARAMS if SWEEP_PARAMS else '(none)')}")
    print(f'Sweep combinations:  {len(sweep_combinations())}')
    print(f'Seeds per combo:     {NUM_SEEDS} (random={RANDOM_SEEDS})')
    print(f"Unique seeds used:   {unique_seeds_used} / {total} runs {('(all unique)' if unique_seeds_used == total else '(!) DUPLICATES FOUND')}")
    print(f'Dry run:             {DRY_RUN}')
    print(f'Total runs:          {total}')
    now = datetime.now()
    min_runtime = total * MIN_MINUTES_PER_RUN
    max_runtime = total * MAX_MINUTES_PER_RUN
    avg_runtime = total * (MIN_MINUTES_PER_RUN + MAX_MINUTES_PER_RUN) / 2
    min_finish = now + timedelta(minutes=min_runtime)
    max_finish = now + timedelta(minutes=max_runtime)
    avg_finish = now + timedelta(minutes=avg_runtime)

    print(f'Current time:        {now:%Y-%m-%d %H:%M:%S}\nEst. runtime:        {format_minutes(min_runtime)}  to  {format_minutes(max_runtime)}  (avg ~{format_minutes(avg_runtime)})\nEst. finish:         {min_finish:%H:%M:%S}  to  {max_finish:%H:%M:%S}  (avg ~{avg_finish:%H:%M:%S})')
    print(f'                     [{MIN_MINUTES_PER_RUN}-{MAX_MINUTES_PER_RUN} min/run assumed]')
    print("=" * 75)

    if DRY_RUN:
        for i, run in enumerate(
            queue,
            start=1,
        ):
            print(f'[{i}/{total}] {describe_run(run)}')
            print(f"           params={run['params']}")
        print()
        print('Dry run only - nothing was executed.')
        return

    completed = 0
    failed = 0
    for i, run in enumerate(
        queue,
        start=1,
    ):
        print()
        print("=" * 75)

        print(f'RUN {i}/{total} | {describe_run(run)}')
        print("=" * 75)
        try:
            apply_config(run)
            success = run_main()
            if success:
                completed += 1
                print(f'[SUCCESS] {describe_run(run)}')
            else:
                failed += 1

                print(f'[FAILED]  {describe_run(run)}')
        except Exception as e:
            failed += 1
            print(f'[ERROR]   {describe_run(run)}')
            print(f'Reason: {e}')
    print()
    print("=" * 75)
    print("RUNS FINISHED")
    print(f'Successful: {completed}')
    print(f'Failed:     {failed}')
    print(f'Total:      {total}')
    print("=" * 75)

if __name__ == "__main__":
    main()