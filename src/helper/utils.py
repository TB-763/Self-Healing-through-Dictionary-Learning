import json
import os
import torch
import shutil
import random
import config
import numpy as np
import pandas as pd
import helper.constants as constants
from datetime import datetime
from torch.utils.data import DataLoader
from baseline.baseline_build import get_baseline

# --------------------------------------------------
# Set Seeds for Reproducability
# --------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# --------------------------------------------------
# Saves a copy of the used config
# --------------------------------------------------
def save_config_copy(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy("./src/config.py", os.path.join(target_dir, "config.py"))

def first_step_above_threshold(steps, values, threshold=0.3, min_step=0):
    for s, v in zip(steps, values):
        if s >= min_step and v is not None and v >= threshold:
            return s
    return None

def first_step_under_threshold_in_defense(steps, values, threshold):
    for s, v in zip(steps, values):
        if s >= config.OPTIMIZATION_STEPS and v is not None and v <= threshold:
            return s
    return None

# --------------------------------------------------
# Save final training statistics to a text file
# --------------------------------------------------
def save_stats_to_txt(log_tuple, training_time_min):
    log_steps      = log_tuple[0]
    train_accs     = log_tuple[1]
    val_accs       = log_tuple[2]
    backdoor_accs  = log_tuple[5]

    final_train_acc = train_accs[-1]
    final_val_acc   = val_accs[-1]
    final_step      = log_steps[-1]

    final_backdoor_acc = (
        backdoor_accs[-1] if backdoor_accs[-1] is not None else None
    )

    clean_baseline = get_baseline(config.MODEL, baseline_type="clean")

    train_30_step = first_step_above_threshold(log_steps, train_accs, 0.3)
    val_30_step   = first_step_above_threshold(log_steps, val_accs, 0.3)
    backdoor_30_step = first_step_above_threshold(log_steps, backdoor_accs, 0.9, min_step=100)
   
    threshold = clean_baseline.get("defense_threshold", None)

    defense_clean_step = None
    if threshold is not None:
        defense_clean_step = first_step_under_threshold_in_defense(
            log_steps, backdoor_accs, threshold
        )


    step_distance = (
        val_30_step - train_30_step
        if train_30_step is not None and val_30_step is not None
        else None
    )

    path = os.path.join(constants.RUN_DIR, "final_accuracy.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Optimization steps : {final_step}\n")
        f.write(f"Training time (m)  : {training_time_min:.2f}\n")
        f.write(f"Training accuracy  : {final_train_acc:.4f}\n")
        f.write(f"Validation accuracy: {final_val_acc:.4f}\n")

        if final_backdoor_acc is not None:
            f.write(f"Backdoor accuracy  : {final_backdoor_acc:.4f}\n")
        f.write(f"Train acc ≥ 30% at step : {train_30_step}\n")
        f.write(f"Val acc ≥ 30% at step   : {val_30_step}\n")
        f.write(f"Backdoor acc ≥ 90% at step: {backdoor_30_step}\n")
        f.write(f"Step distance (val - train): {step_distance}\n")
        f.write(f"Defense clean acc ≤ threshold at step: {defense_clean_step}\n")

# --------------------------------------------------
# Optimizer selection
# --------------------------------------------------
def make_optimizer(model):
    params = model.parameters()
    if config.OPTIMIZER == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=config.LR,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER == "RMSPROP":
        optimizer = torch.optim.RMSprop(
            params,
            lr=config.LR,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    return optimizer

# --------------------------------------------------
# Gets the Ftrojan attack parameters for diffrent datasets
# --------------------------------------------------
def get_FTROJAN_PARAM():
    if config.MODEL.startswith("CIFAR"):
        return constants.FTROJAN_PARAM_CIFAR 
    elif config.MODEL.startswith("MNIST"):
        return constants.FTROJAN_PARAM_MNIST
    elif config.MODEL.startswith("GTSRB"):
        return constants.FTROJAN_PARAM_GTSRB

def fmt(x):
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)

# --------------------------------------------------
# For each run, creates a folder in ./runs with data and timestamp
# --------------------------------------------------
def create_run_folder():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = os.path.join(
        constants.OUTPUT_DIR,
        timestamp
    )

    os.makedirs(run_dir, exist_ok=True)
    constants.RUN_DIR = run_dir

    config_save = {
        # --------------------
        # Model / data
        # --------------------
        "MODEL": config.MODEL,
        "LOSS_FUNCTION": config.LOSS_FUNCTION,
        "BATCH_SIZE": config.BATCH_SIZE,

        # --------------------
        # Optimizer
        # --------------------
        "OPTIMIZER": config.OPTIMIZER,
        "LR": config.LR,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,

        # --------------------
        # Training
        # --------------------
        "INITIALIZATION_SCALE": config.INITIALIZATION_SCALE,
        "OPTIMIZATION_STEPS": config.OPTIMIZATION_STEPS,
        "SEED": config.SEED,
        "DTYPE": str(constants.DTYPE),
        "DEVICE": str(constants.DEVICE),
        
        # --------------------
        # Backdoor / noise
        # --------------------
        "PIXEL_ATTACK": config.PIXEL_ATTACK,
        "FTROJAN_ATTACK": config.FTROJAN_ATTACK,
        "POISONED_FRACTION": config.POISONED_FRACTION,
        "FTROJAN_PARAM": get_FTROJAN_PARAM()
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

# --------------------------------------------------
# logging schedule
# --------------------------------------------------
def is_logging_step(phase_step, steps_per_epoch, phase):

    if phase_step == 0 and phase == 1:
        return False
    elif phase_step == 0 and phase == 2:
        return True
    log = False

    if phase_step % steps_per_epoch == 0 and phase_step <= 3000:
        log = True

    if phase_step >= 3000:
        if phase_step < 10000:
            log = phase_step % 500 == 0
        elif phase_step < 30000:
            log = phase_step % 1000 == 0
        elif phase_step < 50000:
            log = phase_step % 2000 == 0
        else:
            log = phase_step % 5000 == 0

    if phase_step == config.OPTIMIZATION_STEPS - 1:
        log = True
    if config.CONTINUE_CLEAN_TRAINING and phase_step == config.DEFENSE_STEPS - 1:
        log = True
    return log

def is_dict_module(m):
    return (
        hasattr(m, "set_trainable")
        or hasattr(m, "raw_D")
        or (hasattr(m, "D") and hasattr(m, "W_e"))
    )

def set_dictionary_trainable(model, trainable: bool) -> bool:
    base = model.module if hasattr(model, "module") else model
    changed = False

    for m in base.modules():
        if not is_dict_module(m):
            continue

        if hasattr(m, "set_trainable") and callable(getattr(m, "set_trainable")):
            before = getattr(m, "raw_D", None)
            m.set_trainable(trainable)
            changed = True
        else:
            for name, p in m.named_parameters(recurse=False):
                if name == "raw_D" or name == "D" or name == "W_e":
                    if p.requires_grad != trainable:
                        p.requires_grad_(trainable)
                        changed = True

    return changed

def make_eval_loader(dataset, batch_size):
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

def save_accs(log_dict):
    df = pd.DataFrame({
        "step": log_dict["steps"],
        "train_acc": log_dict["train_acc"],
        "val_acc": log_dict["val_acc"],
        "backdoor_acc": log_dict["backdoor_acc"],
    })

    path = os.path.join(constants.RUN_DIR, "accs.csv")
    df.to_csv(path, index=False)