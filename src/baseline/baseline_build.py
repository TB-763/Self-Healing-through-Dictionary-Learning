import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import importlib.util

BASELINE_DB_PATH = os.path.join("./src/baseline/", "baseline_db.json")
BASELINE_ROOT = os.path.join("runs", "baseline")
IMAGE_RECON_ROOT = os.path.join("runs", "image_recon")
FEATURE_RECON_ROOT = os.path.join("runs", "feature_recon")
FEATURE_CODES_ROOT = os.path.join("runs", "feature_codes")

# --------------------------------------------------
# Threshold settings
# --------------------------------------------------
USE_CLEAN_RUNS = False
CLEAN_THRESHOLD_FACTOR = 1.10
CIFAR10_THR = 0.15
GTSRB_THR = 0.07
MNIST_THR = 0.15

# --------------------------------------------------
# IO
# --------------------------------------------------
def _load_db() -> Dict:
    if not os.path.exists(BASELINE_DB_PATH):
        return {}
    try:
        with open(BASELINE_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load baseline DB: {e}")
        return {}

def _save_db(db: Dict):
    os.makedirs(os.path.dirname(BASELINE_DB_PATH), exist_ok=True)
    try:
        with open(BASELINE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save baseline DB: {e}")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _safe_stats(values: List[float]) -> Optional[Dict[str, float]]:
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isnan(fv):
            continue
        vals.append(fv)
    if len(vals) == 0:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def _select_exact_or_nearest_point(
    steps: List[Any],
    values: List[Any],
    target_step: int
) -> Optional[Tuple[int, float, float]]:
    if not steps or not values:
        return None
    n = min(len(steps), len(values))
    if n == 0:
        return None
    exact_idx = None
    for i in range(n):
        try:
            if float(steps[i]) == float(target_step):
                exact_idx = i
                break
        except Exception:
            continue
    if exact_idx is not None:
        try:
            return exact_idx, float(steps[exact_idx]), float(values[exact_idx])
        except Exception:
            return None
    best_idx = None
    best_dist = None
    for i in range(n):
        try:
            s = float(steps[i])
            v = float(values[i])
        except Exception:
            continue
        dist = abs(s - float(target_step))
        if best_idx is None or dist < best_dist:
            best_idx = i
            best_dist = dist
    if best_idx is None:
        return None
    try:
        return best_idx, float(steps[best_idx]), float(values[best_idx])
    except Exception:
        return None

def _fmt_float(v: Any) -> str:
    try:
        return f"{float(v):g}"
    except Exception:
        return str(v)

def _print_run_table_expanded(
    clean_runs_by_model: Dict,
    defense_runs_by_model: Dict,
    image_recon_runs_by_model: Dict,
    feature_recon_runs_by_model: Dict,
    feature_codes_runs_by_model: Dict,
):
    print("\n[SUMMARY] Run counts (expanded)\n")
    header = f"{'MODEL':25} | {'TYPE':14} | {'SUBKEY':45} | RUNS"
    print(header)
    print("-" * len(header))
    all_models = (
        set(clean_runs_by_model.keys())
        | set(defense_runs_by_model.keys())
        | set(image_recon_runs_by_model.keys())
        | set(feature_recon_runs_by_model.keys())
        | set(feature_codes_runs_by_model.keys())
    )
    for model in sorted(all_models):
        if model in clean_runs_by_model:
            print(f"{model:25} | {'clean':14} | {'-':45} | {len(clean_runs_by_model[model])}")
        if model in defense_runs_by_model:
            print(f"{model:25} | {'defense':14} | {'-':45} | {len(defense_runs_by_model[model])}")
        if model in image_recon_runs_by_model:
            for subkey, runs in sorted(image_recon_runs_by_model[model].items()):
                print(f"{model:25} | {'image_recon':14} | {subkey:45} | {len(runs)}")
        if model in feature_recon_runs_by_model:
            for subkey, runs in sorted(feature_recon_runs_by_model[model].items()):
                print(f"{model:25} | {'feature_recon':14} | {subkey:45} | {len(runs)}")
        if model in feature_codes_runs_by_model:
            for subkey, runs in sorted(feature_codes_runs_by_model[model].items()):
                print(f"{model:25} | {'feature_codes':14} | {subkey:45} | {len(runs)}")
    print()

def _get_dataset_threshold(model_key: str) -> float:
    key = (model_key or "").upper()
    if "CIFAR10" in key:
        return float(CIFAR10_THR)
    if "GTSRB" in key:
        return float(GTSRB_THR)
    if "MNIST" in key:
        return float(MNIST_THR)
    return 1.0

def _get_clean_threshold(model_key: str, clean_asr_max_cache: Dict[str, float]) -> float:
    if USE_CLEAN_RUNS:
        base = clean_asr_max_cache.get(model_key, _get_dataset_threshold(model_key))
        result = float(base) * float(CLEAN_THRESHOLD_FACTOR)
        print(
            f"[DEBUG _get_clean_threshold] model_key={model_key}, "
            f"base={base}, factor={CLEAN_THRESHOLD_FACTOR}, result={result}, "
            f"source={'cache' if model_key in clean_asr_max_cache else 'dataset'}"
        )
        return result
    result = _get_dataset_threshold(model_key)
    print(
        f"[DEBUG _get_clean_threshold] model_key={model_key}, "
        f"result={result}, source=dataset_only"
    )
    return result

# --------------------------------------------------
# Config loading
# --------------------------------------------------
def load_config_from_folder(folder_path: str):
    config_path = os.path.join(folder_path, "config.py")
    if not os.path.exists(config_path):
        return None
    try:
        module_name = "config_temp_" + hashlib.md5(folder_path.encode("utf-8")).hexdigest()
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            return None
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        return cfg
    except Exception as e:
        print(f"[WARN] Failed to load config from {folder_path}: {e}")
        return None

def get_baseline_type(cfg) -> Optional[str]:
    if cfg is None:
        return None
    try:
        if (
            not getattr(cfg, "FTROJAN_ATTACK", True)
            and not getattr(cfg, "PIXEL_ATTACK", True)
            and not getattr(cfg, "CONTINUE_CLEAN_TRAINING", True)
            and getattr(cfg, "DICT_MODE", None) == "none"
        ):
            return "clean"
        if (
            getattr(cfg, "FTROJAN_ATTACK", False)
            and not getattr(cfg, "PIXEL_ATTACK", True)
            and getattr(cfg, "CONTINUE_CLEAN_TRAINING", False)
            and getattr(cfg, "DICT_MODE", None) == "none"
        ):
            return "defense"
    except Exception:
        pass
    return None

def get_image_recon_subkey(cfg) -> Optional[str]:
    if cfg is None:
        return None
    try:
        if getattr(cfg, "DICT_MODE", None) != "image":
            return None
        if getattr(cfg, "DICT_OUTPUT", None) != "reconstruction":
            return None
        atoms = getattr(cfg, "DICT_NUM_ATOMS", 512)
        l1 = getattr(cfg, "DICT_L1_LAMBDA", 0.02)
        patch = getattr(cfg, "DICT_PATCH_SIZE", 8)
        poison = getattr(cfg, "DICT_POISON", False)
        poison_str = "T" if poison else "F"
        return f"atoms{atoms}_l1{_fmt_float(l1)}_patch{patch}_poi{poison_str}"
    except Exception:
        return None

def get_feature_recon_subkey(cfg) -> Optional[str]:
    if cfg is None:
        return None
    try:
        if getattr(cfg, "DICT_MODE", None) != "feature":
            return None
        if getattr(cfg, "DICT_OUTPUT", None) != "reconstruction":
            return None
        atoms = getattr(cfg, "DICT_NUM_ATOMS", 512)
        l1 = getattr(cfg, "DICT_L1_LAMBDA", 0.02)
        recon_w = getattr(cfg, "DICT_RECON_WEIGHT", 0.1)
        freeze = getattr(cfg, "DICT_ENABLE_FREEZE_UNFREEZE", False)
        freeze_str = "T" if freeze else "F"
        if freeze:
            freeze_at = getattr(cfg, "DICT_FREEZE_AT", 500)
            unfreeze_at = getattr(cfg, "DICT_UNFREEZE_AT", 1000)
            return (
                f"atoms{atoms}_l1{_fmt_float(l1)}_rw{_fmt_float(recon_w)}"
                f"_fz{freeze_str}_{freeze_at}-{unfreeze_at}"
            )
        return f"atoms{atoms}_l1{_fmt_float(l1)}_rw{_fmt_float(recon_w)}_fz{freeze_str}"
    except Exception:
        return None

def get_feature_codes_subkey(cfg) -> Optional[str]:
    if cfg is None:
        return None
    try:
        if getattr(cfg, "DICT_MODE", None) != "feature":
            return None
        if getattr(cfg, "DICT_OUTPUT", None) != "codes":
            return None
        atoms = getattr(cfg, "DICT_NUM_ATOMS", 512)
        l1 = getattr(cfg, "DICT_L1_LAMBDA", 0.02)
        recon_w = getattr(cfg, "DICT_RECON_WEIGHT", 0.1)
        freeze = getattr(cfg, "DICT_ENABLE_FREEZE_UNFREEZE", False)
        freeze_str = "T" if freeze else "F"
        if freeze:
            freeze_at = getattr(cfg, "DICT_FREEZE_AT", 500)
            unfreeze_at = getattr(cfg, "DICT_UNFREEZE_AT", 1000)
            return (
                f"atoms{atoms}_l1{_fmt_float(l1)}_rw{_fmt_float(recon_w)}"
                f"_fz{freeze_str}_{freeze_at}-{unfreeze_at}"
            )
        return f"atoms{atoms}_l1{_fmt_float(l1)}_rw{_fmt_float(recon_w)}_fz{freeze_str}"
    except Exception:
        return None

# --------------------------------------------------
# CSV loading
# --------------------------------------------------
def load_run_from_csv(csv_path: str, cfg) -> Optional[Dict]:
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return {
            "val_acc": df["val_acc"].tolist(),
            "backdoor_acc": df["backdoor_acc"].tolist(),
            "steps": df["step"].tolist(),
            "final_val_acc": float(df["val_acc"].iloc[-1]),
            "final_asr": float(df["backdoor_acc"].iloc[-1]),
            "opt_steps": int(getattr(cfg, "OPTIMIZATION_STEPS", 0)),
            "defense_steps_limit": int(getattr(cfg, "DEFENSE_STEPS", 0)),
            # metadata
            "dict_num_atoms": getattr(cfg, "DICT_NUM_ATOMS", None),
            "dict_l1_lambda": getattr(cfg, "DICT_L1_LAMBDA", None),
            "dict_patch_size": getattr(cfg, "DICT_PATCH_SIZE", None),
            "dict_poison": getattr(cfg, "DICT_POISON", False),
            "dict_recon_weight": getattr(cfg, "DICT_RECON_WEIGHT", None),
            "dict_enable_freeze_unfreeze": getattr(cfg, "DICT_ENABLE_FREEZE_UNFREEZE", False),
            "image_recon_subkey": get_image_recon_subkey(cfg),
            "feature_recon_subkey": get_feature_recon_subkey(cfg),
            "feature_codes_subkey": get_feature_codes_subkey(cfg),
        }
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}")
        return None

# --------------------------------------------------
# Core aggregation
# --------------------------------------------------
def _find_stable_recovery_step(
    steps: List,
    asr_values: List,
    clean_threshold: float
) -> Tuple[Optional[float], bool]:
    if not steps or not asr_values or clean_threshold is None:
        print("[RECOVERY] Invalid input -> no recovery")
        return None, False
    pairs = []
    for s, a in zip(steps, asr_values):
        try:
            pairs.append((float(s), float(a)))
        except Exception:
            continue
    if not pairs:
        print("[RECOVERY] No valid points -> no recovery")
        return None, False
    pairs.sort(key=lambda x: x[0])
    sorted_steps = [s for s, _ in pairs]
    sorted_asr = [a for _, a in pairs]
    suffix_max = [0.0] * len(sorted_asr)
    running_max = float("-inf")
    for i in range(len(sorted_asr) - 1, -1, -1):
        running_max = max(running_max, sorted_asr[i])
        suffix_max[i] = running_max
    for i in range(len(sorted_steps)):
        if suffix_max[i] <= float(clean_threshold):
            recovery_step = sorted_steps[i]
            print(
                f"[RECOVERY] SUCCESS | "
                f"step={recovery_step:g} | "
                f"threshold={clean_threshold:g}"
            )
            return recovery_step, True
    print(
        f"[RECOVERY] FAILED | "
        f"threshold={clean_threshold:g}"
    )
    return None, False

def _compute_summary(
    runs: List[Dict],
    include_defense: bool = False,
    clean_asr_threshold: Optional[float] = None,
    is_cifar: bool = False
) -> Dict:
    valid_runs = [r for r in runs if r.get("val_acc") and r.get("backdoor_acc")]
    val_last = [r["val_acc"][-1] for r in valid_runs]
    asr_last = [r["backdoor_acc"][-1] for r in valid_runs]
    val_stats = _safe_stats(val_last)
    asr_stats = _safe_stats(asr_last)
    summary = {
        "run_count": len(valid_runs),
        "val_acc_mean": val_stats["mean"] if val_stats else None,
        "val_acc_std": val_stats["std"] if val_stats else None,
        "val_acc_min": val_stats["min"] if val_stats else None,
        "val_acc_max": val_stats["max"] if val_stats else None,
        "asr_mean": asr_stats["mean"] if asr_stats else None,
        "asr_std": asr_stats["std"] if asr_stats else None,
        "asr_min": asr_stats["min"] if asr_stats else None,
        "asr_max": asr_stats["max"] if asr_stats else None,
    }
    if include_defense and clean_asr_threshold is not None:
        asr_at_opt_values = []
        val_at_opt_values = []
        rise_peak_asr_values = []
        rise_peak_step_values = []
        defense_steps = []
        asr_auc_post_opt_values = []
        recovery_success_count = 0
        max_steps = 30000 if is_cifar else 16030
        for r in valid_runs:
            steps_seq = r.get("steps", [])
            asr_seq = r.get("backdoor_acc", [])
            val_seq = r.get("val_acc", [])
            if not steps_seq or not asr_seq:
                defense_steps.append(float(max_steps))
                continue
            opt_steps = int(r.get("opt_steps", 0) or 0)

            opt_asr_point = _select_exact_or_nearest_point(steps_seq, asr_seq, opt_steps)
            if opt_asr_point:
                _, _, asr_at_opt = opt_asr_point
                asr_at_opt_values.append(asr_at_opt)

            opt_val_point = _select_exact_or_nearest_point(steps_seq, val_seq, opt_steps)
            if opt_val_point:
                _, _, val_at_opt = opt_val_point
                val_at_opt_values.append(val_at_opt)

            pre_pairs = [(float(s), float(a)) for s, a in zip(steps_seq, asr_seq) if float(s) < opt_steps]
            if pre_pairs:
                pre_steps, pre_asr = zip(*pre_pairs)
                peak_idx = int(np.argmax(pre_asr))
                rise_peak_asr_values.append(pre_asr[peak_idx])
                rise_peak_step_values.append(pre_steps[peak_idx])

            steps_arr = np.asarray(steps_seq, dtype=float)
            asr_arr = np.asarray(asr_seq, dtype=float)
            order = np.argsort(steps_arr)
            steps_arr = steps_arr[order]
            asr_arr = asr_arr[order]
            opt_idx = np.searchsorted(steps_arr, opt_steps, side="left")
            if len(steps_arr) - opt_idx >= 2:
                auc_post_opt = float(np.trapezoid(asr_arr[opt_idx:], steps_arr[opt_idx:]))
                asr_auc_post_opt_values.append(auc_post_opt)

            phase2_pairs = [
                (float(s), float(a))
                for s, a in zip(steps_seq, asr_seq)
                if float(s) >= opt_steps
            ]
            if len(phase2_pairs) >= 2:
                phase2_steps, phase2_asr = zip(*phase2_pairs)
                recovery_step, is_success = _find_stable_recovery_step(
                    list(phase2_steps),
                    list(phase2_asr),
                    clean_asr_threshold
                )
            else:
                recovery_step, is_success = None, False

            if is_success and recovery_step is not None:
                recovery_success_count += 1
                defense_steps.append(recovery_step)
            else:
                defense_steps.append(float(max_steps))

        asr_at_opt_stats = _safe_stats(asr_at_opt_values)
        val_at_opt_stats = _safe_stats(val_at_opt_values)
        rise_peak_asr_stats = _safe_stats(rise_peak_asr_values)
        rise_peak_step_stats = _safe_stats(rise_peak_step_values)
        defense_step_stats = _safe_stats(defense_steps)
        auc_post_opt_stats = _safe_stats(asr_auc_post_opt_values)

        summary.update({
            "asr_at_opt_mean": asr_at_opt_stats["mean"] if asr_at_opt_stats else None,
            "asr_at_opt_std": asr_at_opt_stats["std"] if asr_at_opt_stats else None,
            "asr_at_opt_min": asr_at_opt_stats["min"] if asr_at_opt_stats else None,
            "asr_at_opt_max": asr_at_opt_stats["max"] if asr_at_opt_stats else None,
            "val_acc_at_opt_mean": val_at_opt_stats["mean"] if val_at_opt_stats else None,
            "val_acc_at_opt_std": val_at_opt_stats["std"] if val_at_opt_stats else None,
            "val_acc_at_opt_min": val_at_opt_stats["min"] if val_at_opt_stats else None,
            "val_acc_at_opt_max": val_at_opt_stats["max"] if val_at_opt_stats else None,
            "asr_rise_peak_mean": rise_peak_asr_stats["mean"] if rise_peak_asr_stats else None,
            "asr_rise_peak_std": rise_peak_asr_stats["std"] if rise_peak_asr_stats else None,
            "asr_rise_peak_min": rise_peak_asr_stats["min"] if rise_peak_asr_stats else None,
            "asr_rise_peak_max": rise_peak_asr_stats["max"] if rise_peak_asr_stats else None,
            "asr_rise_peak_step_mean": rise_peak_step_stats["mean"] if rise_peak_step_stats else None,
            "asr_rise_peak_step_std": rise_peak_step_stats["std"] if rise_peak_step_stats else None,
            "asr_rise_peak_step_min": rise_peak_step_stats["min"] if rise_peak_step_stats else None,
            "asr_rise_peak_step_max": rise_peak_step_stats["max"] if rise_peak_step_stats else None,
            "asr_auc_post_opt_mean": auc_post_opt_stats["mean"] if auc_post_opt_stats else None,
            "asr_auc_post_opt_std": auc_post_opt_stats["std"] if auc_post_opt_stats else None,
            "asr_auc_post_opt_min": auc_post_opt_stats["min"] if auc_post_opt_stats else None,
            "asr_auc_post_opt_max": auc_post_opt_stats["max"] if auc_post_opt_stats else None,
            "defense_step_mean": defense_step_stats["mean"] if defense_step_stats else None,
            "defense_step_std": defense_step_stats["std"] if defense_step_stats else None,
            "defense_step_min": defense_step_stats["min"] if defense_step_stats else None,
            "defense_step_max": defense_step_stats["max"] if defense_step_stats else None,
            "defense_success_rate": round(recovery_success_count / len(valid_runs), 4) if valid_runs else 0.0,
        })
    return summary

def _write_run_folders_txt(
    clean_runs_by_model: Dict,
    defense_runs_by_model: Dict,
    image_recon_runs_by_model: Dict,
    feature_recon_runs_by_model: Dict,
    feature_codes_runs_by_model: Dict,
    output_path: str = "src/baseline/baseline_run_folders.txt"
):
    lines = []
    lines.append("[SUMMARY] Run folders by model / group\n")
    lines.append(f"{'MODEL':25} | {'TYPE':14} | {'SUBKEY':45} | RUN FOLDERS")
    lines.append("-" * 110)
    all_models = sorted(
        set(clean_runs_by_model.keys())
        | set(defense_runs_by_model.keys())
        | set(image_recon_runs_by_model.keys())
        | set(feature_recon_runs_by_model.keys())
        | set(feature_codes_runs_by_model.keys())
    )
    for model in all_models:
        if model in clean_runs_by_model:
            lines.append(f"{model:25} | {'clean':14} | {'-':45} |")
            for r in clean_runs_by_model[model]:
                folder = r.get("folder", r.get("run_folder", "UNKNOWN"))
                lines.append(f"{'':25} | {'':14} | {'':45} | {folder}")
        if model in defense_runs_by_model:
            lines.append(f"{model:25} | {'defense':14} | {'-':45} |")
            for r in defense_runs_by_model[model]:
                folder = r.get("folder", r.get("run_folder", "UNKNOWN"))
                lines.append(f"{'':25} | {'':14} | {'':45} | {folder}")
        if model in image_recon_runs_by_model:
            for subkey, runs in sorted(image_recon_runs_by_model[model].items()):
                lines.append(f"{model:25} | {'image_recon':14} | {subkey:45} |")
                for r in runs:
                    folder = r.get("folder", r.get("run_folder", "UNKNOWN"))
                    lines.append(f"{'':25} | {'':14} | {'':45} | {folder}")
        if model in feature_recon_runs_by_model:
            for subkey, runs in sorted(feature_recon_runs_by_model[model].items()):
                lines.append(f"{model:25} | {'feature_recon':14} | {subkey:45} |")
                for r in runs:
                    folder = r.get("folder", r.get("run_folder", "UNKNOWN"))
                    lines.append(f"{'':25} | {'':14} | {'':45} | {folder}")
        if model in feature_codes_runs_by_model:
            for subkey, runs in sorted(feature_codes_runs_by_model[model].items()):
                lines.append(f"{model:25} | {'feature_codes':14} | {subkey:45} |")
                for r in runs:
                    folder = r.get("folder", r.get("run_folder", "UNKNOWN"))
                    lines.append(f"{'':25} | {'':14} | {'':45} | {folder}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Run folder summary written to {output_path}")

# --------------------------------------------------
# Main build
# --------------------------------------------------
def build_baseline_db():
    db: Dict = {}
    clean_asr_max_cache: Dict[str, float] = {}
    print(f"[INFO] Scanning baseline runs under {BASELINE_ROOT}")
    print(f"[INFO] Scanning image_recon runs under {IMAGE_RECON_ROOT}")
    print(f"[INFO] Scanning feature_recon runs under {FEATURE_RECON_ROOT}")
    print(f"[INFO] Scanning feature_codes runs under {FEATURE_CODES_ROOT}\n")

    clean_runs_by_model = {}
    defense_runs_by_model = {}
    image_recon_runs_by_model = {}
    feature_recon_runs_by_model = {}
    feature_codes_runs_by_model = {}

    # 1. Baseline root: clean + defense
    for root, _, files in os.walk(BASELINE_ROOT):
        if "accs.csv" not in files:
            continue
        cfg = load_config_from_folder(root)
        if not cfg:
            continue
        model_key = getattr(cfg, "MODEL", None)
        btype = get_baseline_type(cfg)
        if not model_key or btype not in ("clean", "defense"):
            continue
        run_data = load_run_from_csv(os.path.join(root, "accs.csv"), cfg)
        if run_data:
            run_data["folder"] = os.path.basename(root)
        if not run_data:
            continue
        print(f"[FOUND] {btype:8} | {model_key:20} | {os.path.basename(root)}")
        if btype == "clean":
            clean_runs_by_model.setdefault(model_key, []).append(run_data)
        else:
            defense_runs_by_model.setdefault(model_key, []).append(run_data)

    # Aggregate Clean
    for model_key, runs in clean_runs_by_model.items():
        db.setdefault(model_key, {})
        summary = _compute_summary(runs, include_defense=False)
        db[model_key]["clean"] = summary
        if summary.get("asr_max") is not None:
            clean_asr_max_cache[model_key] = summary["asr_max"]

    # Aggregate Defense
    for model_key, runs in defense_runs_by_model.items():
        db.setdefault(model_key, {})
        threshold = _get_clean_threshold(model_key, clean_asr_max_cache)
        is_cifar = "CIFAR10" in model_key
        summary = _compute_summary(
            runs,
            include_defense=True,
            clean_asr_threshold=threshold,
            is_cifar=is_cifar
        )
        db[model_key]["defense"] = summary

    # 2. image_recon
    for root, _, files in os.walk(IMAGE_RECON_ROOT):
        if "accs.csv" not in files:
            continue
        cfg = load_config_from_folder(root)
        if not cfg:
            continue
        model_key = getattr(cfg, "MODEL", None)
        subkey = get_image_recon_subkey(cfg)
        if not model_key or not subkey:
            continue
        run_data = load_run_from_csv(os.path.join(root, "accs.csv"), cfg)
        if run_data:
            run_data["folder"] = os.path.basename(root)
        if not run_data:
            continue
        print(f"[FOUND] image_recon | {model_key:20} | {subkey:35} | {os.path.basename(root)}")
        image_recon_runs_by_model.setdefault(model_key, {}).setdefault(subkey, []).append(run_data)

    for model_key, subgroups in image_recon_runs_by_model.items():
        db.setdefault(model_key, {}).setdefault("image_recon", {})
        is_cifar = "CIFAR10" in model_key
        threshold = _get_clean_threshold(model_key, clean_asr_max_cache)
        for subkey, runs in subgroups.items():
            summary = _compute_summary(
                runs,
                include_defense=True,
                clean_asr_threshold=threshold,
                is_cifar=is_cifar
            )
            db[model_key]["image_recon"][subkey] = summary

    # 3. feature_recon
    for root, _, files in os.walk(FEATURE_RECON_ROOT):
        if "accs.csv" not in files:
            continue
        cfg = load_config_from_folder(root)
        if not cfg:
            continue
        model_key = getattr(cfg, "MODEL", None)
        subkey = get_feature_recon_subkey(cfg)
        if not model_key or not subkey:
            continue
        run_data = load_run_from_csv(os.path.join(root, "accs.csv"), cfg)
        if run_data:
            run_data["folder"] = os.path.basename(root)
        if not run_data:
            continue
        print(f"[FOUND] feature_recon | {model_key:20} | {subkey:35} | {os.path.basename(root)}")
        feature_recon_runs_by_model.setdefault(model_key, {}).setdefault(subkey, []).append(run_data)

    for model_key, subgroups in feature_recon_runs_by_model.items():
        db.setdefault(model_key, {}).setdefault("feature_recon", {})
        is_cifar = "CIFAR10" in model_key
        threshold = _get_clean_threshold(model_key, clean_asr_max_cache)
        for subkey, runs in subgroups.items():
            summary = _compute_summary(
                runs,
                include_defense=True,
                clean_asr_threshold=threshold,
                is_cifar=is_cifar
            )
            db[model_key]["feature_recon"][subkey] = summary

    # 4. feature_codes
    for root, _, files in os.walk(FEATURE_CODES_ROOT):
        if "accs.csv" not in files:
            continue
        cfg = load_config_from_folder(root)
        if not cfg:
            continue
        model_key = getattr(cfg, "MODEL", None)
        subkey = get_feature_codes_subkey(cfg)
        if not model_key or not subkey:
            continue
        run_data = load_run_from_csv(os.path.join(root, "accs.csv"), cfg)
        if run_data:
            run_data["folder"] = os.path.basename(root)
        if not run_data:
            continue
        print(f"[FOUND] feature_codes | {model_key:20} | {subkey:35} | {os.path.basename(root)}")
        feature_codes_runs_by_model.setdefault(model_key, {}).setdefault(subkey, []).append(run_data)

    for model_key, subgroups in feature_codes_runs_by_model.items():
        db.setdefault(model_key, {}).setdefault("feature_codes", {})
        is_cifar = "CIFAR10" in model_key
        threshold = _get_clean_threshold(model_key, clean_asr_max_cache)
        for subkey, runs in subgroups.items():
            summary = _compute_summary(
                runs,
                include_defense=True,
                clean_asr_threshold=threshold,
                is_cifar=is_cifar
            )
            db[model_key]["feature_codes"][subkey] = summary

    _save_db(db)
    _print_run_table_expanded(
        clean_runs_by_model,
        defense_runs_by_model,
        image_recon_runs_by_model,
        feature_recon_runs_by_model,
        feature_codes_runs_by_model,
    )
    print(f"\n[SUCCESS] Baseline DB saved to {BASELINE_DB_PATH}")
    print(f" Models processed: {list(db.keys())}")
    _write_run_folders_txt(
        clean_runs_by_model,
        defense_runs_by_model,
        image_recon_runs_by_model,
        feature_recon_runs_by_model,
        feature_codes_runs_by_model,
    )

# --------------------------------------------------
# Getters
# --------------------------------------------------
def get_baseline(model_name: str = None, baseline_type: str = "clean", subkey: str = None) -> dict:
    db = _load_db()
    if model_name is None:
        try:
            import config
            model_name = config.MODEL
        except Exception:
            print("[WARN] Could not import config.MODEL")
            return {}
    if model_name not in db:
        print(f"[WARN] Model '{model_name}' not found.")
        return {}
    if baseline_type in ("image_recon", "feature_recon", "feature_codes"):
        recon_db = db[model_name].get(baseline_type, {})
        if subkey:
            return recon_db.get(subkey, {})
        return recon_db
    return db[model_name].get(baseline_type, {})

def get_baseline_value(model_name: str = None, baseline_type: str = "clean",
                       key: str = "val_acc_mean", subkey: str = None):
    data = get_baseline(model_name, baseline_type, subkey)
    return data.get(key)

if __name__ == "__main__":
    build_baseline_db()