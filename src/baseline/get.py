import json

input_file = "/workspace/src/baseline/baseline_db.json"
output_file = "/workspace/src/baseline/summary.txt"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for model, model_data in data.items():
        for setting, setting_data in model_data.items():
            if setting in ["clean", "defense"]:
                subconfig = "-"
                stats = setting_data
            elif setting == "image_recon":
                for subconfig, stats in setting_data.items():
                    pass
            else:
                continue

            if setting == "image_recon":
                configs = setting_data.items()
            else:
                configs = [("-", setting_data)]

            for subconfig, stats in configs:
                def fmt(val, decimals=4):
                    if val is None or isinstance(val, str):
                        return str(val)
                    return f"{val:.{decimals}f}"

                line = f"{model} | {setting} | {subconfig} | "
                line += f"val_acc: {fmt(stats.get('val_acc_mean'))} | "
                line += f"asr_at_opt: {fmt(stats.get('asr_at_opt_mean'))} | "
                line += f"asr_mean: {fmt(stats.get('asr_mean'))} | "
                line += f"iasr: {fmt(stats.get('asr_auc_post_opt_mean'), 2)} | "
                line += f"recovery_step: {fmt(stats.get('defense_step_mean'), 1)}\n"
                
                f.write(line)

print(f"Summary successfully written to {output_file}")