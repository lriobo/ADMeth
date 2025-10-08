#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import re
import ast
import matplotlib.pyplot as plt
from pathlib import Path

def run(cfg):
    print("############   02-EVALUATION   ############")
    
    #model_dir = "/home/77462217B/lois/ADMeth/model/heavymodelv1"
    model_dir = Path(cfg["paths"]["admodel"])
    #datasets = ["/home/77462217B/lois/ADMeth/data/datasets/FraCas_float16.npy", ]
    #datasets = cfg["paths"]["raw"]
    raw_cfg = cfg["paths"]["raw"]
    groups = {
        "cases": raw_cfg.get("cases_files", []),
        "controls": raw_cfg.get("controls_files", [])
    }

    #output_base_dir = "/home/77462217B/lois/ADMeth/outcomes/griddatasetv2outcomes/"
    output_base_dir  = Path(cfg["paths"]["msemetrics"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info_path = os.path.join(model_dir, "model_info.txt")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"No se encontró model_info.txt en {model_dir}")
    
    with open(info_path, "r") as f:
        info_lines = [line.rstrip("\n") for line in f.readlines()]
    
    info_kv = {}
    for line in info_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            info_kv[key.strip()] = value.strip()
    
    if "Hidden neurons" in info_kv:
        hidden_neurons = int(re.search(r"\d+", info_kv["Hidden neurons"]).group())
    else:
        raise KeyError("Missing 'Hidden neurons' in model_info.txt")
    
    if "Latent dimensions" in info_kv:
        latent_dim = int(re.search(r"\d+", info_kv["Latent dimensions"]).group())
    else:
        raise KeyError("Missing 'Latent dim' in model_info.txt")
    
    if "BatchNorm" in info_kv:
        use_batchnorm = info_kv["BatchNorm"].strip().lower().startswith("t")
    else:
        use_batchnorm = True
    
    dropout_line = info_kv.get("Dropout", "False")
    m_bool = re.search(r"(True|False)", dropout_line, re.IGNORECASE)
    use_dropout = (m_bool.group(1).lower() == "true") if m_bool else False
    
    m_rate = re.search(r"rate\s*=\s*([0-9]*\.?[0-9]+)", dropout_line)
    dropout_rate = float(m_rate.group(1)) if m_rate else (0.0 if not use_dropout else 0.2)
    
    segment_size = 10000

    print(
        f"Configuration: Hidden={hidden_neurons}, Latent={latent_dim}, "
        f"Dropout={use_dropout} (rate={dropout_rate}), BatchNorm={use_batchnorm}, Segment size={segment_size}"
    )
    
    class SegmentAutoencoder(nn.Module):
        def __init__(self, input_size: int, hidden_neurons: int, latent_size: int,
                     use_dropout: bool, dropout_rate: float, use_batchnorm: bool):
            super().__init__()
            enc = [nn.Linear(input_size, hidden_neurons), nn.ReLU()]
            if use_batchnorm:
                enc.append(nn.BatchNorm1d(hidden_neurons))
            if use_dropout and dropout_rate > 0:
                enc.append(nn.Dropout(dropout_rate))
            enc += [nn.Linear(hidden_neurons, latent_size), nn.ReLU()]
            if use_batchnorm:
                enc.append(nn.BatchNorm1d(latent_size))
            if use_dropout and dropout_rate > 0:
                enc.append(nn.Dropout(dropout_rate))
    
            dec = [nn.Linear(latent_size, hidden_neurons), nn.ReLU()]
            if use_batchnorm:
                dec.append(nn.BatchNorm1d(hidden_neurons))
            if use_dropout and dropout_rate > 0:
                dec.append(nn.Dropout(dropout_rate))
            # Usa Identity() si tus datos no están en [0,1]
            dec += [nn.Linear(hidden_neurons, input_size), nn.Sigmoid()]
    
            self.encoder = nn.Sequential(*enc)
            self.decoder = nn.Sequential(*dec)
    
        def forward(self, x):
            return self.decoder(self.encoder(x))
       
    
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("autoencoder_segment_") and f.endswith(".pth")],
        key=lambda x: int(re.search(r"_(\d+)\.pth", x).group(1))
    )
    
    num_segments = len(model_files)
    print(f"{num_segments} segment models found")
   
    for group, file_list in groups.items():
        group_out_dir = os.path.join(str(output_base_dir), group)
        os.makedirs(group_out_dir, exist_ok=True)
    
        for dataset_path in file_list:
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
            dataset_output_dir = os.path.join(group_out_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
    
            print(f"\n→ Evaluating [{group}] dataset: {dataset_name}")
            X_data = np.load(dataset_path).astype(np.float16).T  # (n_samples, total_features)
            n_samples, total_features = X_data.shape
            expected_features = 320000
            assert total_features == expected_features, f"{dataset_path} has {total_features} columns, {expected_features} expected"
    
            mse_matrix = np.zeros((n_samples, total_features), dtype=np.float16)
            segment_mse_avgs = []
    
            for idx, model_file in enumerate(model_files):
                start = idx * segment_size
                end = start + segment_size
                X_segment = X_data[:, start:end]
    
                model = SegmentAutoencoder(
                    input_size=segment_size,
                    hidden_neurons=hidden_neurons,
                    latent_size=latent_dim,
                    use_dropout=use_dropout,
                    dropout_rate=dropout_rate,
                    use_batchnorm=use_batchnorm
                ).to(device)
    
                state = torch.load(os.path.join(model_dir, model_file), map_location=device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                    state = {k.replace("module.", "", 1): v for k, v in state.items()}
                model.load_state_dict(state, strict=True)
                model.eval()
    
                with torch.no_grad():
                    X_tensor = torch.tensor(X_segment, device=device, dtype=torch.float32)
                    preds = model(X_tensor).cpu().numpy().astype(np.float16)
    
                X_seg32 = X_segment.astype(np.float32)
                mse_per_position = (preds - X_seg32) ** 2
    
                missing_mask = (X_segment == 0)
                mse_per_position[missing_mask] = -1.0
    
                mse_matrix[:, start:end] = mse_per_position.astype(np.float16)
    
                valid_mask = ~missing_mask
                if np.any(valid_mask):
                    seg_mean = mse_per_position[valid_mask].mean()
                else:
                    seg_mean = np.nan
                segment_mse_avgs.append(seg_mean)
    
            missing_mask_all = (mse_matrix == -1)
            total_entries = mse_matrix.size
            num_missing = int(missing_mask_all.sum())
            missing_pct = 100.0 * num_missing / total_entries
    
            valid_mask_all = ~missing_mask_all
            if np.any(valid_mask_all):
                ae_valid = np.sqrt(mse_matrix[valid_mask_all].astype(np.float32))
                mae_global = float(ae_valid.mean())
                rmse_global = float(np.sqrt(mse_matrix[valid_mask_all].astype(np.float32).mean()))
                median_ae = float(np.median(ae_valid))
                p95_ae = float(np.percentile(ae_valid, 95))
                dataset_mse_global = float(mse_matrix[valid_mask_all].astype(np.float32).mean())
            else:
                mae_global = rmse_global = median_ae = p95_ae = np.nan
                dataset_mse_global = np.nan
    
            print(f"✅ [{group}] MSE global: {dataset_mse_global:.6f}")
            print(f"✅ MAE: {mae_global:.6f} | RMSE: {rmse_global:.6f} | Median AE: {median_ae:.6f} | P95 AE: {p95_ae:.6f}")
            print(f"✅ Missing: {missing_pct:.2f}%")
    
            max_points_for_plot = 2_000_000
            ae_for_plot = ae_valid
            if ae_for_plot.size > max_points_for_plot:
                idx = np.random.choice(ae_for_plot.size, size=max_points_for_plot, replace=False)
                ae_for_plot = ae_for_plot[idx]
    
            plot_path = os.path.join(dataset_output_dir, f"{dataset_name}_abs_error_hist.png")
            plt.figure()
            plt.hist(ae_for_plot, bins=100)
            plt.xlabel("Absolute Error")
            plt.ylabel("Count")
            plt.title(f"Abs Error Distribution – {dataset_name} [{group}]")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved plot: {plot_path}")
    
            mse_npy_path = os.path.join(dataset_output_dir, f"{dataset_name}_mse_per_sample_per_position.npy")
            np.save(mse_npy_path, mse_matrix)
            print(f"Saved: {mse_npy_path}")
    
            summary_row = {
                "Dataset": dataset_name,
                "Group": group,
                "Missing_Pct": missing_pct,
                "MSE_Global": dataset_mse_global,
                "MAE_Global": mae_global,
                "RMSE_Global": rmse_global,
                "Median_AE": median_ae,
                "P95_AE": p95_ae,
                **{f"MSE_Segment_{i+1}": segment_mse_avgs[i] for i in range(num_segments)}
            }
            summary_df = pd.DataFrame([summary_row])
            summary_csv_path = os.path.join(dataset_output_dir, f"{dataset_name}_mse_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Saved: {summary_csv_path}")
    
            info_copy_path = os.path.join(dataset_output_dir, "model_info.txt")
            with open(info_copy_path, "w") as f:
                f.writelines([line + ("\n" if not line.endswith("\n") else "") for line in info_lines])
            print(f"Saved: {info_copy_path}")
