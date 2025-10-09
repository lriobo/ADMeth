import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.base import clone
from joblib import Parallel, delayed
from utils_banner import print_banner


def run(cfg):
    project = str(cfg.get("run", {}).get("project", "default"))
    print_banner(step="mlmodels", project=project)
    print("############   03-MLMODELS (CV + bootstrap)   ############")
    run_cfg = cfg.get("run", {})
    project = str(run_cfg.get("project", "default"))


    # ========== 0) Ajustes de entorno / warnings ==========
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
    warnings.filterwarnings("ignore", message=".*LGBMClassifier.*")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # ========== 1) Leer rutas y parámetros desde config ==========
    mlcfg = cfg.get("mlmodels", {})
    paths = cfg["paths"]

    # Carpeta base donde están los resultados de recscores
    recscores_dir = (Path(cfg["paths"]["recscores"]).resolve() / project)
    cases_dir     = recscores_dir / "cases"
    controls_dir  = recscores_dir / "controls"

    if not cases_dir.exists() or not controls_dir.exists():
        raise FileNotFoundError(f"Controls and cases folders not found in: {recscores_dir}")

    # Buscar recursivamente los archivos .npy
    cases_files = sorted(cases_dir.rglob("*.npy"))
    controls_files = sorted(controls_dir.rglob("*.npy"))

    if not cases_files:
        raise RuntimeError(f"No .npy files found in: {cases_dir}")
    if not controls_files:
        raise RuntimeError(f"No .npy files found in: {controls_dir}")

    print(f"[mlmodels] Cases files found: {len(cases_files)}")
    print(f"[mlmodels] Controls files found: {len(controls_files)}")

    # Directorio de salida
    out_dir = (Path(cfg.get("mlmodels", {}).get("out_dir", "data/reports/mlmodels")).resolve() / project)
    out_dir.mkdir(parents=True, exist_ok=True)
    micro_path = out_dir / "AUC.csv"
    bootstrap_path = out_dir / "AUC_bootstrap.csv"
    top_features_path = out_dir / "FClassif_top20_features.csv"

    # Parámetros
    n_splits = int(mlcfg.get("n_splits", 5))
    random_state = int(mlcfg.get("random_state", 42))
    apply_feature_filter = bool(mlcfg.get("apply_feature_filter", False))
    feature_filter_threshold = float(mlcfg.get("feature_filter_threshold", 0.01))
    max_features = int(mlcfg.get("max_features", 500))
    n_bootstrap = int(mlcfg.get("n_bootstrap", 1000))
    n_jobs = int(mlcfg.get("n_jobs", 4))
    k_grid = list(range(1, max_features + 1, 1))

    # ========== 2) Cargar y concatenar todos los .npy ==========
    def load_and_concat(npy_list):
        arrays = []
        for f in npy_list:
            arr = np.load(f).astype(np.float32, copy=False)
            if arr.ndim != 2:
                raise ValueError(f"File {f} is not a 2D matrix.")
            arrays.append(arr)
        return np.vstack(arrays)

    print("[mlmodels] Cargando casos y controles...")
    Cases_arr = load_and_concat(cases_files)
    Controls_arr = load_and_concat(controls_files)

    if Cases_arr.shape[1] != Controls_arr.shape[1]:
        raise ValueError(f"Number of features in cases ({Cases_arr.shape[1]}) and controls ({Controls_arr.shape[1]}) is not the same.")

    n_features = Cases_arr.shape[1]
    shared_cols = np.array([f"f{i}" for i in range(n_features)], dtype=object)

    X = np.vstack([Cases_arr, Controls_arr]).astype(np.float32, copy=False)
    y = np.hstack([
        np.ones(Cases_arr.shape[0], dtype=np.int8),
        np.zeros(Controls_arr.shape[0], dtype=np.int8),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ========== 3) Clasificadores ==========
    base_classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
    }
    try:
        from xgboost import XGBClassifier
        base_classifiers["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state, n_jobs=1
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        base_classifiers["LightGBM"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, verbose=-1, n_jobs=1
        )
    except Exception:
        pass

    # ========== 4) Filtro opcional (>= +10% en casos) ==========
    def apply_feature_threshold_filter(Xtr, ytr, thr=0.10):
        cases = Xtr[ytr == 1]
        controls = Xtr[ytr == 0]
        mean_cases = cases.mean(axis=0)
        mean_controls = controls.mean(axis=0)
        mask = mean_cases >= (1.0 + thr) * np.maximum(mean_controls, 1e-12)
        return mask

    # ========== 5) Evaluar un modelo dado para top-k ==========
    def eval_model_on_topk(model_name, model_proto, Xtr, ytr, Xte, yte, ranked_idx, k):
        idx = ranked_idx[:k]
        Xtr_k = Xtr[:, idx]
        Xte_k = Xte[:, idx]
        model = clone(model_proto)
        model.fit(Xtr_k, ytr)
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(Xte_k)
        elif hasattr(model, "predict_proba"):
            y_score = model.predict_proba(Xte_k)[:, 1]
        else:
            y_score = model.predict(Xte_k)
        auc = roc_auc_score(yte, y_score)
        return model_name, k, auc

    # ========== 6) Precalcular folds y ranking f_classif ==========
    folds_data = []
    rng = np.random.default_rng(seed=random_state)

    for p in [micro_path, bootstrap_path, top_features_path]:
        if Path(p).exists():
            Path(p).unlink()

    pd.DataFrame(columns=["fold", "rank", "feature_name", "feature_index"]).to_csv(
        top_features_path, index=False
    )

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        feature_names = np.array(shared_cols, dtype=object)

        if apply_feature_filter:
            mask = apply_feature_threshold_filter(Xtr, ytr, feature_filter_threshold)
            Xtr = Xtr[:, mask]
            Xte = Xte[:, mask]
            feature_names = feature_names[mask]

        F, _ = f_classif(Xtr, ytr)
        F = np.nan_to_num(F, nan=-np.inf)
        ranked_idx = np.argsort(F)[::-1]

        top20_idx = ranked_idx[:20]
        top20_names = feature_names[top20_idx]
        tf_rows = [
            {"fold": fold, "rank": rnk, "feature_name": str(fname), "feature_index": int(findex)}
            for rnk, (fname, findex) in enumerate(zip(top20_names, top20_idx), start=1)
        ]
        pd.DataFrame(tf_rows).to_csv(top_features_path, mode="a", header=False, index=False)
        folds_data.append((fold, Xtr, ytr, Xte, yte, ranked_idx))

    # ========== 7) Bootstrap helper ==========
    def bootstrap_mean_ci(values, n_resamples=1000, alpha=0.05, rng=None):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.nan, np.nan, np.nan
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
        samples = values[idx].mean(axis=1)
        mean_boot = samples.mean()
        lo = np.quantile(samples, alpha / 2)
        hi = np.quantile(samples, 1 - alpha / 2)
        return float(mean_boot), float(lo), float(hi)
    print(f"[mlmodels] Training...")
    # ========== 8) Bucle principal sobre k ==========
    all_models = list(base_classifiers.keys())
    cols_order = ["k"] + all_models
    header_written = False
    pd.DataFrame(columns=["k", "model", "mean_auc_boot", "ci95_low", "ci95_high"]).to_csv(
        bootstrap_path, index=False
    )

    for k in k_grid:
        tasks = (
            (name, model, Xtr, ytr, Xte, yte, ranked_idx, k)
            for (fold, Xtr, ytr, Xte, yte, ranked_idx) in folds_data
            for name, model in base_classifiers.items()
        )

        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(eval_model_on_topk)(name, model, Xtr, ytr, Xte, yte, ranked_idx, k)
            for (name, model, Xtr, ytr, Xte, yte, ranked_idx, k) in tasks
        )

        model_to_scores = {m: [] for m in all_models}
        for name, _k, auc in results:
            model_to_scores[name].append(auc)

        row = {"k": int(k)}
        for m in all_models:
            row[m] = float(np.mean(model_to_scores[m])) if model_to_scores[m] else np.nan
        pd.DataFrame([row], columns=cols_order).to_csv(
            micro_path, mode="a", header=not header_written, index=False
        )
        header_written = True

        bs_rows = []
        for m in all_models:
            mean_boot, lo, hi = bootstrap_mean_ci(model_to_scores[m], n_resamples=n_bootstrap, alpha=0.05, rng=rng)
            bs_rows.append({"k": int(k), "model": m, "mean_auc_boot": mean_boot, "ci95_low": lo, "ci95_high": hi})
        pd.DataFrame(bs_rows).to_csv(bootstrap_path, mode="a", header=False, index=False)

    print(f"✅ [mlmodels] Results per-k: {micro_path}")
    print(f"✅ [mlmodels] Bootstrap summaries: {bootstrap_path}")
    print(f"✅ [mlmodels] Top-20 features per fold: {top_features_path}")
