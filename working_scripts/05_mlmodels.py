#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from joblib import Parallel, delayed
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
warnings.filterwarnings("ignore", message=".*LGBMClassifier.*")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# In[15]:


# ====== Load and basic cleaning ======
Cases_clean = pd.read_csv("/home/77462217B/lois/AAImageneAnomalyDetection/data/evaluatedatasets/BetasMicCas.csv")
Controls_clean = pd.read_csv("/home/77462217B/lois/AAImageneAnomalyDetection/data/evaluatedatasets/BetasMicCon.csv")

# Ensure both datasets share the same columns and order
shared_cols = [c for c in Cases_clean.columns if c in Controls_clean.columns]
Cases_clean = Cases_clean[shared_cols].copy()
Controls_clean = Controls_clean[shared_cols].copy()

micro_path = "/home/77462217B/lois/AAImageneAnomalyDetection/results/betas/crossvalidation/MicCas.csv"
bootstrap_path = micro_path.replace(".csv", "_bootstrap.csv")  
top_features_path = micro_path.replace(".csv", "_top20_features.csv")  

# ====== Parameters ======
n_splits = 5
random_state = 42
apply_feature_filter = False   
feature_filter_threshold = 0.01
max_features = 500             
k_grid = list(range(1, max_features + 1, 1)) 
n_bootstrap = 1000  

# ====== Combined dataset and CV ======
X = np.vstack([Cases_clean.values, Controls_clean.values]).astype(np.float32, copy=False)
y = np.hstack([np.ones(Cases_clean.shape[0], dtype=np.int8),
               np.zeros(Controls_clean.shape[0], dtype=np.int8)])

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# ====== Classifiers ======
base_classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
    #'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=random_state),
    #'SVM_linear': SVC(kernel='linear', probability=False, random_state=random_state),  
    #'KNN': KNeighborsClassifier(n_neighbors=5),  
    #'NaiveBayes': GaussianNB(),
    'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=random_state),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
}

try:
    from xgboost import XGBClassifier
    base_classifiers['XGBoost'] = XGBClassifier(
        n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        eval_metric='logloss', random_state=random_state, n_jobs=1
    )
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    base_classifiers['LightGBM'] = LGBMClassifier(
        n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        random_state=random_state, verbose=-1, n_jobs=1
    )
except Exception:
    pass

# ====== Optional +10% filter in cases ======
def apply_feature_threshold_filter(Xtr, ytr, thr=0.10):
    # class-wise means in training
    cases = Xtr[ytr == 1]
    controls = Xtr[ytr == 0]
    mean_cases = cases.mean(axis=0)
    mean_controls = controls.mean(axis=0)
    mask = mean_cases >= (1.0 + thr) * np.maximum(mean_controls, 1e-12)
    return mask

# ====== Evaluate a given model (with top-k already selected) ======
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
        y_score = model.predict(Xte_k)  # fallback
    auc = roc_auc_score(yte, y_score)
    return model_name, k, auc

# ====== Precompute folds and f_classif ranking (once) ======
folds_data = []
rng = np.random.default_rng(seed=random_state)

# prepare folders/files
os.makedirs(os.path.dirname(micro_path), exist_ok=True)
for p in [micro_path, bootstrap_path, top_features_path]:
    if os.path.exists(p):
        os.remove(p)

# save header for top-20 features file
pd.DataFrame(columns=["fold", "rank", "feature_name", "feature_index"]).to_csv(
    top_features_path, index=False
)

for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    # we keep track of feature names (aligned to shared_cols and any mask later)
    feature_names = np.array(shared_cols, dtype=object)

    # Optional filter
    if apply_feature_filter:
        mask = apply_feature_threshold_filter(Xtr, ytr, feature_filter_threshold)
        Xtr = Xtr[:, mask]
        Xte = Xte[:, mask]
        feature_names = feature_names[mask]

    # f_classif once per fold
    F, _ = f_classif(Xtr, ytr)
    F = np.nan_to_num(F, nan=-np.inf)
    ranked_idx = np.argsort(F)[::-1]

    # NEW: save top-20 features for this fold
    top20_idx = ranked_idx[:20]
    top20_names = feature_names[top20_idx]
    tf_rows = []
    for rnk, (fname, findex) in enumerate(zip(top20_names, top20_idx), start=1):
        tf_rows.append({"fold": fold, "rank": rnk, "feature_name": str(fname), "feature_index": int(findex)})
    pd.DataFrame(tf_rows).to_csv(top_features_path, mode="a", header=False, index=False)

    folds_data.append((fold, Xtr, ytr, Xte, yte, ranked_idx))

# ====== Helper: bootstrap mean and 95% CI from list of fold AUCs ======
def bootstrap_mean_ci(values, n_resamples=1000, alpha=0.05, rng=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[idx].mean(axis=1)
    mean_boot = samples.mean()
    lo = np.quantile(samples, alpha/2)
    hi = np.quantile(samples, 1 - alpha/2)
    return float(mean_boot), float(lo), float(hi)

# ====== Loop over k: train, aggregate by model, and save incrementally ======
all_models = list(base_classifiers.keys())
cols_order = ["k"] + all_models
header_written = False

# bootstrap output header
pd.DataFrame(columns=["k", "model", "mean_auc_boot", "ci95_low", "ci95_high"]).to_csv(
    bootstrap_path, index=False
)

for k in k_grid:
    # Evaluate (fold × model) in parallel for this k
    tasks = (
        (name, model, Xtr, ytr, Xte, yte, ranked_idx, k)
        for (fold, Xtr, ytr, Xte, yte, ranked_idx) in folds_data
        for name, model in base_classifiers.items()
    )

    results = Parallel(n_jobs=40, backend="loky", verbose=0)(
        delayed(eval_model_on_topk)(name, model, Xtr, ytr, Xte, yte, ranked_idx, k)
        for (name, model, Xtr, ytr, Xte, yte, ranked_idx, k) in tasks
    )

    # Aggregate over folds and write ONE row per k (columns = models)
    model_to_scores = {m: [] for m in all_models}
    for name, _k, auc in results:
        model_to_scores[name].append(auc)

    row = {"k": int(k)}
    for m in all_models:
        # keep the original behavior: mean over folds in main CSV
        row[m] = float(np.mean(model_to_scores[m])) if model_to_scores[m] else np.nan
    pd.DataFrame([row], columns=cols_order).to_csv(
        micro_path, mode="a", header=not header_written, index=False
    )
    header_written = True

    # NEW: also write bootstrap mean and 95% CI per model to a separate CSV
    bs_rows = []
    for m in all_models:
        mean_boot, lo, hi = bootstrap_mean_ci(model_to_scores[m], n_resamples=n_bootstrap, alpha=0.05, rng=rng)
        bs_rows.append({
            "k": int(k),
            "model": m,
            "mean_auc_boot": mean_boot,
            "ci95_low": lo,
            "ci95_high": hi
        })
    pd.DataFrame(bs_rows).to_csv(bootstrap_path, mode="a", header=False, index=False)

print(f"All done! Incremental per-k results saved to: {micro_path}")
print(f"Bootstrap summaries saved to: {bootstrap_path}")
print(f"Top-20 features per fold saved to: {top_features_path}")


# In[10]:


# ============================== USER PARAMETERS =======================================
# Paths: training and external validation, with Cases and Controls separated
train_cases_path    = "/home/77462217B/lois/AAImageneAnomalyDetection/outcomes/MRS/heavymodelv1/All/StdPlus1ScoreMicCasMSE.csv"
train_controls_path = "/home/77462217B/lois/AAImageneAnomalyDetection/outcomes/MRS/heavymodelv1/All/StdPlus1ScoreMicConMSE.csv"

valid_cases_path = "/home/77462217B/lois/AAImageneAnomalyDetection/outcomes/MRS/heavymodelv1/All/StdPlus1ScoreFraCasMSE.csv"
valid_controls_path = "/home/77462217B/lois/AAImageneAnomalyDetection/outcomes/MRS/heavymodelv1/All/StdPlus1ScoreFraConMSE.csv"


# In[14]:


# Choose the model and number of features (k)
selected_model_name = "HistGradientBoosting"   
k_selected = 346                       

# Optional: +10% filter in cases 
apply_feature_filter = False
feature_filter_threshold = 0.10

# Bootstrapping parameters
n_bootstrap = 1000
alpha = 0.05
random_state = 42

# Outputs
results_dir = "/home/77462217B/lois/AAImageneAnomalyDetection/results/heavymodelv1/EVFramingham"
os.makedirs(results_dir, exist_ok=True)
micro_path = os.path.join(results_dir, f"external_{selected_model_name}_k{k_selected}.csv")
bootstrap_path = micro_path.replace(".csv", "_bootstrap.csv")
top_features_path = micro_path.replace(".csv", "_top20_features.csv")


# In[12]:


# ============================== Load data  ==============================
print("Loading training and validation datasets (cases/controls kept separate)...")
Cases_train = pd.read_csv(train_cases_path)
Controls_train = pd.read_csv(train_controls_path)
Cases_valid = pd.read_csv(valid_cases_path)
Controls_valid = pd.read_csv(valid_controls_path)

X_train = np.vstack([Cases_train.values, Controls_train.values]).astype(np.float32, copy=False)
y_train = np.hstack([
    np.ones(Cases_train.shape[0], dtype=np.int8),
    np.zeros(Controls_train.shape[0], dtype=np.int8)
])

X_valid = np.vstack([Cases_valid.values, Controls_valid.values]).astype(np.float32, copy=False)
y_valid = np.hstack([
    np.ones(Cases_valid.shape[0], dtype=np.int8),
    np.zeros(Controls_valid.shape[0], dtype=np.int8)
])


# In[15]:


# Basic sanity checks without aligning anything
if X_train.shape[1] != X_valid.shape[1]:
    raise ValueError("Train and validation must have the SAME number of features (no alignment performed).")

feature_names = np.array(Cases_train.columns, dtype=object)  # for reporting top-20

print(f"Train: {X_train.shape[0]} samples "
      f"(cases={Cases_train.shape[0]}, controls={Controls_train.shape[0]}), "
      f"Validation: {X_valid.shape[0]} samples "
      f"(cases={Cases_valid.shape[0]}, controls={Controls_valid.shape[0]}), "
      f"Features: {X_train.shape[1]}")

# ============================== Model zoo  ==========================
base_classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
    'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=random_state),
    'SVM_linear': SVC(kernel='linear', probability=False, random_state=random_state),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'NaiveBayes': GaussianNB(),
    'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=random_state),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
}
try:
    from xgboost import XGBClassifier
    base_classifiers['XGBoost'] = XGBClassifier(
        n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        eval_metric='logloss', random_state=random_state, n_jobs=1
    )
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    base_classifiers['LightGBM'] = LGBMClassifier(
        n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        random_state=random_state, verbose=-1, n_jobs=1
    )
except Exception:
    pass

if selected_model_name not in base_classifiers:
    raise ValueError(f"Unknown model '{selected_model_name}'. Available: {list(base_classifiers.keys())}")

# ============================== Optional +10% filter (train only) ======================
def apply_feature_threshold_filter(Xtr, ytr, thr=0.10):
    # Class-wise means in training
    cases = Xtr[ytr == 1]
    controls = Xtr[ytr == 0]
    mean_cases = cases.mean(axis=0)
    mean_controls = controls.mean(axis=0)
    mask = mean_cases >= (1.0 + thr) * np.maximum(mean_controls, 1e-12)
    return mask

if apply_feature_filter:
    print("Applying +10% filter on training features (cases vs controls)...")
    mask = apply_feature_threshold_filter(X_train, y_train, feature_filter_threshold)
    X_train = X_train[:, mask]
    X_valid = X_valid[:, mask]  
    feature_names = feature_names[mask]
    print(f"Remaining features after filter: {X_train.shape[1]}")

# ============================== Rank features on TRAIN (f_classif) ====================
F, _ = f_classif(X_train, y_train)
F = np.nan_to_num(F, nan=-np.inf)
ranked_idx = np.argsort(F)[::-1]

# Save top-20 features
top20_idx = ranked_idx[:20]
top20_names = feature_names[top20_idx]
pd.DataFrame({
    "rank": np.arange(1, len(top20_idx) + 1),
    "feature_name": top20_names,
    "feature_index": top20_idx
}).to_csv(top_features_path, index=False)
print(f"Saved top-20 features to: {top_features_path}")

# ============================== Select top-k & train model ============================
k_eff = int(min(k_selected, X_train.shape[1]))
idx_k = ranked_idx[:k_eff]
Xtr_k = X_train[:, idx_k]
Xva_k = X_valid[:, idx_k]

model = clone(base_classifiers[selected_model_name])
print(f"Training model: {selected_model_name} with k={k_eff} features...")
model.fit(Xtr_k, y_train)

# ============================== Validate on external set ==============================
if hasattr(model, "decision_function"):
    y_score = model.decision_function(Xva_k)
elif hasattr(model, "predict_proba"):
    y_score = model.predict_proba(Xva_k)[:, 1]
else:
    y_score = model.predict(Xva_k)  

auc_val = roc_auc_score(y_valid, y_score)
print(f"External validation AUC: {auc_val:.4f}")

# Save single-shot result
pd.DataFrame([{
    "model": selected_model_name,
    "k": k_eff,
    "auc_external": float(auc_val),
    "n_train_cases": int(Cases_train.shape[0]),
    "n_train_controls": int(Controls_train.shape[0]),
    "n_valid_cases": int(Cases_valid.shape[0]),
    "n_valid_controls": int(Controls_valid.shape[0]),
    "n_features_total": int(feature_names.size)
}]).to_csv(micro_path, index=False)
print(f"Saved external validation result to: {micro_path}")

# ============================== Bootstrapping on validation ===========================
rng = np.random.default_rng(seed=random_state)

def bootstrap_auc(y_true, y_score, n_resamples=1000, alpha=0.05, rng=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_resamples, n))  # resample with replacement
    aucs = []
    for row in idx:
        try:
            aucs.append(roc_auc_score(y_true[row], y_score[row]))
        except ValueError:
            # Skip samples with only one class
            continue
    aucs = np.asarray(aucs, dtype=float)
    mean_boot = float(np.mean(aucs)) if aucs.size else np.nan
    lo = float(np.quantile(aucs, alpha/2)) if aucs.size else np.nan
    hi = float(np.quantile(aucs, 1 - alpha/2)) if aucs.size else np.nan
    return mean_boot, lo, hi

print("Bootstrapping external validation AUC...")
mean_boot, ci_lo, ci_hi = bootstrap_auc(y_valid, y_score, n_resamples=n_bootstrap, alpha=alpha, rng=rng)

pd.DataFrame([{
    "model": selected_model_name,
    "k": k_eff,
    "auc_external": float(auc_val),
    "mean_auc_boot": float(mean_boot),
    "ci95_low": float(ci_lo),
    "ci95_high": float(ci_hi),
    "n_bootstrap": int(n_bootstrap)
}]).to_csv(bootstrap_path, index=False)

print(f"Bootstrap summary saved to: {bootstrap_path}")
print("Done.")


# In[ ]:




