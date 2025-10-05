#!/usr/bin/env python3
"""
Script to create missing model artifacts from the trained models in the notebook.
This extracts the imputer, scaler, and model from the notebook execution and saves them.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import xgboost as xgb
from lightgbm import LGBMClassifier

# Paths
DATA_DIR = Path("data")
ART_DIR = Path("model_artifacts")
ART_DIR.mkdir(exist_ok=True)

print("Loading data...")
# Load aligned tables
koi_aligned = pd.read_csv(DATA_DIR / "koi_aligned.csv")
toi_aligned = pd.read_csv(DATA_DIR / "toi_aligned.csv")

# Core features
CORE_FEATURES = [
    'period_days','dur_hrs','depth_ppm','rp_re','teq_k',
    'st_teff_k','st_logg_cgs','st_rad_rsun','st_mass_msun'
]

print("Preprocessing data...")
# If a TOI column is entirely missing, seed from KOI median
for c in CORE_FEATURES:
    if c not in toi_aligned.columns:
        toi_aligned[c] = np.nan
    if toi_aligned[c].isna().all():
        toi_aligned[c] = koi_aligned[c].median()

# Per-dataset coercion + median impute
for df in (koi_aligned, toi_aligned):
    for c in CORE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

# Clamp obviously positive quantities
pos_clamp = ["period_days","dur_hrs","depth_ppm","rp_re","st_rad_rsun","st_mass_msun","teq_k","st_teff_k"]
for df in (koi_aligned, toi_aligned):
    for c in pos_clamp:
        med = df[c].median()
        df.loc[df[c] <= 0, c] = med if np.isfinite(med) and med > 0 else 1.0

print("Building features...")
# Physics-aware features
def build_features(df):
    X = df[CORE_FEATURES].copy()
    X["depth_over_dur"] = X["depth_ppm"] / (X["dur_hrs"] + 1e-9)
    X["rp_over_star"]   = X["rp_re"] / (X["st_rad_rsun"]*109.2 + 1e-9)
    X["log_period"]     = np.log1p(X["period_days"])
    X["log_depth"]      = np.log1p(X["depth_ppm"])
    X["duty_cycle"]     = X["dur_hrs"]/(24.0*X["period_days"] + 1e-9)
    X["star_rad_mass_ratio"] = X["st_rad_rsun"]/(X["st_mass_msun"] + 1e-9)
    X["rp_est_from_depth"]   = (X["depth_ppm"]/1e6)**0.5 * 109.2 * X["st_rad_rsun"]
    y = df["label"].astype(int).values
    return X.replace([np.inf,-np.inf], np.nan), y

X_koi, y_koi = build_features(koi_aligned)
X_toi, y_toi = build_features(toi_aligned)

print("Splitting data...")
# Split TESS = 20% val / 80% test
X_val, X_test, y_val, y_test = train_test_split(
    X_toi, y_toi, test_size=0.80, random_state=42, stratify=y_toi
)

print("Creating and fitting imputer...")
# Imputer fitted on KOI (no leakage)
imp = SimpleImputer(strategy="median").fit(X_koi)
X_koi_imp  = pd.DataFrame(imp.transform(X_koi),  columns=X_koi.columns)
X_val_imp  = pd.DataFrame(imp.transform(X_val),  columns=X_koi.columns)
X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_koi.columns)

print("Rebalancing data...")
# REBALANCE KOI with resample (upsample minority)
koi_imp_wy = X_koi_imp.copy()
koi_imp_wy["label"] = y_koi

n_pos = int((koi_imp_wy["label"]==1).sum())
n_neg = int((koi_imp_wy["label"]==0).sum())
print(f"[KOI before resample] pos={n_pos}, neg={n_neg}")

df_pos = koi_imp_wy[koi_imp_wy["label"]==1]
df_neg = koi_imp_wy[koi_imp_wy["label"]==0]

# Choose strategy: upsample minority to match majority
minority = df_pos if len(df_pos) < len(df_neg) else df_neg
majority = df_neg if len(df_neg) > len(df_pos) else df_pos
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)

koi_bal = pd.concat([majority, minority_up], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
X_koi_bal = koi_bal.drop(columns=["label"]).values
y_koi_bal = koi_bal["label"].values

print(f"[KOI after resample]  pos={int((y_koi_bal==1).sum())}, neg={int((y_koi_bal==0).sum())}")

print("Creating and fitting scaler...")
# Scale (fit on balanced KOI)
scaler = StandardScaler().fit(X_koi_bal)
Xs_koi  = scaler.transform(X_koi_bal)
Xs_val  = scaler.transform(X_val_imp)
Xs_test = scaler.transform(X_test_imp)

print("Training models...")
# Models (turn OFF class weights now that KOI is balanced)
xgb_base = xgb.XGBClassifier(
    n_estimators=700, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=1.0,  # balanced
    objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=4
)
lgb_base = LGBMClassifier(
    n_estimators=1200, num_leaves=63, learning_rate=0.03,
    subsample=0.9, colsample_bytree=0.9,
    class_weight=None,  # balanced
    random_state=42, n_jobs=-1
)
log_base = LogisticRegression(max_iter=5000, class_weight=None, n_jobs=-1)

# Fit on KOI (balanced, imputed, scaled)
xgb_base.fit(Xs_koi, y_koi_bal)
lgb_base.fit(Xs_koi, y_koi_bal)
log_base.fit(Xs_koi, y_koi_bal)

print("Calibrating models...")
# Calibrate on TESS-val
xgb_cal = CalibratedClassifierCV(xgb_base, method="sigmoid", cv="prefit").fit(Xs_val, y_val)
lgb_cal = CalibratedClassifierCV(lgb_base, method="sigmoid", cv="prefit").fit(Xs_val, y_val)
log_cal = CalibratedClassifierCV(log_base, method="sigmoid", cv="prefit").fit(Xs_val, y_val)

print("Saving artifacts...")
# Save imputer
joblib.dump(imp, ART_DIR / "imputer.joblib")
print("✓ Saved imputer.joblib")

# Save scaler
joblib.dump(scaler, ART_DIR / "scaler.joblib")
print("✓ Saved scaler.joblib")

# Save individual models
joblib.dump(xgb_cal, ART_DIR / "model_kepX_tessCal_XGB.joblib")
joblib.dump(lgb_cal, ART_DIR / "model_kepX_tessCal_LGB.joblib")
joblib.dump(log_cal, ART_DIR / "model_kepX_tessCal_LOG.joblib")
print("✓ Saved individual calibrated models")

# Create ensemble model (weighted average)
class WeightedAverageEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1.0/len(models)] * len(models)
    
    def predict_proba(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred

# Create ensemble with equal weights
ensemble = WeightedAverageEnsemble([xgb_cal, lgb_cal, log_cal])
joblib.dump(ensemble, ART_DIR / "model_kepX_tessCal_ENSEMBLE.joblib")
print("✓ Saved ensemble model")

# Update metadata
metadata = {
    "model_type": "WeightedAverageEnsemble",
    "base_models": ["XGBoost", "LightGBM", "LogisticRegression"],
    "training_data": "Kepler + TESS",
    "features": 16,
    "created": "2024-12-19",
    "description": "Ensemble model for exoplanet classification combining XGBoost, LightGBM, and Logistic Regression with calibration"
}

with open(ART_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("✓ Updated metadata.json")

print("\nAll artifacts created successfully!")
print(f"Artifacts saved to: {ART_DIR.absolute()}")
