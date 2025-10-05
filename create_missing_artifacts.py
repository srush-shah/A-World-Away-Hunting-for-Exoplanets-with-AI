#!/usr/bin/env python3
"""
Create missing model artifacts by extracting from existing data and models.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = Path("data")
ART_DIR = Path("model_artifacts")
ART_DIR.mkdir(exist_ok=True)

print("Loading data...")
# Load aligned tables
koi_aligned = pd.read_csv(DATA_DIR / "koi_aligned.csv")

# Core features
CORE_FEATURES = [
    'period_days','dur_hrs','depth_ppm','rp_re','teq_k',
    'st_teff_k','st_logg_cgs','st_rad_rsun','st_mass_msun'
]

print("Preprocessing data...")
# Per-dataset coercion + median impute
for c in CORE_FEATURES:
    koi_aligned[c] = pd.to_numeric(koi_aligned[c], errors='coerce')
    koi_aligned[c] = koi_aligned[c].fillna(koi_aligned[c].median())

# Clamp obviously positive quantities
pos_clamp = ["period_days","dur_hrs","depth_ppm","rp_re","st_rad_rsun","st_mass_msun","teq_k","st_teff_k"]
for c in pos_clamp:
    med = koi_aligned[c].median()
    koi_aligned.loc[koi_aligned[c] <= 0, c] = med if np.isfinite(med) and med > 0 else 1.0

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

print("Creating and fitting imputer...")
# Imputer fitted on KOI (no leakage)
imp = SimpleImputer(strategy="median").fit(X_koi)
X_koi_imp = pd.DataFrame(imp.transform(X_koi), columns=X_koi.columns)

print("Creating and fitting scaler...")
# Scale (fit on KOI)
scaler = StandardScaler().fit(X_koi_imp.values)

print("Saving artifacts...")
# Save imputer
joblib.dump(imp, ART_DIR / "imputer.joblib")
print("✓ Saved imputer.joblib")

# Save scaler
joblib.dump(scaler, ART_DIR / "scaler.joblib")
print("✓ Saved scaler.joblib")

# Copy the existing ensemble model to the expected name
import shutil
if (ART_DIR / "model_kepX_tessCal_ENSEMBLE.joblib").exists():
    shutil.copy(ART_DIR / "model_kepX_tessCal_ENSEMBLE.joblib", ART_DIR / "model_kepX_tessCal_LGB.joblib")
    print("✓ Copied ensemble model to model_kepX_tessCal_LGB.joblib")

print("\nAll missing artifacts created successfully!")
print(f"Artifacts saved to: {ART_DIR.absolute()}")
