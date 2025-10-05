# api/deploy_api.py
from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd, io, json, os

ART = "model_artifacts"
IMP     = joblib.load(f"{ART}/imputer.joblib")
SCALER  = joblib.load(f"{ART}/scaler.joblib")
MODEL   = joblib.load(f"{ART}/model_kepX_tessCal_LGB.joblib")
CONF    = json.load(open(f"{ART}/thresholds.json"))
FEATURE_ORDER = CONF["feature_order"]

REQUIRED = ['period_days','dur_hrs','depth_ppm','rp_re','teq_k',
            'st_teff_k','st_logg_cgs','st_rad_rsun','st_mass_msun']

class Req(BaseModel):
    period_days: float; dur_hrs: float; depth_ppm: float; rp_re: float; teq_k: float
    st_teff_k: float; st_logg_cgs: float; st_rad_rsun: float; st_mass_msun: float

def build_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[REQUIRED].copy()
    # physics-aware derived (must match notebook)
    X["depth_over_dur"]      = X["depth_ppm"]/(X["dur_hrs"]+1e-9)
    X["rp_over_star"]        = X["rp_re"]/(X["st_rad_rsun"]*109.2 + 1e-9)
    X["log_period"]          = np.log1p(X["period_days"])
    X["log_depth"]           = np.log1p(X["depth_ppm"])
    X["duty_cycle"]          = X["dur_hrs"]/(24.0*X["period_days"] + 1e-9)
    X["star_rad_mass_ratio"] = X["st_rad_rsun"]/(X["st_mass_msun"] + 1e-9)
    X["rp_est_from_depth"]   = np.sqrt(X["depth_ppm"]/1e6) * 109.2 * X["st_rad_rsun"]
    # align to training order
    return X.reindex(columns=FEATURE_ORDER)

def transform_for_model(X: pd.DataFrame) -> np.ndarray:
    Xi = IMP.transform(X)                  # impute like training
    Xs = SCALER.transform(Xi)              # scale like training
    return Xs

def categorize(p, feats, tau_lo, tau_hi):
    if feats["rp_est_from_depth"] > 20.0:
        return "FALSE_POSITIVE", "VETO: rp_est_from_depth > 20 R_earth"
    if feats["duty_cycle"] > 0.20:
        return "FALSE_POSITIVE", f"VETO: duty_cycle={feats['duty_cycle']:.2f}"
    if p >= tau_hi: return "CONFIRMED_ALGO", ""
    if p >= tau_lo: return "CANDIDATE", ""
    return "FALSE_POSITIVE", ""

app = FastAPI(title="ExoEdge API (LGB-calibrated)")

@app.get("/")
def root():
    return {
        "message": "ExoEdge API - Exoplanet Detection Service",
        "version": "1.0.0",
        "endpoints": {
            "/version": "Get model metadata and thresholds",
            "/predict": "Single exoplanet prediction (POST)",
            "/predict-batch": "Batch exoplanet prediction (POST)"
        },
        "required_fields": REQUIRED,
        "docs": "/docs"
    }

@app.get("/version")
def version():
    meta = json.load(open(f"{ART}/metadata.json"))
    return {"meta": meta, "thresholds": CONF}

@app.post("/predict")
def predict(req: Req,
            tau_lo: float = Query(CONF["thr_best_macroF1"]),
            tau_hi: float = Query(CONF["thr_target_fpr_10"])):
    df = pd.DataFrame([{k:getattr(req,k) for k in REQUIRED}])
    X   = build_features_frame(df)
    Xs  = transform_for_model(X)
    p   = float(MODEL.predict_proba(Xs)[:,1][0])
    feats = X.iloc[0].to_dict()
    label, reason = categorize(p, feats, tau_lo, tau_hi)
    return {"probability": p, "label": label, "reasons": reason, "features": feats}

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...),
                        tau_lo: float = Query(CONF["thr_best_macroF1"]),
                        tau_hi: float = Query(CONF["thr_target_fpr_10"])):
    raw = await file.read()
    df  = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: return {"error": f"missing columns: {missing}"}
    X  = build_features_frame(df)
    Xs = transform_for_model(X)
    probs = MODEL.predict_proba(Xs)[:,1]
    out = []
    for i,p in enumerate(probs):
        feats = X.iloc[i].to_dict()
        label, reason = categorize(float(p), feats, tau_lo, tau_hi)
        row = {k: df.iloc[i][k] for k in REQUIRED}
        row.update(probability=float(p), label=label, reasons=reason)
        out.append(row)
    return {"n": len(out), "thresholds": {"tau_lo":tau_lo,"tau_hi":tau_hi}, "results": out}