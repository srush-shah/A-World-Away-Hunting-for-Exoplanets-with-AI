# A-World-Away-Hunting-for-Exoplanets-with-AI
NASA International Space Apps Challenge 2025



A world away turns NASA’s public exoplanet tables into calibrated planet probabilities with transparent reasons so scientists can decide what to follow up—fast.

Train: 100% Kepler KOI (cumulative)

Evaluate: TESS TOI split → 20% validation (calibration only), 80% held-out test

Model: LightGBM + Platt calibration (benchmarked vs XGBoost & Logistic)

Explainability: physics vetoes + tunable thresholds; optional plain-English explanations (Gemini API, not used for inference)

App: Streamlit UI with single entry + CSV batch upload, backed by FastAPI (Render)

Outputs: ranked probabilities, reasons, downloadable CSV, and a leaderboard


Hunt for the exoplanets [HERE](https://a-world-away-hunting-for-exoplanets-with-d7es.onrender.com/)!

Check out [this video](https://drive.google.com/file/d/1uDW9WdQq4psa6G1mbURNno3uMu1hUeTN/view?usp=sharing) of us hunting for the exoplanets.
