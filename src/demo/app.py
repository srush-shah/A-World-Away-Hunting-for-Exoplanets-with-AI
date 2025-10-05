# src/demo/app.py
import streamlit as st, requests, pandas as pd, io, os

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # Gemini is optional

API = st.secrets.get("API_BASE", "http://localhost:8000")


def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None

st.title("Galaxy Gazers — ExoEdge: Discovering Worlds Beyond Our Solar System")
st.markdown("### *Kepler-trained, TESS-calibrated*")

st.divider()

with st.sidebar:
    st.header("Model & thresholds")
    info = requests.get(f"{API}/version").json()
    tau_lo = st.slider("tau_lo (CANDIDATE ≥)", 0.0, 1.0, float(info["thresholds"]["thr_best_macroF1"]), 0.001)
    tau_hi = st.slider("tau_hi (CONFIRMED_ALGO ≥)", 0.0, 1.0, float(info["thresholds"]["thr_target_fpr_10"]), 0.001)
    st.caption("Set macro-F1 or FPR≈10% operating points")


st.subheader("Test a Single Exoplanet")
fields = ['period_days','dur_hrs','depth_ppm','rp_re','teq_k','st_teff_k','st_logg_cgs','st_rad_rsun','st_mass_msun']
f_labels = ['Orbit Period (in days)','TransitDuration (in hours)','Transit Depth (in ppm)','PlanetRadius (in Earth radii)','Planet Equilibrium Temperature (in Kelvin)','Star Temperature (in Kelvin)','Star Surface Gravity (in cgs)','Star Radius (in Solar radii)','Star Mass (in Solar masses)']
cols = st.columns(2)
payload = {}
for i,k in enumerate(fields):
    with cols[i%2]:
        payload[k] = st.number_input(f_labels[i], value=1.0 if k!="teq_k" else 800.0, step=0.001, format="%.6f")
    # Add some vertical spacing every 2 inputs (after each row)
    if (i + 1) % 2 == 0:
        st.write("")  # Empty line for spacing

if st.button("Predict"):
    r = requests.post(f"{API}/predict", params={"tau_lo":tau_lo,"tau_hi":tau_hi}, json=payload)
    result = r.json()
    
    # Display results in a user-friendly format
    st.subheader("🔮 Prediction Results")
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display label with appropriate styling
        label = result.get('label', 'Unknown')
        if label == 'CONFIRMED_ALGO':
            st.success(f"**Classification:** ✅ {label}")
        elif label == 'CANDIDATE':
            st.warning(f"**Classification:** ⚠️ {label}")
        else:
            st.info(f"**Classification:** ℹ️ {label}")
    
    with col2:
        # Display probability
        probability = result.get('probability', 0)
        st.metric("**Confidence**", f"{probability:.1%}")
    
    # Display reason if available
    if 'reasons' in result and result['reasons']!='':
        st.subheader("📋 Explanation")
        st.info(f"**Reason:** {result['reasons']}")

    model = get_gemini_model()
    if model is not None:
        with st.expander("🤖 Ask Gemini to explain this prediction"):
            guidance = st.text_area(
                "Optional: add a question or focus (e.g., key drivers, astrophysical plausibility)",
                value="Explain the likely drivers of this classification and any caveats.",
            )
            if st.button("Ask Gemini"):
                try:
                    prompt = (
                        "You are assisting with exoplanet candidate triage. "
                        "Given input features and a model's classification with probability, "
                        "provide a concise, technically accurate explanation suitable for an astronomer. "
                        "Avoid fabricating model internals; reason from the feature values and thresholds.\n\n"
                        f"Features (JSON): {payload}\n"
                        f"Thresholds: tau_lo={tau_lo:.3f}, tau_hi={tau_hi:.3f}\n"
                        f"Model output: label={result.get('label')}, probability={result.get('probability')}\n"
                        f"API provided reasons (may be empty): {result.get('reasons','')}\n\n"
                        f"User guidance: {guidance}"
                    )
                    response = model.generate_content(prompt)
                    text = getattr(response, "text", None) or "No response."
                    st.markdown(text)
                except Exception as e:
                    st.warning(f"Gemini error: {e}")
    
    # # Optional: Show raw data in an expander for debugging
    # with st.expander("🔧 Raw API Response (for debugging)"):
    #     st.json(result)

st.divider()
st.subheader("Test a Batch of Exoplanets (CSV)")
st.caption("Upload CSV with columns: " + ", ".join(fields))
up = st.file_uploader("CSV", type=["csv"])
if up and st.button("Score file"):
    r = requests.post(f"{API}/predict-batch", params={"tau_lo":tau_lo,"tau_hi":tau_hi}, files={"file": up})
    js = r.json()
    
    # Display batch results in an attractive format
    st.subheader("📊 Batch Processing Results")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📈 **Rows Processed**",
            value=js.get("n", 0),
            help="Total number of exoplanets analyzed"
        )
    
    with col2:
        thresholds = js.get("thresholds", {})
        tau_lo_val = thresholds.get("tau_lo", tau_lo)
        st.metric(
            label="⚠️ **CANDIDATE Threshold**",
            value=f"{tau_lo_val:.3f}",
            help="Minimum probability for CANDIDATE classification"
        )
    
    with col3:
        tau_hi_val = thresholds.get("tau_hi", tau_hi)
        st.metric(
            label="✅ **CONFIRMED Threshold**",
            value=f"{tau_hi_val:.3f}",
            help="Minimum probability for CONFIRMED_ALGO classification"
        )
    
    # Display results table if available
    if "results" in js:
        st.subheader("🔍 Detailed Results")
        df = pd.DataFrame(js["results"])
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Results CSV", 
            data=csv, 
            file_name="exoedge_scored.csv", 
            mime="text/csv",
            help="Download the complete results with predictions and probabilities"
        )