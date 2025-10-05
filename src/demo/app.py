# src/demo/app.py
import streamlit as st, requests, pandas as pd, io, os

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # Gemini is optional

API = st.secrets.get("API_BASE", os.environ.get("API_BASE", "http://localhost:8000"))


@st.cache_resource(show_spinner=False)
def _list_gemini_models(api_key: str):
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        # Keep only models that support text generation
        usable = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if any(x.lower() == "generatecontent" for x in methods):
                usable.append(m)
        return usable
    except Exception:
        return []


def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key or genai is None:
        return None
    usable = _list_gemini_models(api_key)
    if not usable:
        return None
    # Preferred order; otherwise fall back to first usable
    preferred = [
        "gemini-1.5-flash-002",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
    ]
    name_to_model = {getattr(m, "name", ""): m for m in usable}
    chosen_name = None
    for p in preferred:
        # API returns names like models/gemini-1.0-pro; handle both
        exact = f"models/{p}"
        if exact in name_to_model or p in name_to_model:
            chosen_name = exact if exact in name_to_model else p
            break
    if chosen_name is None:
        chosen_name = getattr(usable[0], "name", None)
    if not chosen_name:
        return None
    # Normalize to bare model id for constructor
    bare = chosen_name.split("/")[-1]
    try:
        model = genai.GenerativeModel(bare)
        st.session_state["gemini_model_name"] = bare
        return model
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

    # Gemini status indicator
    # _gemini = get_gemini_model()
    # if _gemini is not None:
    #     st.success("Gemini: configured")
    # else:
    #     st.info("Gemini: not configured (set GEMINI_API_KEY)")


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
    st.session_state["last_result"] = r.json()
    st.session_state["last_payload"] = payload
    st.session_state["last_tau_lo"] = float(tau_lo)
    st.session_state["last_tau_hi"] = float(tau_hi)

# Render last prediction if available
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    last_payload = st.session_state.get("last_payload", {})
    last_tau_lo = st.session_state.get("last_tau_lo", tau_lo)
    last_tau_hi = st.session_state.get("last_tau_hi", tau_hi)

    st.subheader("🔮 Prediction Results")

    col1, col2 = st.columns([1, 1])
    with col1:
        label = result.get('label', 'Unknown')
        if label == 'CONFIRMED_ALGO':
            st.success(f"**Classification:** ✅ {label}")
        elif label == 'CANDIDATE':
            st.warning(f"**Classification:** ⚠️ {label}")
        else:
            st.info(f"**Classification:** ℹ️ {label}")

    with col2:
        probability = result.get('probability', 0)
        st.metric("**Confidence**", f"{probability:.1%}")

    if 'reasons' in result and result['reasons']!='':
        st.subheader("📋 Explanation")
        st.info(f"**Reason:** {result['reasons']}")

    model = get_gemini_model()
    if model is not None:
        with st.expander("🤖 Ask Gemini to explain this prediction", expanded=bool(st.session_state.get("gemini_response"))):
            guidance = st.text_area(
                "Optional: add a question or focus (e.g., key drivers, astrophysical plausibility)",
                value=st.session_state.get("gemini_guidance", "Explain the likely drivers of this classification and any caveats."),
                key="gemini_guidance",
            )
            if st.button("Ask Gemini", key="ask_gemini_btn"):
                try:
                    prompt = (
                        "You are assisting with exoplanet candidate triage. "
                        "Given input features and a model's classification with probability, "
                        "provide a concise, technically accurate explanation suitable for an astronomer. "
                        "Avoid fabricating model internals; reason from the feature values and thresholds.\n\n"
                        f"Features (JSON): {last_payload}\n"
                        f"Thresholds: tau_lo={last_tau_lo:.3f}, tau_hi={last_tau_hi:.3f}\n"
                        f"Model output: label={result.get('label')}, probability={result.get('probability')}\n"
                        f"API provided reasons (may be empty): {result.get('reasons','')}\n\n"
                        f"User guidance: {guidance}"
                    )
                    response = model.generate_content(prompt)
                    st.session_state["gemini_response"] = getattr(response, "text", None) or "No response."
                except Exception as e:
                    st.session_state["gemini_response"] = f"Gemini error: {e}"

            if "gemini_response" in st.session_state:
                st.markdown(st.session_state["gemini_response"])
    else:
        st.info("Gemini explanation is unavailable.")
    
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