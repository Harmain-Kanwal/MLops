import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import threading
import time
import uvicorn
from main import app as fastapi_app

@st.cache_resource
def start_api_server():
    def run_server():
        uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(2)  # give it a moment to finish starting before the UI tries to call it
    return True

start_api_server()
api_url = "http://localhost:8000"

st.set_page_config(page_title="Churn Predictor | MLOps Demo", page_icon="📉", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #DB2777 100%);
    padding: 2.4rem 2rem; border-radius: 18px; margin-bottom: 1.5rem; color: white;
    box-shadow: 0 10px 30px rgba(79,70,229,0.25);
}
.main-header h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
.main-header p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.02rem; }

.metric-card {
    background: white; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align:center;
}
.metric-card .label { color: #6B7280; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { color: #111827; font-size: 1.9rem; font-weight: 800; margin-top: 0.3rem; }

.result-churn {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    border: 1px solid #FCA5A5; border-radius: 16px; padding: 1.6rem; text-align: center;
}
.result-stay {
    background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
    border: 1px solid #6EE7B7; border-radius: 16px; padding: 1.6rem; text-align: center;
}
.result-churn h2 { color: #991B1B; margin: 0; font-size: 1.6rem; }
.result-stay h2 { color: #065F46; margin: 0; font-size: 1.6rem; }

.factor-chip {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.7rem 1rem; border-radius: 10px; margin-bottom: 0.5rem; font-size: 0.88rem;
    transition: transform 0.15s ease;
}
.factor-chip:hover { transform: translateX(4px); }
.factor-churn { background: #FEF2F2; border-left: 4px solid #EF4444; color: #7F1D1D; }
.factor-stay { background: #F0FDF4; border-left: 4px solid #10B981; color: #14532D; }

section[data-testid="stSidebar"] { background-color: #111827; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: #E5E7EB !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>📉 Customer Churn Predictor</h1>
  <p>Will this customer stay or leave? A RandomForest model tracked with MLflow, served by FastAPI.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Monitoring", "ℹ️ About"])

HIDDEN_DEFAULTS = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "PaperlessBilling": "Yes",
}

DEFAULT_FIELDS = {
    "tenure": 12, "contract": "Month-to-month", "internet": "DSL",
    "monthly": 70.0, "payment": "Electronic check", "techsupport": "No",
}
for k, v in DEFAULT_FIELDS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_example(risk):
    if risk == "high":
        st.session_state.update({
            "tenure": 1, "contract": "Month-to-month", "internet": "Fiber optic",
            "monthly": 95.0, "payment": "Electronic check", "techsupport": "No",
        })
    else:
        st.session_state.update({
            "tenure": 60, "contract": "Two year", "internet": "DSL",
            "monthly": 45.0, "payment": "Credit card (automatic)", "techsupport": "Yes",
        })

with tab1:
    st.subheader("Enter the key details")
    st.caption("These 6 fields are the strongest churn predictors — everything else uses a realistic default.")

    ec1, ec2 = st.columns(2)
    ec1.button("⚡ Load a high-risk customer", on_click=load_example, args=("high",), use_container_width=True)
    ec2.button("🛡️ Load a loyal customer", on_click=load_example, args=("low",), use_container_width=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months as a customer)", 0, 72, key="tenure")
        Contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"], key="contract")
        InternetService = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="internet")
    with col2:
        MonthlyCharges = st.number_input("Monthly charges ($)", 0.0, 200.0, key="monthly")
        PaymentMethod = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="payment")
        if InternetService == "No":
            TechSupport = "No internet service"
            st.caption("Tech support: N/A (no internet service)")
        else:
            TechSupport = st.selectbox("Tech support", ["Yes", "No"], key="techsupport")

    if InternetService == "No":
        OnlineSecurity = OnlineBackup = DeviceProtection = StreamingTV = StreamingMovies = "No internet service"
    else:
        OnlineSecurity = OnlineBackup = DeviceProtection = StreamingTV = StreamingMovies = "No"

    TotalCharges = round(tenure * MonthlyCharges, 2) if tenure > 0 else MonthlyCharges

    if st.button("🔮 Predict churn", type="primary", use_container_width=True):
        payload = {
            **HIDDEN_DEFAULTS,
            "tenure": tenure, "Contract": Contract, "InternetService": InternetService,
            "MonthlyCharges": MonthlyCharges, "PaymentMethod": PaymentMethod, "TechSupport": TechSupport,
            "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
            "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies, "TotalCharges": TotalCharges,
        }
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            pct = result["churn_probability"] * 100
            is_churn = result["churn_prediction"] == "Yes"

            gcol, rcol = st.columns([1, 1.4])
            with gcol:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#DC2626" if is_churn else "#059669"},
                        'steps': [
                            {'range': [0, 33], 'color': "#D1FAE5"},
                            {'range': [33, 66], 'color': "#FEF3C7"},
                            {'range': [66, 100], 'color': "#FEE2E2"},
                        ],
                        'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.8, 'value': 50},
                    },
                ))
                fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with rcol:
                if is_churn:
                    st.markdown(f'<div class="result-churn"><h2>⚠️ Likely to CHURN</h2><p>Churn probability: <strong>{pct:.1f}%</strong></p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-stay"><h2>✅ Likely to STAY</h2><p>Churn probability: <strong>{pct:.1f}%</strong></p></div>', unsafe_allow_html=True)
                    st.balloons()
                st.caption(f"API latency: {result['latency_ms']} ms")

            factors = result.get("top_factors", [])
            if factors:
                st.markdown("##### Why this prediction")
                for f in factors:
                    chip_class = "factor-churn" if f["pushes_toward"] == "churn" else "factor-stay"
                    arrow = "↑ toward churn" if f["pushes_toward"] == "churn" else "↓ toward staying"
                    st.markdown(f'<div class="factor-chip {chip_class}"><span>{f["explanation"]}</span><strong>{arrow}</strong></div>', unsafe_allow_html=True)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API at {api_url}. Details: {e}")

with tab2:
    st.subheader("Live monitoring")
    if st.button("🔄 Refresh metrics"):
        st.rerun()
    try:
        resp = requests.get(f"{api_url}/metrics", timeout=15)
        resp.raise_for_status()
        m = resp.json()
        if m.get("total_predictions", 0) == 0:
            st.info("No predictions logged yet — make a prediction in the Predict tab first.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="label">Total Predictions</div><div class="value">{m["total_predictions"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="label">Churn Rate</div><div class="value">{m["churn_rate"]*100:.1f}%</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="label">Avg Latency</div><div class="value">{m["avg_latency_ms"]:.0f} ms</div></div>', unsafe_allow_html=True)

            st.write("")
            dc1, dc2 = st.columns([1, 1.5])
            with dc1:
                total = m["total_predictions"]
                churn_count = round(m["churn_rate"] * total)
                stay_count = total - churn_count
                donut = go.Figure(go.Pie(
                    labels=["Predicted churn", "Predicted stay"],
                    values=[churn_count, stay_count], hole=0.6,
                    marker=dict(colors=["#EF4444", "#10B981"]),
                ))
                donut.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title_text="Prediction split")
                st.plotly_chart(donut, use_container_width=True)
            with dc2:
                st.markdown("#### Data drift check")
                st.caption("Flags a feature if its live-traffic average has moved more than 20% from training.")
                st.dataframe(pd.DataFrame(m["drift"]).T, use_container_width=True)
                for col, d in m["drift"].items():
                    if d["drift_alert"]:
                        st.warning(f"Possible drift in **{col}**: training mean {d['train_mean']} vs live mean {d['live_mean']} ({d['pct_diff']}% difference)")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {api_url}. Details: {e}")

with tab3:
    st.subheader("About this project")
    st.markdown("""
    **Pipeline:** telecom customer data → scikit-learn RandomForest trained and tracked with **MLflow**
    → exported and served by a **FastAPI** REST API → this **Streamlit** app as client + monitoring dashboard.
    """)
    try:
        resp = requests.get(f"{api_url}/metrics", timeout=15)
        m = resp.json()
        if m.get("training_metrics"):
            tm = m["training_metrics"]
            st.markdown("#### Training metrics (from MLflow)")
            cols = st.columns(5)
            labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            for c, label, key in zip(cols, labels, keys):
                with c:
                    st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{tm[key]*100:.1f}%</div></div>', unsafe_allow_html=True)
    except requests.exceptions.RequestException:
        st.caption("Training metrics will appear here once the API is reachable.")