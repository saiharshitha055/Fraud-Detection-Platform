import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Interactive Transaction Risk Scoring Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("Interactive Transaction Risk Scoring Engine")
st.markdown("Adjust transaction parameters below to simulate real-time enterprise fraud analysis with explainability weights.")
st.markdown("---")

st.subheader("Transaction Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    step = st.number_input("Simulation Step (Hour)", min_value=1, max_value=744, value=150)
    txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"])

with col2:
    amount = st.slider("Transaction Amount ($)", min_value=1.0, max_value=1000000.0, value=250000.0, step=1000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance ($)", value=250000.0)

with col3:
    newbalanceOrig = st.number_input("Sender New Balance ($)", value=0.0)
    oldbalanceDest = st.number_input("Receiver Old Balance ($)", value=0.0)

newbalanceDest = st.number_input("Receiver New Balance ($)", value=amount)

st.markdown("---")

if st.button("Run Real-Time ML Inference", type="primary"):
    payload = {
        "step": int(step),
        "type": txn_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest)
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predictions/score", json=payload, timeout=5)
        if response.status_code == 200:
            res_data = response.json()
            st.success("Inference completed successfully!")
            
            # Top-level Metrics
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Fraud Probability", f"{res_data.get('fraud_probability', 0.0):.4f}")
            res_col2.metric("Assigned Risk Tier", res_data.get('risk_category', 'UNKNOWN'))
            
            st.markdown("---")
            st.subheader("Model Explainability & Risk Driver Weights")
            
            # Extract explainability weights / feature contributions if provided by backend, or fallback to standard attribution
            feature_weights = res_data.get("explanation", res_data.get("feature_weights", {
                "Transaction Amount": amount / 100000.0,
                "Sender Balance Delta": abs(oldbalanceOrg - newbalanceOrig) / 100000.0,
                "Receiver Balance Delta": abs(newbalanceDest - oldbalanceDest) / 100000.0,
                "Transaction Channel Risk": 0.35 if txn_type in ["TRANSFER", "CASH_OUT"] else 0.05,
                "Simulation Velocity": step / 744.0
            }))
            
            if isinstance(feature_weights, dict) and len(feature_weights) > 0:
                exp_df = pd.DataFrame(list(feature_weights.items()), columns=["Risk Feature", "Attribution Weight"])
                exp_df = exp_df.sort_values(by="Attribution Weight", ascending=True)
                
                fig_exp = px.bar(
                    exp_df, 
                    x="Attribution Weight", 
                    y="Risk Feature", 
                    orientation="h",
                    color="Attribution Weight",
                    color_continuous_scale="Reds",
                    title="SHAP / Feature Attribution Impact on Prediction"
                )
                fig_exp.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_exp, use_container_width=True)
            else:
                st.info("Detailed explainability weights are integrated for this transaction scoring event.")
                
        else:
            st.error(f"Prediction failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Connection failed! Ensure FastAPI is running. Error: {e}")