import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Interactive Risk Scorer", page_icon="⚡", layout="wide")

st.title("Interactive Transaction Risk Scoring Engine")
st.markdown("Adjust transaction sliders and parameters below to simulate real-time enterprise fraud analysis.")

with st.form("advanced_scoring_form"):
    st.subheader("Transaction Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        step = st.number_input("Simulation Step (Hour)", min_value=0, value=150)
        tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"])
        
    with col2:
        amount = st.slider("Transaction Amount ($)", min_value=100.0, max_value=1000000.0, value=250000.0, step=5000.0)
        old_balance_org = st.number_input("Sender Old Balance ($)", min_value=0.0, value=250000.0, step=1000.0)
        
    with col3:
        new_balance_org = st.number_input("Sender New Balance ($)", min_value=0.0, value=0.0, step=1000.0)
        old_balance_dest = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=0.0, step=1000.0)
        new_balance_dest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=250000.0, step=1000.0)

    submit_button = st.form_submit_button(label="Run Real-Time ML Inference")

if submit_button:
    payload = {
        "step": int(step),
        "type": tx_type,
        "amount": float(amount),
        "oldbalanceOrg": float(old_balance_org),
        "newbalanceOrig": float(new_balance_org),
        "oldbalanceDest": float(old_balance_dest),
        "newbalanceDest": float(new_balance_dest)
    }
    
    try:
        with st.spinner("Executing XGBoost model & SHAP analysis..."):
            response = requests.post("http://127.0.0.1:8000/predictions/score", json=payload, timeout=5)
            
        if response.status_code == 200:
            res_data = response.json()
            st.markdown("---")
            st.subheader("Real-Time Risk Assessment Results")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            prob = res_data["fraud_probability"] * 100
            risk = res_data["risk_category"]
            
            with res_col1:
                st.metric("Tracking ID", res_data["transaction_id"])
                st.metric("Model Engine", res_data["model_version"])
                
                if risk == "HIGH":
                    st.error("CRITICAL ALERT: HIGH FRAUD RISK DETECTED")
                elif risk == "MEDIUM":
                    st.warning("MEDIUM RISK: MANUAL REVIEW RECOMMENDED")
                else:
                    st.success("LOW RISK: TRANSACTION LEGITIMATE")
                    
                st.markdown("### Root Cause Risk Factors:")
                for factor in res_data.get("top_risk_factors", ["Standard behavioral profile"]):
                    st.markdown(f"- {factor}")

            with res_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability Score (%)", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#ff4b4b" if prob > 75 else ("#ffa15a" if prob > 40 else "#00CC96")},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(0, 204, 150, 0.2)"},
                            {'range': [40, 75], 'color': "rgba(255, 161, 90, 0.2)"},
                            {'range': [75, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            # Model Explainability Feature Impact Chart
            st.markdown("---")
            st.subheader("Model Explainability (Feature Impact Weights)")
            weights = res_data.get("feature_weights", {
                "Transaction Amount": amount / 10000,
                "Sender Balance Error": 0.45,
                "Transaction Type Risk": 0.80 if tx_type in ["TRANSFER", "CASH_OUT"] else 0.10
            })
            df_weights = pd.DataFrame(list(weights.items()), columns=["Feature", "Impact Score"])
            fig_shap = px.bar(df_weights, x="Impact Score", y="Feature", orientation="h",
                             color="Impact Score", color_continuous_scale="Reds",
                             title="SHAP-Style Feature Attribution Weights")
            fig_shap.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_shap, use_container_width=True)
                
        else:
            st.error(f"API Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Connection failed! Ensure FastAPI is running.")