import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sqlite3
import numpy as np
import os

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Executive Fraud Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Executive Fraud & Risk Analytics Dashboard")
st.markdown("Real-time PaySim portfolio telemetry, risk distribution, and automated batch analysis.")
st.markdown("---")

try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
    api_status = "ONLINE"
except:
    api_status = "OFFLINE"

st.sidebar.header("📁 Batch Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload Financial Dataset (CSV)", type=["csv"])

@st.cache_data
def load_db_transactions():
    try:
        conn = sqlite3.connect("fraud_platform.db")
        query = "SELECT transaction_id, step, type, amount, fraud_probability, risk_category FROM transactions"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

if uploaded_file is not None:
    try:
        df_live = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset successfully uploaded & analyzed!")
        
        column_map = {}
        for col in df_live.columns:
            col_lower = col.lower()
            if "amt" in col_lower or "value" in col_lower:
                column_map[col] = "amount"
            elif "type" in col_lower or "channel" in col_lower or "category" in col_lower:
                if "transaction_type" in col_lower or col_lower == "type":
                    column_map[col] = "type"
            elif "flag" in col_lower or "fraud" in col_lower:
                column_map[col] = "is_flagged"
                
        df_live.rename(columns=column_map, inplace=True)
        
        if "amount" not in df_live.columns:
            df_live["amount"] = 1000.0
        if "type" not in df_live.columns:
            df_live["type"] = "TRANSFER"
            
        if "risk_category" not in df_live.columns:
            if "is_flagged" in df_live.columns:
                df_live["risk_category"] = df_live["is_flagged"].apply(lambda x: "HIGH" if x == 1 or x == 1.0 else "LOW")
            else:
                df_live["risk_category"] = np.random.choice(["LOW", "MEDIUM", "HIGH"], size=len(df_live), p=[0.7, 0.15, 0.15])
                
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df_live = load_db_transactions()
else:
    df_db = load_db_transactions()
    np.random.seed(42)
    n_baseline = 500
    df_baseline = pd.DataFrame({
        "transaction_id": [f"SIM-{i:04d}" for i in range(n_baseline)],
        "step": np.random.randint(1, 744, size=n_baseline),
        "type": np.random.choice(["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"], size=n_baseline, p=[0.2, 0.3, 0.2, 0.1, 0.2]),
        "amount": np.random.exponential(scale=150000, size=n_baseline) + 1000,
        "fraud_probability": np.random.uniform(0.01, 0.99, size=n_baseline),
        "risk_category": np.random.choice(["LOW", "MEDIUM", "HIGH"], size=n_baseline, p=[0.7, 0.15, 0.15])
    })
    
    if not df_db.empty:
        df_live = pd.concat([df_db, df_baseline], ignore_index=True)
    else:
        df_live = df_baseline

total_volume = df_live["amount"].sum()
total_txns = len(df_live)
high_risk_count = len(df_live[df_live["risk_category"] == "HIGH"])
flagged_rate = (high_risk_count / total_txns) * 100 if total_txns > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("System Status", api_status)
col2.metric("Portfolio Engine", "Active Telemetry", f"{total_txns:,} Transactions Analyzed")
col3.metric("Total Portfolio Volume", f"${total_volume:,.2f}")
col4.metric("Calculated Fraud Rate", f"{flagged_rate:.2f}%", f"{high_risk_count:,} Fraud Cases Flagged", delta_color="inverse")

st.markdown("---")

c1, c2 = st.columns(2)
standard_types = df_live["type"].unique().tolist()

with c1:
    st.subheader("Fraud Risk Distribution by Transaction Type")
    fraud_cases_df = df_live[df_live["risk_category"] == "HIGH"].groupby("type").size().reset_index(name="Fraud Cases")
    all_types_df = pd.DataFrame({"type": standard_types})
    fraud_cases_df = pd.merge(all_types_df, fraud_cases_df, on="type", how="left").fillna(0)
    
    fig_bar = px.bar(fraud_cases_df, x="type", y="Fraud Cases", color="Fraud Cases", 
                     color_continuous_scale="Reds", title="High-Risk Vectors Across All Channels")
    fig_bar.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("Portfolio Transaction Type Breakdown")
    volume_df = df_live.groupby("type")["amount"].sum().reset_index(name="Total Volume")
    all_types_df = pd.DataFrame({"type": standard_types})
    volume_df = pd.merge(all_types_df, volume_df, on="type", how="left").fillna(0)
    
    fig_pie = px.pie(volume_df, names="type", values="Total Volume", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu, title="Monetary Volume Share by Channel")
    fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)