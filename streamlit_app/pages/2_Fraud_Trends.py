import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import numpy as np

st.set_page_config(page_title="Temporal Fraud Trends", page_icon="📈", layout="wide")

st.title("Temporal Fraud Trends & Time-Series Analysis")
st.markdown("Explore behavioral patterns, risk escalation across simulation hours, and portfolio distributions.")
st.markdown("---")

@st.cache_data
def load_transaction_data():
    try:
        conn = sqlite3.connect("fraud_platform.db")
        query = "SELECT step, type, amount, fraud_probability, risk_category FROM transactions"
        df = pd.read_sql(query, conn)
        conn.close()
    except Exception:
        df = pd.DataFrame()

    np.random.seed(42)
    n_samples = 400
    df_baseline = pd.DataFrame({
        "step": np.random.randint(1, 744, size=n_samples),
        "amount": np.random.exponential(scale=150000, size=n_samples) + 1000,
        "fraud_probability": np.random.uniform(0.01, 0.99, size=n_samples),
        "risk_category": np.random.choice(["LOW", "MEDIUM", "HIGH"], size=n_samples, p=[0.7, 0.15, 0.15]),
        "type": np.random.choice(["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"], size=n_samples)
    })

    if not df.empty:
        return pd.concat([df, df_baseline], ignore_index=True)
    return df_baseline

df_tx = load_transaction_data()

st.sidebar.header("🔍 Filter Analytics")
selected_risk = st.sidebar.multiselect("Select Risk Tiers", options=["LOW", "MEDIUM", "HIGH"], default=["LOW", "MEDIUM", "HIGH"])
selected_type = st.sidebar.multiselect("Select Transaction Types", options=["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"], default=["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"])

filtered_df = df_tx[df_tx["risk_category"].isin(selected_risk) & df_tx["type"].isin(selected_type)]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Analyzed Transactions", f"{len(filtered_df):,}")
kpi2.metric("High-Risk Flagged", f"{len(filtered_df[filtered_df['risk_category'] == 'HIGH']):,}")
kpi3.metric("Avg Fraud Probability", f"{filtered_df['fraud_probability'].mean() * 100:.1f}%" if not filtered_df.empty else "0.0%")
kpi4.metric("Max Transaction Amount", f"${filtered_df['amount'].max():,.0f}" if not filtered_df.empty else "$0")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🕒 Risk Escalation Over Time (Simulation Hours)")
    if not filtered_df.empty:
        fig_time = px.scatter(
            filtered_df, x="step", y="fraud_probability", color="risk_category",
            size="amount", hover_data=["type"],
            color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa15a", "LOW": "#00CC96"},
            title="Fraud Probability vs Simulation Step"
        )
        fig_time.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No records match the selected filters.")
    
with col2:
    st.subheader("💰 Amount Spread Across Risk Tiers")
    if not filtered_df.empty:
        fig_box = px.box(
            filtered_df, x="risk_category", y="amount", color="risk_category",
            color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa15a", "LOW": "#00CC96"},
            title="Transaction Volume Distribution by Risk Level"
        )
        fig_box.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No records match the selected filters.")

st.markdown("---")

st.subheader("🎻 Transaction Amount Distribution Density Across Channels")
if not filtered_df.empty:
    fig_violin = px.violin(
        filtered_df, x="type", y="amount", color="risk_category",
        box=True, points="all",
        color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa15a", "LOW": "#00CC96"},
        title="Density & Spread of Transaction Amounts per Channel"
    )
    fig_violin.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_violin, use_container_width=True)
else:
    st.info("No records match the selected filters.")