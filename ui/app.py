"""Streamlit frontend. It calls the FastAPI backend. (Bindu)

Run (API must be running first):  streamlit run ui/app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender")
st.caption("Streamlit app that calls the FastAPI backend")

# ---- Sidebar: API status and model info ----
with st.sidebar:
    st.header("API status")
    info = None
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health["model_loaded"]:
            info = requests.get(f"{API_URL}/info", timeout=5).json()
            st.success("API online, model loaded")
        else:
            st.warning("API online but model not trained yet")
    except requests.RequestException:
        st.error(f"Cannot reach API at {API_URL}")

    if info:
        st.write(f"**Model:** {info['model_name']}")
        for key, value in info["metrics"].items():
            st.write(f"{key}: {value:.3f}")

# ---- Main: ask for recommendations ----
if info:
    col1, col2 = st.columns(2)
    user_id = col1.number_input(
        "User ID", min_value=info["min_user_id"], max_value=info["max_user_id"],
        value=info["min_user_id"], step=1,
    )
    top_n = col2.slider("How many movies?", 1, 30, 10)

    if st.button("Get recommendations", type="primary"):
        resp = requests.get(f"{API_URL}/recommend/{int(user_id)}", params={"top_n": top_n}, timeout=30)
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json()["recommendations"])
            df["score"] = (df["score"] * 100).round(1).astype(str) + "%"
            st.dataframe(df.rename(columns={"score": "chance you like it"}), hide_index=True)
        else:
            st.error(resp.json().get("detail", "Something went wrong"))
else:
    st.info("Start the API and train the model first. See the README.")
