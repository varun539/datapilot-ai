import streamlit as st
import pandas as pd

from src.data_loader import load_csv
from src.eda import basic_profile

st.set_page_config(page_title="DataPilot AI", layout="wide")

st.title("🚀 DataPilot AI")
st.write("Upload a CSV file and get instant data profiling.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file)

    st.success("File uploaded successfully!")

    # ------------------------
    # Data Preview
    # ------------------------
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ------------------------
    # Profiling
    # ------------------------
    profile = basic_profile(df)

    st.subheader("📈 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", profile["rows"])
    col2.metric("Columns", profile["columns"])
    col3.metric("Duplicate Rows", profile["duplicates"])

    # ------------------------
    # Data Types
    # ------------------------
    st.subheader("🧬 Column Data Types")
    st.dataframe(profile["dtypes"])

    # ------------------------
    # Missing Values
    # ------------------------
    st.subheader("🚨 Missing Values")
    missing_df = profile["missing"].reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    st.dataframe(missing_df)

    # ------------------------
    # Numeric vs Categorical
    # ------------------------
    st.subheader("📦 Column Categories")

    col1, col2 = st.columns(2)

    col1.write("### 🔢 Numeric Columns")
    col1.write(profile["numeric_cols"])

    col2.write("### 🔤 Categorical Columns")
    col2.write(profile["categorical_cols"])

    # ------------------------
    # Statistics
    # ------------------------
    st.subheader("📊 Statistical Summary")
    st.dataframe(profile["describe"])

else:
    st.info("Please upload a CSV file to begin.")
