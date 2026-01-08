import streamlit as st
import pandas as pd

st.set_page_config(page_title="DataPilot AI", layout="wide")

st.title("🚀 DataPilot AI")
st.write("Upload a CSV file and start exploring your data.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Basic Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
else:
    st.info("Please upload a CSV file to begin.")
