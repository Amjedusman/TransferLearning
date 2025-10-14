import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Title and Introduction
# -----------------------------
st.title("📊 My First Streamlit Data Explorer")
st.write("Welcome! This simple app lets you upload a CSV file and explore your data interactively.")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Show basic info
    st.subheader("📈 Data Summary")
    st.write(df.describe())

    # Choose column for visualization
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) > 0:
        st.subheader("📉 Visualize Data")
        col = st.selectbox("Choose a numeric column to plot", numeric_cols)

        fig, ax = plt.subplots()
        df[col].plot(kind='hist', bins=20, color='skyblue', edgecolor='black', ax=ax)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for visualization.")
else:
    st.info("Please upload a CSV file to begin.")