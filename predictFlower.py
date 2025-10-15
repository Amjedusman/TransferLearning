# app.py
import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load model and dataset
model = joblib.load("iris_model.pkl")
iris = load_iris()

# -------------------------------
# 🌸 App Title and Styling
# -------------------------------
st.set_page_config(page_title="Iris Flower Classifier 🌸", page_icon="🌺", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
    }
    h1 {
        color: #6a1b9a;
        text-align: center;
    }
    .stButton>button {
        background-color: #8e24aa;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stButton>button:hover {
        background-color: #7b1fa2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 🌼 App Content
# -------------------------------
st.title("🌸 Iris Flower Classification App")
st.write("This app predicts the **type of Iris flower** based on its sepal and petal measurements.")

# # Add a general Iris image
# st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg", 
#          caption="Iris Flower (Versicolor)", 
#          use_container_width=True)

st.markdown("---")
st.header("🔧 Input Flower Measurements")

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# -------------------------------
# 🧠 Prediction and Display
# -------------------------------
if st.button("🌼 Predict Iris Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    predicted_class = iris.target_names[prediction[0]]

    # Map species names to images
    species_images = {
        "setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
        "versicolor": "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg",
        "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }

    # Display prediction
    st.markdown("---")
    st.success(f"🎉 Predicted Iris Species: **{predicted_class.capitalize()}**")

    # Display corresponding image
    st.image(species_images[predicted_class], 
             caption=f"Iris {predicted_class.capitalize()}",
             use_container_width=True)

# -------------------------------
# 📊 Dataset Info Section
# -------------------------------
st.markdown("---")
st.subheader("📘 About the Iris Dataset")
st.info(
    """
    The **Iris dataset** is a classic dataset in machine learning with 150 samples of iris flowers —  
    50 for each species:
    - **Setosa** 🌺  
    - **Versicolor** 🌸  
    - **Virginica** 🌼  
    
    Each flower is described by:
    - Sepal length & width  
    - Petal length & width  
    """
)
