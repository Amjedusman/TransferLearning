# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model
joblib.dump(model, "iris_model.pkl")
print("✅ Model trained and saved as iris_model.pkl")
