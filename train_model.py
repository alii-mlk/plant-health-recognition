import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
df = pd.read_csv("leaf_dataset_features.csv")

# Prepare features and target
X = df.drop(columns=["Filename", "Severity"])
y = df["Severity"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
model_path = os.path.join("models", "leaf_severity_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

