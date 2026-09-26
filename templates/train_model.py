
import pandas as pd
import string
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset/password_data.csv")

print("Dataset loaded successfully!")
print(df.head())

# Extract password features
def extract_features(password):
    return [
        len(password),
        sum(c.isupper() for c in password),
        sum(c.islower() for c in password),
        sum(c.isdigit() for c in password),
        sum(c in string.punctuation for c in password)
    ]

# Prepare input and output
X = df["password"].apply(extract_features).tolist()
y = df["strength"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Save trained model
joblib.dump(model, "password_model.pkl")

print("Model saved successfully!")