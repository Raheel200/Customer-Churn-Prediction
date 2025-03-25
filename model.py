import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df["Gender"] = label_encoder.fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    return df


# Preprocess data
def preprocess_data(df):
    X = df.drop(columns=["Exited"])
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler


# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Save model
def save_model(model, scaler, feature_names, filename="rf_model.pkl"):
    with open(filename, "wb") as file:
        pickle.dump({"model": model, "scaler": scaler, "features": feature_names}, file)


# Execute training pipeline
if __name__ == "__main__":
    file_path = "Churn_Modelling.csv"  # Update path as needed
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)
    model = train_model(X_train, y_train)
    save_model(model, scaler, feature_names)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


import matplotlib.pyplot as plt

# Load trained model
with open("rf_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    feature_names = model_data["features"]

# Get feature importance
importances = model.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Customer Churn Prediction")
plt.gca().invert_yaxis()
plt.show()
