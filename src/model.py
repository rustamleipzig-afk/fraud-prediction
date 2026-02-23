"""Model training and prediction utilities for the fraud prediction app."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import get_features_and_target, scale_features

MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"


def train_model(df: pd.DataFrame):
    """Train a RandomForest classifier and return the model, scaler, and evaluation metrics."""
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    feature_names = X.columns.tolist()
    importances = dict(zip(feature_names, model.feature_importances_))

    return model, scaler, report, cm, importances


def save_model(model, scaler, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """Persist the trained model and scaler to disk."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """Load a previously saved model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, scaler, input_df: pd.DataFrame) -> tuple[int, float]:
    """Return the predicted class (0/1) and fraud probability for a single transaction."""
    X_scaled = scaler.transform(input_df)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    return prediction, probability
