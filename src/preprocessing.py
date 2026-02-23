"""Preprocessing utilities for the fraud prediction app."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "amount",
    "time_of_day",
    "day_of_week",
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_online",
    "online_order",
]

TARGET_COLUMN = "fraud"


def load_data(filepath: str) -> pd.DataFrame:
    """Load the transaction dataset from a CSV file."""
    return pd.read_csv(filepath)


def get_features_and_target(df: pd.DataFrame):
    """Split a DataFrame into feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """Convert a single input dictionary into a one-row DataFrame ready for prediction."""
    df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit a StandardScaler on training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
