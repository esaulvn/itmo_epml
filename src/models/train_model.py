import json
import pickle
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def load_data():
    with open("data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv("data/raw/dataset.csv")

    return data


def prepare_features_simple(df, config):
    df = df.copy()

    drop_columns = config["data"]["preprocessing"].get("drop_columns", [])
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)

    date_columns = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
    if date_columns:
        df = df.drop(columns=date_columns)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "loan_status"]

    if config["data"]["preprocessing"].get("encode_categorical", True):
        for col in categorical_cols:
            if df[col].nunique() <= 10:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            else:
                df = pd.get_dummies(df, columns=[col], drop_first=True)

    if config["data"]["preprocessing"].get("scale_numerical", False):
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != "loan_status"]

        if numerical_cols:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    return X, y


def create_model(config):
    algo_name = config["algorithm"]["name"]
    params = config["algorithm"]["hyperparameters"]

    if algo_name == "random_forest":
        return RandomForestClassifier(**params)
    elif algo_name == "logistic_regression":
        return LogisticRegression(**params)
    elif algo_name == "svm":
        return SVC(**params, probability=True)
    elif algo_name == "knn":
        return KNeighborsClassifier(**params)
    elif algo_name == "xgboost":
        return XGBClassifier(**params, eval_metric="logloss")
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def calculate_metrics(y_test, y_pred, y_pred_proba):
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))

    return metrics


def save_metrics_files(y_test, y_pred, y_pred_proba, run_name):
    Path("metrics").mkdir(exist_ok=True)

    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    with open(f"metrics/{run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, y_pred, output_dict=True)
    with open(f"metrics/{run_name}_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, y_pred).tolist()
    with open(f"metrics/{run_name}_confusion_matrix.json", "w") as f:
        json.dump(cm, f)

    return metrics


def train_with_config(config):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["experiment"]["name"])

    with mlflow.start_run(run_name=config["experiment"]["run_name"]):
        mlflow.log_params(config["algorithm"]["hyperparameters"])
        mlflow.log_params({f"data_{k}": v for k, v in config["data"].items()})

        for tag_key, tag_value in config["mlflow"]["tags"].items():
            mlflow.set_tag(tag_key, tag_value)

        df = load_data()
        X, y = prepare_features_simple(df, config)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            stratify=y,
        )

        model = create_model(config)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = save_metrics_files(
            y_test, y_pred, y_pred_proba, config["experiment"]["run_name"]
        )

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        Path("models").mkdir(exist_ok=True)
        model_path = f"models/{config['experiment']['run_name']}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, "model")

        if config["mlflow"]["log_artifacts"]:
            mlflow.log_artifact(
                f"metrics/{config['experiment']['run_name']}_metrics.json"
            )
            mlflow.log_artifact(
                f"metrics/{config['experiment']['run_name']}_classification_report.json"
            )

        return metrics


def main():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    metrics = train_with_config(config)
    return metrics


if __name__ == "__main__":
    main()
