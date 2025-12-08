import json
import pickle
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
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

warnings.filterwarnings("ignore")


def load_data():
    """Загрузка обработанных данных из DVC"""
    with open("data/processed/data.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def prepare_features(df):
    df = df.copy()

    if "customer_id" in df.columns:
        df = df.drop("customer_id", axis=1)

    date_columns = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
    if date_columns:
        df = df.drop(columns=date_columns)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "loan_status"]

    df = df.drop(columns=categorical_cols)
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    return X, y


def save_metrics(y_test, y_pred, y_pred_proba):
    """Сохранение метрик в файл для DVC"""
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, y_pred, output_dict=True)
    with open("metrics/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, y_pred).tolist()
    with open("metrics/confusion_matrix.json", "w") as f:
        json.dump(cm, f)

    return metrics


def main():
    print("Загружаем параметры из params.yaml")
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data = load_data()
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.read_csv("data/raw/dataset.csv")

    X, y = prepare_features(df)
    print(f"Признаков: {X.shape[1]}")
    print(f"Примеров: {X.shape[0]}")
    print(f"Распределение целевой переменной:\n{y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]} ")
    print(f"Test: {X_test.shape[0]} ")

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run():
        print("Начинаем эксперимент")

        mlflow.log_params(params["model"])
        mlflow.log_params(params["data"])

        mlflow.set_tag("problem_type", "binary_classification")
        mlflow.set_tag("target_variable", "loan_status")
        mlflow.set_tag("features_count", X.shape[1])
        mlflow.set_tag("data_shape", f"{X.shape[0]}x{X.shape[1]}")

        model = RandomForestClassifier(
            n_estimators=params["model"]["n_estimators"],
            max_depth=params["model"]["max_depth"],
            random_state=params["model"]["random_state"],
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        print("Модель обучена")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = save_metrics(y_test, y_pred, y_pred_proba)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            print(f"   {metric_name}: {value:.4f}")

        print("Сохраняем модель")

        model_path = "models/random_forest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        feature_names = list(X.columns)
        with open("models/feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)

        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=X_train.iloc[:5],
            registered_model_name="loan_status_predictor",
        )
        mlflow.log_artifact("metrics/metrics.json")
        mlflow.log_artifact("metrics/classification_report.json")
        mlflow.log_artifact("metrics/confusion_matrix.json")

        mlflow.set_tag("dvc_stage", "train")
        mlflow.set_tag("dataset_version", "1.0")

        print(f"   Модель сохранена: {model_path}")
        print("   Метрики сохранены в папке metrics/")
        print(f"   MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"\n Откройте MLflow UI: {params['mlflow']['tracking_uri']}")

        return metrics


if __name__ == "__main__":
    main()
