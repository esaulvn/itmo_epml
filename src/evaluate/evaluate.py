import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pickle
import pandas as pd
import yaml
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from src.models.train_model import prepare_features_simple, create_model

def load_test_data():
    with open("data/processed/test.pkl", "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv("data/raw/dataset.csv")
        from sklearn.model_selection import train_test_split
        _, data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['loan_status'])
    
    return data

def evaluate_best_model():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("evaluation")
    
    with mlflow.start_run(run_name="evaluate_best_model"):
        test_df = load_test_data()
        
        X_test, y_test = prepare_features_simple(test_df, config)
        
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        Path("evaluation").mkdir(exist_ok=True)
        with open("evaluation/evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("Evaluation metrics:", metrics)
        
        mlflow.sklearn.log_model(model, "best_model_evaluated")
        
        return metrics

if __name__ == "__main__":
    evaluate_best_model()