import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pickle
import pandas as pd
import yaml
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from clearml import Task

from src.models.train_model_clearml import prepare_features_simple, load_test_data
from src.utils.clearml_experiment_utils import ClearMLModelRegistry


def load_test_data():
    with open("data/processed/test.pkl", "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv("data/raw/dataset.csv")
        from sklearn.model_selection import train_test_split
        _, data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['loan_status'])
    
    return data


def evaluate_best_model_clearml(config_path: str = "params.yaml"):
    """Оценка лучшей модели с ClearML трекингом"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    task = Task.init(
        project_name="LoanClassification_Evaluation",
        task_name="evaluate_best_model_clearml",
        tags=["evaluation", "production"],
        reuse_last_task_id=False
    )
    
    try:
        test_df = load_test_data()
        
        X_test, y_test = prepare_features_simple(test_df, config)
        
        model_path = "models/best_model_clearml.pkl"
        if not os.path.exists(model_path):
            print(f"Файл модели не найден: {model_path}")
            model_files = list(Path("models").glob("*.pkl"))
            if model_files:
                model_path = str(model_files[0])
                print(f"Используем модель: {model_path}")
        
        with open(model_path, "rb") as f:
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
        
        logger = task.get_logger()
        for metric_name, value in metrics.items():
            logger.report_scalar(
                title="evaluation_metrics",
                series=metric_name,
                value=value,
                iteration=0
            )
        
        Path("evaluation").mkdir(exist_ok=True)
        with open("evaluation/evaluation_metrics_clearml.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        task.upload_artifact('evaluation_metrics', "evaluation/evaluation_metrics_clearml.json")
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "evaluation/classification_report_clearml.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        task.upload_artifact('classification_report', report_path)
        
        print(f"Метрики оценки: {metrics}")
        
        model_registry = ClearMLModelRegistry(project_name="LoanClassification_Models")
        registered_model = model_registry.register_model(
            model_path=model_path,
            model_name="evaluated_loan_classifier",
            tags=["evaluated", "production_ready"],
            metadata={
                "evaluation_metrics": metrics,
                "evaluation_date": datetime.now().isoformat(),
                "test_samples": len(y_test),
                "model_type": type(model).__name__
            },
            description=f"Model evaluated on test set with accuracy {metrics['accuracy']:.4f}"
        )
        
        print(f"Модель зарегистрирована в реестре: {registered_model.id}")
        
        if metrics['accuracy'] > 0.7:  
            model_registry.promote_model(
                model_id=registered_model.id,
                stage="production",
                description=f"Promoted to production after evaluation with accuracy {metrics['accuracy']:.4f}"
            )
        
        task.set_tags({
            "status": "success",
            "evaluation_accuracy": metrics['accuracy'],
            "production_ready": metrics['accuracy'] > 0.7
        })
        task.close()
        
        return metrics
        
    except Exception as e:
        print(f"Ошибка при оценке: {e}")
        task.set_tags({"status": "failed", "error": str(e)})
        task.close()
        raise


if __name__ == "__main__":
    evaluate_best_model_clearml()