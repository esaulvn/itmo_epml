from typing import Dict, List

import mlflow
import pandas as pd
import yaml


class ExperimentManager:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    def load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def create_experiment(self, experiment_name: str) -> str:
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        return experiment_id

    def get_best_run(
        self, experiment_name: str, metric: str = "accuracy", ascending: bool = False
    ) -> Dict:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError()
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            return None
        if metric not in runs.columns:
            metric = "metrics.accuracy"
        best_run = runs.sort_values(metric, ascending=ascending).iloc[0]

        return {
            "run_id": best_run.run_id,
            "accuracy": best_run.get("metrics.accuracy", None),
            "f1_score": best_run.get("metrics.f1_score", None),
            "params": {
                k.replace("params.", ""): v
                for k, v in best_run.items()
                if k.startswith("params.")
            },
            "tags": {
                k.replace("tags.", ""): v
                for k, v in best_run.items()
                if k.startswith("tags.")
            },
        }

    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        results = []
        for exp_name in experiment_names:
            exp = mlflow.get_experiment_by_name(exp_name)
            if not exp:
                continue
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if runs.empty:
                continue
            avg_metrics = {
                "experiment": exp_name,
                "runs_count": len(runs),
                "avg_accuracy": runs["metrics.accuracy"].mean(),
                "best_accuracy": runs["metrics.accuracy"].max(),
                "avg_f1_score": runs["metrics.f1_score"].mean(),
                "best_f1_score": runs["metrics.f1_score"].max(),
            }
            results.append(avg_metrics)

        return pd.DataFrame(results)
