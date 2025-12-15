import functools
import inspect
import time
from typing import Any, Callable, Dict

import mlflow


def mlflow_track(
    experiment_name: str = None,
    log_params: bool = True,
    log_metrics: bool = True,
    log_model: bool = False,
    log_artifacts: bool = True,
):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            exp_name = experiment_name or func.__name__
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=func.__name__):
                if log_params:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    # логирование параметров
                    params_to_log = {}
                    for param_name, param_value in bound_args.arguments.items():
                        if isinstance(param_value, (int, float, str, bool)):
                            params_to_log[param_name] = param_value

                    mlflow.log_params(params_to_log)

                mlflow.set_tag("function_name", func.__name__)
                mlflow.set_tag("module", func.__module__)
                start_time = time.time()

                # логирование метрик и модели
                try:
                    result = func(*args, **kwargs)
                    if log_metrics and isinstance(result, dict):
                        metrics_to_log = {}
                        for key, value in result.items():
                            if isinstance(value, (int, float)):
                                metrics_to_log[key] = value
                        mlflow.log_metrics(metrics_to_log)

                    if log_model and hasattr(result, "predict"):
                        mlflow.sklearn.log_model(result, "model")

                    execution_time = time.time() - start_time
                    mlflow.log_metric("execution_time", execution_time)

                    return result

                except Exception as e:
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
                    raise

        return wrapper

    return decorator


class MLflowContext:
    """Контекстный менеджер для MLflow"""

    def __init__(self, experiment_name: str, run_name: str = None, tags: Dict = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}

    def __enter__(self):
        mlflow.set_experiment(self.experiment_name)
        self.active_run = mlflow.start_run(run_name=self.run_name)

        # Устанавливаем теги
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        return self

    def log_param(self, key: str, value: Any):
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float):
        mlflow.log_metric(key, value)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(exc_val))
        else:
            mlflow.set_tag("status", "success")

        mlflow.end_run()
