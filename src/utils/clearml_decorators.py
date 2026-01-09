import functools
import inspect
import time
import os
from typing import Any, Callable, Dict, Optional

from clearml import Task
import warnings


def clearml_track(
    project_name: str = None,
    task_name: str = None,
    tags: list = None,
    log_params: bool = True,
    log_metrics: bool = True,
    log_model: bool = False,
    log_artifacts: bool = True,
    reuse_last_task_id: bool = False,
    auto_connect_frameworks: bool = True,
):
    """
    Args:
        project_name: Название проекта в ClearML
        task_name: Название эксперимента
        tags: Список тегов
        log_params: Логировать параметры функции
        log_metrics: Логировать метрики из результата
        log_model: Логировать модель, если результат имеет метод predict
        log_artifacts: Логировать артефакты
        reuse_last_task_id: Переиспользовать последнюю задачу
        auto_connect_frameworks: Автоматически подключать фреймворки
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            exp_project = project_name or os.environ.get('CLEARML_PROJECT', 'DefaultProject')
            exp_task = task_name or func.__name__
            exp_tags = tags or []
            exp_tags.extend([func.__name__, func.__module__])
            
            task = Task.init(
                project_name=exp_project,
                task_name=exp_task,
                tags=exp_tags,
                reuse_last_task_id=reuse_last_task_id,
                auto_connect_frameworks=auto_connect_frameworks
            )
            
            if log_params:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                params_to_log = {}
                for param_name, param_value in bound_args.arguments.items():
                    if isinstance(param_value, (int, float, str, bool, type(None))):
                        params_to_log[param_name] = param_value
                    elif isinstance(param_value, (list, tuple, dict)):
                        params_to_log[param_name] = str(param_value)
                
                if params_to_log:
                    task.connect(params_to_log, name='function_parameters')
            
            task.set_tags({
                "function_name": func.__name__,
                "module": func.__module__,
                "status": "running"
            })
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                if log_metrics and isinstance(result, dict):
                    metrics_to_log = {}
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            metrics_to_log[key] = value
                    
                    logger = task.get_logger()
                    for metric_name, metric_value in metrics_to_log.items():
                        logger.report_scalar(
                            title="metrics",
                            series=metric_name,
                            value=metric_value,
                            iteration=0
                        )
                
                if log_model and hasattr(result, "predict"):
                    import pickle
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                        pickle.dump(result, f)
                        temp_path = f.name
                    
                    task.upload_artifact('model', temp_path)
                    os.unlink(temp_path)
                
                execution_time = time.time() - start_time
                logger.report_scalar(
                    title="performance",
                    series="execution_time",
                    value=execution_time,
                    iteration=0
                )
                
                task.set_tags({"status": "success"})
                task.close()
                
                return result
                
            except Exception as e:
                task.set_tags({
                    "status": "failed",
                    "error": str(e)[:100] 
                })
                task.close()
                raise
            
            finally:
                if task.running():
                    task.close()
        
        return wrapper
    
    return decorator


class ClearMLContext:
    """
    Контекстный менеджер для ClearML задач
    
    ```
    with ClearMLContext(
        project_name="LoanClassification",
        task_name="experiment_1",
        tags=["baseline", "random_forest"]
    ) as exp:
        exp.log_parameter("n_estimators", 100)
        exp.log_metric("accuracy", 0.95)
        exp.log_artifact("metrics.json")
    ```
    """
    
    def __init__(
        self,
        project_name: str,
        task_name: str,
        tags: Optional[list] = None,
        auto_connect_frameworks: bool = True,
        reuse_last_task_id: bool = False
    ):
        self.project_name = project_name
        self.task_name = task_name
        self.tags = tags or []
        self.auto_connect_frameworks = auto_connect_frameworks
        self.reuse_last_task_id = reuse_last_task_id
        self.task = None
        self.logger = None
        
    def __enter__(self):
        self.task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            tags=self.tags,
            auto_connect_frameworks=self.auto_connect_frameworks,
            reuse_last_task_id=self.reuse_last_task_id
        )
        
        self.logger = self.task.get_logger()
        return self
    
    def log_parameter(self, key: str, value: Any):
        if self.task:
            if isinstance(value, (int, float, str, bool, type(None))):
                self.task.connect({key: value})
            else:
                self.task.connect({key: str(value)})
    
    def log_parameters(self, params: Dict[str, Any]):
        if self.task:
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    processed_params[key] = value
                else:
                    processed_params[key] = str(value)
            self.task.connect(processed_params)
    
    def log_metric(self, name: str, value: float, iteration: int = 0):
        if self.logger:
            self.logger.report_scalar(
                title="metrics",
                series=name,
                value=value,
                iteration=iteration
            )
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int = 0):
        if self.logger:
            for name, value in metrics.items():
                self.logger.report_scalar(
                    title="metrics",
                    series=name,
                    value=value,
                    iteration=iteration
                )
    
    def log_artifact(self, local_path: str, artifact_name: Optional[str] = None):
        if self.task:
            name = artifact_name or os.path.basename(local_path)
            self.task.upload_artifact(name, local_path)
    
    def log_model(self, model, model_name: str = "model"):
        if self.task:
            import pickle
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                temp_path = f.name
            
            self.task.upload_artifact(model_name, temp_path)
            os.unlink(temp_path)
    
    def log_text(self, text: str, name: str = "log"):
        if self.task:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(text)
                temp_path = f.name
            
            self.task.upload_artifact(name, temp_path)
            os.unlink(temp_path)
    
    def set_tag(self, key: str, value: str):
        if self.task:
            self.task.set_tags({key: value})
    
    def set_tags(self, tags: Dict[str, str]):
        if self.task:
            self.task.set_tags(tags)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            if exc_type is not None:
                self.task.set_tags({
                    "status": "failed",
                    "error": str(exc_val)[:100]
                })
            else:
                self.task.set_tags({"status": "success"})
            
            if self.task.running():
                self.task.close()


class ClearMLHydraIntegration:
    def __init__(self, task: Task):
        self.task = task
    
    @classmethod
    def from_current_task(cls):
        return cls(Task.current_task())
    
    def log_hydra_config(self, cfg):
        if not self.task:
            return
        
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(cfg, DictConfig):
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                self.task.connect(cfg_dict, name='hydra_config')
                
                import tempfile
                import yaml
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
                    yaml.dump(cfg_dict, f, default_flow_style=False)
                    temp_path = f.name
                
                self.task.upload_artifact('hydra_config', temp_path)
                os.unlink(temp_path)
        except Exception as e:
            warnings.warn(f"Failed to log Hydra config: {e}")