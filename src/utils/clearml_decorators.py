"""
Модуль для интеграции с ClearML, предоставляющий декораторы и контекстные менеджеры
для автоматического трекинга экспериментов.
"""

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
    Декоратор для автоматического трекинга выполнения функций в ClearML.
    
    Автоматически создает задачу ClearML при вызове декорируемой функции,
    логирует параметры, метрики, модели и артефакты.
    
    Пример использования:
    ```python
    @clearml_track(
        project_name="LoanClassification",
        task_name="train_model",
        tags=["baseline", "xgboost"]
    )
    def train_model(data_path: str, n_estimators: int = 100) -> dict:
        # Тренировка модели
        model = XGBClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Расчет метрик
        accuracy = model.score(X_test, y_test)
        
        return {
            "model": model,
            "accuracy": accuracy,
            "f1_score": 0.92
        }
    ```
    
    Args:
        project_name (str, optional): Название проекта в ClearML. 
            Если не указано, используется значение из переменной окружения 
            CLEARML_PROJECT или "DefaultProject"
        task_name (str, optional): Название задачи (эксперимента). 
            По умолчанию используется имя декорируемой функции
        tags (list, optional): Список тегов для задачи
        log_params (bool): Логировать параметры функции. 
            Логируются только параметры простых типов (int, float, str, bool, None)
        log_metrics (bool): Логировать метрики из результата функции.
            Ожидает, что функция возвращает словарь, из которого извлекаются 
            числовые значения
        log_model (bool): Логировать модель, если результат имеет метод predict
        log_artifacts (bool): Логировать артефакты (в текущей реализации 
            используется для логирования моделей при log_model=True)
        reuse_last_task_id (bool): Переиспользовать последнюю задачу с тем же именем
        auto_connect_frameworks (bool): Автоматически подключать фреймворки 
            (PyTorch, TensorFlow и т.д.)
    
    Returns:
        Callable: Декорированная функция
        
    Notes:
        - Функция должна возвращать словарь для логирования метрик
        - Для логирования модели результат функции должен иметь метод predict
        - В случае исключения задача помечается как failed и исключение пробрасывается
        - Время выполнения функции автоматически логируется как метрика
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
    Контекстный менеджер для работы с ClearML задачами.
    
    Позволяет более гибко управлять трекингом экспериментов по сравнению
    с декоратором clearml_track. Подходит для случаев, когда нужно
    логировать данные из разных частей кода.
    
    Пример использования:
    ```python
    with ClearMLContext(
        project_name="LoanClassification",
        task_name="experiment_1",
        tags=["baseline", "random_forest"]
    ) as exp:
        exp.log_parameter("n_estimators", 100)
        exp.log_parameter("max_depth", 10)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        exp.log_metric("accuracy", accuracy)
        
        exp.log_model(model, "random_forest_model")
        exp.log_artifact("metrics.json")
    ```
    
    Args:
        project_name (str): Название проекта в ClearML
        task_name (str): Название задачи (эксперимента)
        tags (list, optional): Список тегов для задачи
        auto_connect_frameworks (bool): Автоматически подключать фреймворки
        reuse_last_task_id (bool): Переиспользовать последнюю задачу с тем же именем
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
        """
        Вход в контекст. Создает задачу ClearML.
        
        Returns:
            ClearMLContext: Сам объект контекста
        """
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
        """
        Логирует один параметр.
        
        Args:
            key (str): Ключ параметра
            value (Any): Значение параметра. Простые типы логируются как есть,
                сложные типы конвертируются в строку
        """
        if self.task:
            if isinstance(value, (int, float, str, bool, type(None))):
                self.task.connect({key: value})
            else:
                self.task.connect({key: str(value)})
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Логирует несколько параметров.
        
        Args:
            params (Dict[str, Any]): Словарь параметров для логирования
        """
        if self.task:
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    processed_params[key] = value
                else:
                    processed_params[key] = str(value)
            self.task.connect(processed_params)
    
    def log_metric(self, name: str, value: float, iteration: int = 0):
        """
        Логирует одну метрику.
        
        Args:
            name (str): Название метрики
            value (float): Значение метрики
            iteration (int): Номер итерации (по умолчанию 0)
        """
        if self.logger:
            self.logger.report_scalar(
                title="metrics",
                series=name,
                value=value,
                iteration=iteration
            )
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int = 0):
        """
        Логирует несколько метрик.
        
        Args:
            metrics (Dict[str, float]): Словарь метрик для логирования
            iteration (int): Номер итерации (по умолчанию 0)
        """
        if self.logger:
            for name, value in metrics.items():
                self.logger.report_scalar(
                    title="metrics",
                    series=name,
                    value=value,
                    iteration=iteration
                )
    
    def log_artifact(self, local_path: str, artifact_name: Optional[str] = None):
        """
        Загружает артефакт в ClearML.
        
        Args:
            local_path (str): Путь к локальному файлу
            artifact_name (str, optional): Имя артефакта в ClearML.
                Если не указано, используется имя файла
        """
        if self.task:
            name = artifact_name or os.path.basename(local_path)
            self.task.upload_artifact(name, local_path)
    
    def log_model(self, model, model_name: str = "model"):
        """
        Сериализует и загружает модель в ClearML.
        
        Args:
            model: Объект модели (должен поддерживать pickle)
            model_name (str): Имя модели в ClearML
        """
        if self.task:
            import pickle
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                temp_path = f.name
            
            self.task.upload_artifact(model_name, temp_path)
            os.unlink(temp_path)
    
    def log_text(self, text: str, name: str = "log"):
        """
        Сохраняет текстовые данные как артефакт.
        
        Args:
            text (str): Текст для сохранения
            name (str): Имя артефакта
        """
        if self.task:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(text)
                temp_path = f.name
            
            self.task.upload_artifact(name, temp_path)
            os.unlink(temp_path)
    
    def set_tag(self, key: str, value: str):
        """
        Устанавливает один тег для задачи.
        
        Args:
            key (str): Ключ тега
            value (str): Значение тега
        """
        if self.task:
            self.task.set_tags({key: value})
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Устанавливает несколько тегов для задачи.
        
        Args:
            tags (Dict[str, str]): Словарь тегов
        """
        if self.task:
            self.task.set_tags(tags)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Выход из контекста. Закрывает задачу ClearML.
        
        Args:
            exc_type: Тип исключения (если было)
            exc_val: Значение исключения (если было)
            exc_tb: Traceback исключения (если было)
        """
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
    """
    Класс для интеграции ClearML с Hydra конфигурацией.
    
    Позволяет логировать конфигурации Hydra в ClearML как параметры
    и как YAML артефакты.
    
    Пример использования:
    ```python
    @hydra.main(config_path="conf", config_name="config")
    def main(cfg):
        task = Task.current_task()
        hydra_integration = ClearMLHydraIntegration.from_current_task()
        hydra_integration.log_hydra_config(cfg)
        
        # ... основной код ...
    ```
    """
    
    def __init__(self, task: Task):
        """
        Инициализирует интеграцию с Hydra.
        
        Args:
            task (Task): Объект задачи ClearML
        """
        self.task = task
    
    @classmethod
    def from_current_task(cls):
        """
        Создает экземпляр класса из текущей задачи ClearML.
        
        Returns:
            ClearMLHydraIntegration: Экземпляр класса
            
        Raises:
            RuntimeError: Если нет текущей задачи ClearML
        """
        return cls(Task.current_task())
    
    def log_hydra_config(self, cfg):
        """
        Логирует конфигурацию Hydra в ClearML.
        
        Args:
            cfg: Конфигурация Hydra (обычно DictConfig)
            
        Notes:
            - Конфигурация логируется как параметры задачи
            - Также сохраняется как YAML файл артефакт
            - В случае ошибки выводится предупреждение, но исключение не пробрасывается
        """
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