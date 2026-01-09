import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import sys

from clearml import Task, Logger


class ClearMLExperimentMonitor:
    def __init__(self, log_dir: str = "logs/clearml"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
        self.start_time = None
        self.end_time = None
        self.experiment_name = None
        self.metrics = {}
        self.errors = []
        
        self.task = None
        self.clearml_logger = None
        
    def _setup_logger(self):
        logger = logging.getLogger("clearml_experiment_monitor")
        logger.setLevel(logging.INFO)
        
        log_file = self.log_dir / f"clearml_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def init_clearml_task(self, project_name: str, task_name: str, tags: list = None, reuse_task: bool = False):
        current_task = Task.current_task()
        if current_task:
            current_task.close()
        
        try:
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                tags=tags or [],
                reuse_last_task_id=reuse_task,
                auto_connect_frameworks=True
            )
            self.clearml_logger = self.task.get_logger()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ClearML task: {e}")
            return False
    
    def start_experiment(self, experiment_name: str):
        self.start_time = datetime.now()
        if self.task:
            self.task.add_tags([f"exp:{experiment_name}"])
            
            execution_info = {
                "start_time": self.start_time.isoformat(),
                "status": "running",
                "experiment_name": experiment_name
            }
            self.task.connect(execution_info, name="Execution Details")

    
    def log_metrics(self, metrics: Dict[str, Any], iteration: int = 0):
        self.metrics.update(metrics)
        self.logger.info(f"Метрики: {metrics}")
        
        if self.clearml_logger:
            for name, value in metrics.items():
                try:
                    if isinstance(value, (int, float)):
                        self.clearml_logger.report_scalar(
                            title="metrics",
                            series=name,
                            value=float(value),
                            iteration=iteration
                        )
                    else:
                        self.task.set_tag(f"metric_{name}", str(value))
                except Exception as e:
                    self.logger.warning(f"Failed to log metric {name}={value}: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Конфигурация сохранена: {config_file}")
        
        if self.task:
            try:
                flat_config = self._flatten_dict(config)
                for key, value in flat_config.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        self.task.connect({key: value})
                    else:
                        self.task.connect({key: str(value)})
                
                self.task.upload_artifact('config', str(config_file))
            except Exception as e:
                self.logger.warning(f"Failed to log config to ClearML: {e}")
    
    def log_error(self, error_msg: str):
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "message": error_msg
        })
        self.logger.error(f"Ошибка: {error_msg}")
        
        if self.task:
            self.task.add_tags(["has_errors"])
            self.task.set_user_properties(name="last_error", value=error_msg[:200])

    
    def log_warning(self, warning_msg: str):
        self.logger.warning(f"Предупреждение: {warning_msg}")
    
    def log_info(self, info_msg: str):
        self.logger.info(f"Инфо: {info_msg}")
    
    def log_performance(self, cpu_percent: float, memory_percent: float):
        if self.clearml_logger:
            try:
                self.clearml_logger.report_scalar(
                    title="performance",
                    series="cpu_percent",
                    value=cpu_percent,
                    iteration=0
                )
                self.clearml_logger.report_scalar(
                    title="performance",
                    series="memory_percent",
                    value=memory_percent,
                    iteration=0
                )
            except Exception as e:
                self.logger.warning(f"Failed to log performance metrics: {e}")
    
    def end_experiment(self, status: str = "completed"):
        self.end_time = datetime.now()
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
        else:
            duration = 0
        
        summary = {
            "experiment_name": self.experiment_name,
            "status": status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration,
            "metrics": self.metrics,
            "errors": self.errors,
            "config_file": f"{self.experiment_name}_config.json"
        }
        
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Эксперимент завершен: {status}")
        self.logger.info(f"Длительность: {duration:.2f} секунд")
        self.logger.info(f"Сводка сохранена: {summary_file}")
        
        if self.task:
            self.task.add_tags([status])
            self.task.connect({
                "status": status,
                "duration_seconds": duration,
                "end_time": self.end_time.isoformat()
            }, name="Execution Details") 
    
    def close_clearml_task(self):
        if self.task:
            try:
                self.task.close()
                self.logger.info("ClearML task closed successfully")
            except Exception as e:
                self.logger.warning(f"Error closing ClearML task: {e}")

    
    def send_notification(self, message: str, level: str = "info"):
        safe_message = message.encode('ascii', 'ignore').decode('ascii')
        self.logger.info(f"Уведомление [{level}]: {safe_message}")
        
        notification_file = self.log_dir / "notifications.log"
        with open(notification_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} - {level.upper()} - {safe_message}\n")
        
        if self.task:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()} - {level} - {message}\n")
                    temp_path = f.name
                
                self.task.upload_artifact(f'notification_{datetime.now().strftime("%H%M%S")}', temp_path)
                import os
                os.unlink(temp_path)
            except Exception as e:
                self.logger.warning(f"Failed to log notification to ClearML: {e}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ClearMLPerformanceMonitor:
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.metrics = []
        except ImportError:
            self.psutil = None
            self.metrics = []
            logging.warning("psutil not installed, performance monitoring disabled")
    
    def capture_snapshot(self):
        if not self.psutil:
            return {}
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": self.psutil.cpu_percent(),
            "memory_percent": self.psutil.virtual_memory().percent,
            "memory_available_gb": self.psutil.virtual_memory().available / 1e9,
            "disk_usage_percent": self.psutil.disk_usage('/').percent
        }
        self.metrics.append(snapshot)
        return snapshot
    
    def get_report(self):
        if not self.metrics:
            return {}
        
        df = pd.DataFrame(self.metrics)
        report = {
            "avg_cpu_percent": df["cpu_percent"].mean(),
            "max_cpu_percent": df["cpu_percent"].max(),
            "avg_memory_percent": df["memory_percent"].mean(),
            "max_memory_percent": df["memory_percent"].max(),
            "samples": len(self.metrics)
        }
        return report