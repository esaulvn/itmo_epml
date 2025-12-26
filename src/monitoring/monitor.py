import mlflow
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import sys

class ExperimentMonitor:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.start_time = None
        self.end_time = None
        self.experiment_name = None
        self.metrics = {}
        self.errors = []
        
    def _setup_logger(self):
        logger = logging.getLogger("experiment_monitor")
        logger.setLevel(logging.INFO)
        
        log_file = self.log_dir / f"experiment_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
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
    
    def start_experiment(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.metrics = {}
        self.errors = []
        
        self.logger.info(f"Начало эксперимента: {experiment_name}")
        self.logger.info(f"Время начала: {self.start_time}")
        
    def log_metrics(self, metrics: Dict[str, Any]):
        self.metrics.update(metrics)
        self.logger.info(f"Метрики: {metrics}")
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
    
    def log_config(self, config: Dict[str, Any]):
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        mlflow.log_params(self._flatten_dict(config))
        self.logger.info(f"Конфигурация сохранена: {config_file}")
    
    def log_error(self, error_msg: str):
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "message": error_msg
        })
        self.logger.error(f"Ошибка: {error_msg}")
    
    def log_warning(self, warning_msg: str):
        self.logger.warning(f"Предупреждение: {warning_msg}")
    
    def log_info(self, info_msg: str):
        self.logger.info(f"Инфо: {info_msg}")
    
    def end_experiment(self, status: str = "completed"):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        summary = {
            "experiment_name": self.experiment_name,
            "status": status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration,
            "metrics": self.metrics,
            "errors": self.errors,
            "config_file": f"{self.experiment_name}_config.json"
        }
        
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        mlflow.set_tag("status", status)
        mlflow.set_tag("duration_seconds", duration)
        
        self.logger.info(f"Эксперимент завершен: {status}")
        self.logger.info(f"Длительность: {duration:.2f} секунд")
        self.logger.info(f"Саммари сохранено: {summary_file}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:

        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def send_notification(self, message: str, level: str = "info"):
        self.logger.info(f"📢 Уведомление [{level}]: {message}")
        
        notification_file = self.log_dir / "notifications.log"
        with open(notification_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {level.upper()} - {message}\n")


class PerformanceMonitor:
    def __init__(self):
        import psutil
        self.psutil = psutil
        self.metrics = []
    
    def capture_snapshot(self):
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