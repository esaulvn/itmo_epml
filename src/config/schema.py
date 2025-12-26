from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from hydra.core.config_store import ConfigStore
import omegaconf
from datetime import datetime

@dataclass
class DataConfig:
    test_size: float = 0.2
    random_state: int = 42
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "drop_columns": ["customer_id"],
        "encode_categorical": True,
        "scale_numerical": False
    })
    paths: Dict[str, str] = field(default_factory=lambda: {
        "raw": "data/raw/dataset.csv",
        "processed": "data/processed",
        "train": "data/processed/train.pkl",
        "test": "data/processed/test.pkl"
    })

@dataclass
class AlgorithmConfig:
    name: str = "random_forest"
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    })

@dataclass
class ExperimentConfig:
    name: str = "binary_classification_loan"
    run_name: str = "default_run"
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "loan_classification_experiments"
    tags: Dict[str, str] = field(default_factory=lambda: {
        "dataset": "loan_dataset",
        "author": "student",
        "project": "hw1_epml",
        "hydra_config": "true"
    })
    log_artifacts: bool = True
    register_model: bool = False
    autolog: bool = False

@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

def validate_config(cfg) -> dict:
    """Конвертирует конфиг Hydra в обычный dict для обратной совместимости"""
    if hasattr(cfg, '_metadata'):
        try:
            cfg_dict = {}
            for key in cfg.keys():
                if hasattr(cfg[key], '_metadata'):
                    cfg_dict[key] = omegaconf.OmegaConf.to_container(cfg[key], resolve=True)
                else:
                    cfg_dict[key] = cfg[key]
        except:
            cfg_dict = dict(cfg)
    elif isinstance(cfg, dict):
        cfg_dict = cfg.copy()
    else:
        cfg_dict = dict(cfg)
    
    if 'experiment' not in cfg_dict:
        cfg_dict['experiment'] = {}
    if 'run_name' not in cfg_dict['experiment']:
        cfg_dict['experiment']['run_name'] = 'default_run_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if "mlflow" in cfg_dict and "tags" in cfg_dict["mlflow"]:
        if "algorithm" not in cfg_dict["mlflow"]["tags"]:
            cfg_dict["mlflow"]["tags"]["algorithm"] = cfg_dict.get("algorithm", {}).get("name", "unknown")
    
    if "data" in cfg_dict and "test_size" in cfg_dict["data"]:
        test_size = cfg_dict["data"]["test_size"]
        if test_size <= 0 or test_size >= 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    if "algorithm" in cfg_dict and "name" in cfg_dict["algorithm"]:
        algo_name = cfg_dict["algorithm"]["name"]
        valid_algorithms = ["random_forest", "logistic_regression", "svm", "knn", "xgboost"]
        if algo_name not in valid_algorithms:
            raise ValueError(f"Unsupported algorithm: {algo_name}. Valid: {valid_algorithms}")
    
    return cfg_dict