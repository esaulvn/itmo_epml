import pandas as pd
import pickle
import mlflow
from datetime import datetime
from pathlib import Path
import json
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.train_model import train_with_config
from src.config.schema import validate_config
from src.monitoring.monitor import ExperimentMonitor

def save_results(cfg, metrics, status):
    results_dir = Path("experiments_results")
    results_dir.mkdir(exist_ok=True)
    
    if hasattr(cfg, 'experiment'):
        exp_name = cfg.experiment.run_name
        algo_name = cfg.algorithm.name
    elif isinstance(cfg, dict):
        exp_name = cfg.get('experiment', {}).get('run_name', 'unknown')
        algo_name = cfg.get('algorithm', {}).get('name', 'unknown')
    else:
        exp_name = 'unknown'
        algo_name = 'unknown'
    
    result = {
        "experiment_name": exp_name,
        "algorithm": algo_name,
        "hyperparameters": cfg.algorithm.hyperparameters if hasattr(cfg, 'algorithm') else {},
        "metrics": metrics,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    
    results_file = results_dir / f"{exp_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Результаты сохранены в {results_file}")
    return result

@hydra.main(version_base="1.3", config_path="configs/conf", config_name="config")
def run_single_experiment(cfg: DictConfig):
    print(f"Конфигурация эксперимента:")
    print(OmegaConf.to_yaml(cfg))
    
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=False) 
    except Exception as e:
        config_dict = {}
    
    if 'mlflow' in config_dict and 'tags' in config_dict['mlflow']:
        config_dict['mlflow']['tags']['algorithm'] = config_dict.get('algorithm', {}).get('name', 'unknown')
    
    try:
        validated_cfg = validate_config(config_dict)
    except Exception as e:
        print(f"Ошибка валидации конфигурации: {e}")
        raise
    
    monitor = ExperimentMonitor()
    monitor.start_experiment(validated_cfg['experiment']['run_name'])
    
    try:
        metrics = train_with_config(validated_cfg)
        
        monitor.log_metrics(metrics)
        monitor.log_config(validated_cfg)
        monitor.end_experiment("success")
        
        save_results(validated_cfg, metrics, "success")
        
        return metrics
        
    except Exception as e:
        monitor.log_error(str(e))
        monitor.end_experiment("failed")
        save_results(validated_cfg, {}, f"failed: {str(e)}")
        raise

def run_all_experiments_from_yaml(config_path: str = "configs/conf/experiments/all_experiments.yaml"):
    import yaml
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiments = config["experiments"]
    common_settings = config["common_settings"]
    
    results = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"Эксперимент {i}/{len(experiments)}: {exp_config['name']}")

        full_config = {
            "experiment": {
                "name": common_settings["mlflow"]["experiment_name"],
                "run_name": exp_config["name"],
            },
            "algorithm": {
                "name": exp_config["algorithm"],
                "hyperparameters": exp_config["hyperparameters"],
            },
            "data": common_settings["data"],
            "mlflow": common_settings["mlflow"],
        }
        
        full_config["mlflow"]["tags"]["algorithm"] = exp_config["algorithm"]
        
        try:
            metrics = train_with_config(full_config)
            result = {
                "experiment_name": exp_config['name'],
                "algorithm": exp_config['algorithm'],
                "accuracy": metrics.get("accuracy", 0),
                "f1_score": metrics.get("f1_score", 0),
                "roc_auc": metrics.get("roc_auc", 0),
                "status": "success",
            }
            results.append(result)
            
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            result = {
                "experiment_name": exp_config['name'],
                "algorithm": exp_config['algorithm'],
                "accuracy": 0,
                "f1_score": 0,
                "roc_auc": 0,
                "status": f"failed: {str(e)}",
            }
            results.append(result)
            print(f"Ошибка: {e}")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiments_results.csv", index=False)
    
    successful = df_results[df_results["status"] == "success"]
    if not successful.empty:
        best_exp = successful.loc[successful["accuracy"].idxmax()]
        
        try:
            best_model_path = f"models/{best_exp['experiment_name']}.pkl"
            best_model = pickle.load(open(best_model_path, 'rb'))
            pickle.dump(best_model, open("models/best_model.pkl", 'wb'))
            print(f"\nЛучшая модель сохранена: {best_exp['experiment_name']}")
        except:
            print("\nНе удалось сохранить лучшую модель")

        print("\nBest result:")
        print(f"  Название: {best_exp['experiment_name']}")
        print(f"  Алгоритм: {best_exp['algorithm']}")
        print(f"  Accuracy: {best_exp['accuracy']:.4f}")
        print(f"  F1-score: {best_exp['f1_score']:.4f}")

    summary = {
        "total_experiments": len(experiments),
        "successful": len(successful),
        "failed": len(df_results) - len(successful),
        "best_accuracy": successful["accuracy"].max() if not successful.empty else 0,
        "best_algorithm": best_exp["algorithm"] if not successful.empty else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("experiments_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nСводка сохранена в experiments_summary.json")
    
    try:
        experiment = mlflow.get_experiment_by_name(
            common_settings["mlflow"]["experiment_name"]
        )
        print(f"\nMLflow эксперимент: {experiment.name}")
        print(f"ID: {experiment.experiment_id}")
        print(f"Всего успешных запусков: {len(successful)}")
    except:
        print("\nИнформация MLflow недоступна")

    return df_results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_experiments_from_yaml()
    else:
        run_single_experiment()