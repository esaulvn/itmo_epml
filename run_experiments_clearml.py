"""
Запуск экспериментов с ClearML
"""
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
import json
import sys
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.train_model_clearml import train_with_config_clearml
from src.config.schema import validate_config
from src.monitoring.clearml_monitor import ClearMLExperimentMonitor
from src.utils.clearml_experiment_utils import (
    ClearMLExperimentManager, 
    ClearMLModelRegistry
)


def save_results_clearml(cfg, metrics, status):
    """
    Сохраняет результаты эксперимента ClearML в JSON файл.
    
    Args:
        cfg: Конфигурация эксперимента (словарь или объект конфигурации)
        metrics: Метрики, полученные в результате эксперимента
        status: Статус завершения эксперимента ("success" или "failed")
    
    Returns:
        dict: Словарь с сохраненными результатами
    """
    results_dir = Path("experiments_results_clearml")
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
        "metrics": metrics,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "tracking_tool": "clearml"
    }
    
    results_file = results_dir / f"{exp_name}_results.json"
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты ClearML сохранены в {results_file}")
    return result


@hydra.main(version_base="1.3", config_path="configs/conf", config_name="config")
def run_single_experiment_clearml(cfg: DictConfig):
    """
    Запускает одиночный эксперимент с использованием ClearML.
    
    Функция-обертка для Hydra, которая:
    1. Загружает и валидирует конфигурацию
    2. Инициализирует мониторинг ClearML
    3. Запускает обучение модели
    4. Логирует результаты и метрики
    5. Обрабатывает ошибки
    
    Args:
        cfg: Конфигурация эксперимента в формате DictConfig
    
    Returns:
        dict: Метрики эксперимента
    """
    print(f"Конфигурация эксперимента (ClearML):")
    print(OmegaConf.to_yaml(cfg))
    
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        validated_cfg = validate_config(config_dict)
    except Exception as e:
        print(f"Ошибка валидации конфигурации: {e}")
        raise
    
    monitor = ClearMLExperimentMonitor()
    monitor.start_experiment(validated_cfg['experiment']['run_name'])
    
    try:
        metrics = train_with_config_clearml(validated_cfg)
        
        monitor.log_metrics(metrics)
        monitor.log_config(validated_cfg)
        monitor.end_experiment("success")
        
        save_results_clearml(validated_cfg, metrics, "success")
        
        print(f"Эксперимент завершен успешно!")
        print(f"Метрики: {metrics}")
        
        return metrics
        
    except Exception as e:
        monitor.log_error(str(e))
        monitor.end_experiment("failed")
        save_results_clearml(validated_cfg, {}, f"failed: {str(e)}")
        print(f"\n❌ Эксперимент завершен с ошибкой: {e}")
        raise
        
    finally:
        monitor.close_clearml_task()

import traceback
def run_all_experiments_clearml(config_path: str = "configs/conf/experiments/all_experiments.yaml"):
    """
    Запускает несколько экспериментов из YAML файла конфигурации.
    
    Функция выполняет:
    1. Загрузку конфигураций экспериментов из YAML файла
    2. Последовательный запуск всех экспериментов
    3. Сбор и сохранение результатов
    4. Выбор и регистрацию лучшей модели
    5. Генерацию сводного отчета
    
    Args:
        config_path: Путь к YAML файлу с конфигурациями экспериментов
    
    Returns:
        pd.DataFrame: DataFrame с результатами всех экспериментов
    """
    import yaml
    
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    experiments = config["experiments"]
    common_settings = config["common_settings"]
    
    print(f"Запуск {len(experiments)} экспериментов с ClearML")
    
    results = []
    successful_models = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"Эксперимент {i}/{len(experiments)}: {exp_config['name']}")
        print(f"Алгоритм: {exp_config['algorithm']}")
        
        full_config = {
            "experiment": {
                "name": "LoanClassification_ClearML",
                "run_name": exp_config["name"],
            },
            "algorithm": {
                "name": exp_config["algorithm"],
                "hyperparameters": exp_config["hyperparameters"],
            },
            "data": common_settings["data"],
        }
        
        try:
            metrics = train_with_config_clearml(full_config)
            
            result = {
                "experiment_name": exp_config['name'],
                "algorithm": exp_config['algorithm'],
                "accuracy": metrics.get("accuracy", 0),
                "f1_score": metrics.get("f1_score", 0),
                "roc_auc": metrics.get("roc_auc", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            successful_models.append({
                "name": exp_config['name'],
                "algorithm": exp_config['algorithm'],
                "accuracy": metrics.get("accuracy", 0),
                "path": f"models/{exp_config['name']}.pkl"
            })
            
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"F1-score: {metrics.get('f1_score', 0):.4f}")
            print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            
        except Exception as e:
            result = {
                "experiment_name": exp_config['name'],
                "algorithm": exp_config['algorithm'],
                "accuracy": 0,
                "f1_score": 0,
                "roc_auc": 0,
                "precision": 0,
                "recall": 0,
                "status": f"failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            print(f"❌ Ошибка в эксперименте {exp_config['name']}: {type(e).__name__}: {e}")
            traceback.print_exc() 
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiments_results_clearml.csv", index=False, encoding='utf-8')
    
    successful = df_results[df_results["status"] == "success"]
    if not successful.empty:
        best_exp = successful.loc[successful["accuracy"].idxmax()]
        
        try:
            best_model_path = f"models/{best_exp['experiment_name']}.pkl"
            if os.path.exists(best_model_path):
                best_model = pickle.load(open(best_model_path, 'rb'))
                pickle.dump(best_model, open("models/best_model_clearml.pkl", 'wb'))
                print(f"Лучшая модель сохранена: {best_exp['experiment_name']}")
                
                model_registry = ClearMLModelRegistry(project_name="LoanClassification_Models")
                model_registry.register_model(
                    model_path="models/best_model_clearml.pkl",
                    model_name="best_loan_classifier",
                    tags=["best", "production_candidate", best_exp['algorithm']],
                    metadata={
                        "best_accuracy": best_exp["accuracy"],
                        "algorithm": best_exp['algorithm'],
                        "experiment_name": best_exp['experiment_name'],
                        "selected_at": datetime.now().isoformat()
                    },
                    description=f"Best model from experiments with accuracy {best_exp['accuracy']:.4f}"
                )
        except Exception as e:
            print(f"Не удалось сохранить лучшую модель: {e}")
        

        print("ЛУЧШИЙ РЕЗУЛЬТАТ:")
        print(f"Название: {best_exp['experiment_name']}")
        print(f"Алгоритм: {best_exp['algorithm']}")
        print(f"Accuracy: {best_exp['accuracy']:.4f}")
        print(f"F1-score: {best_exp['f1_score']:.4f}")
        print(f"ROC-AUC: {best_exp['roc_auc']:.4f}")
    
    summary = {
        "total_experiments": len(experiments),
        "successful": len(successful),
        "failed": len(df_results) - len(successful),
        "best_accuracy": successful["accuracy"].max() if not successful.empty else 0,
        "best_algorithm": best_exp["algorithm"] if not successful.empty else None,
        "best_experiment": best_exp["experiment_name"] if not successful.empty else None,
        "timestamp": datetime.now().isoformat(),
        "tracking_tool": "clearml"
    }
    
    with open("experiments_summary_clearml.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Сводка сохранена в experiments_summary_clearml.json")
    
    try:
        manager = ClearMLExperimentManager("LoanClassification")
        comparison_df = manager.compare_experiments(["LoanClassification"])
        if not comparison_df.empty:
            comparison_df.to_csv("experiments_comparison_clearml.csv", index=False, encoding='utf-8')
            top_experiments = comparison_df.nlargest(5, 'accuracy')[['task_name', 'algorithm', 'accuracy', 'f1_score']]
            print(top_experiments.to_string(index=False))
    except Exception as e:
        print(f"{e}")
    
    return df_results


def compare_all_models():
    """
    Сравнивает все модели в реестре ClearML для различных алгоритмов.
    
    Функция выполняет:
    1. Получение списка моделей для каждого алгоритма
    2. Сравнение метрик разных версий моделей
    3. Сохранение результатов сравнения в CSV файл
    4. Определение лучшей модели по accuracy
    """
    try:
        model_registry = ClearMLModelRegistry(project_name="LoanClassification_Models")
        algorithms = ["random_forest", "logistic_regression", "xgboost", "svm", "knn"]
        all_comparisons = []
        for algo in algorithms:
            try:
                comparison_df = model_registry.compare_models(f"loan_classifier_{algo}")
                if not comparison_df.empty:
                    all_comparisons.append(comparison_df)
                    print(comparison_df[['version', 'created', 'metric_accuracy']].to_string(index=False))
            except Exception as e:
                print(f"{algo}: {e}")
        
        if all_comparisons:
            full_comparison = pd.concat(all_comparisons, ignore_index=True)
            full_comparison.to_csv("models_comparison_clearml.csv", index=False, encoding='utf-8')
            print(f"Полное сравнение сохранено: models_comparison_clearml.csv")
            
            if 'metric_accuracy' in full_comparison.columns:
                best_model = full_comparison.loc[full_comparison['metric_accuracy'].idxmax()]
                print(f"Лучшая модель: {best_model['model_name']} v{best_model['version']}")
                print(f" Accuracy: {best_model['metric_accuracy']:.4f}")
                print(f"   Алгоритм: {best_model['algorithm']}")
    
    except Exception as e:
        print(f"{e}")


if __name__ == "__main__":
    """
    Точка входа в приложение.
    
    Обрабатывает аргументы командной строки:
    - Без аргументов: запускает одиночный эксперимент
    - --all: запускает все эксперименты и сравнивает модели
    - --compare: только сравнивает модели в реестре
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            run_all_experiments_clearml()
            compare_all_models()
            
        elif sys.argv[1] == "--compare":
            compare_all_models()

    else:
        run_single_experiment_clearml()