import yaml
import glob
import sys
from pathlib import Path
import pandas as pd
from src.models.train_model import train_with_config
from src.utils.experiment_utils import ExperimentManager

def generate_configs():

    base_config = {
        "experiment": {
            "name": "binary_classification_loan",
            "run_name": ""
        },
        "data": {
            "test_size": 0.2,
            "random_state": 42,
            "preprocessing": {
                "drop_columns": ["customer_id"],
                "encode_categorical": True,
                "scale_numerical": False
            }
        },
        "mlflow": {
            "tags": {
                "dataset": "loan_dataset",
                "author": "student"
            },
            "log_artifacts": True,
            "register_model": False
        }
    }
    
    algorithms = [
        {
            "name": "random_forest",
            "hyperparameters": [
                {"n_estimators": 100, "max_depth": 10},
                {"n_estimators": 200, "max_depth": 15},
                {"n_estimators": 50, "max_depth": 5, "min_samples_split": 5},
                {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 2}
            ]
        },
        {
            "name": "logistic_regression",
            "hyperparameters": [
                {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000},
                {"C": 0.5, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000},
                {"C": 2.0, "penalty": "l1", "solver": "liblinear", "max_iter": 1000}
            ]
        },
        {
            "name": "xgboost",
            "hyperparameters": [
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 8},
                {"n_estimators": 50, "learning_rate": 0.2, "max_depth": 4}
            ]
        },
        {
            "name": "svm",
            "hyperparameters": [
                {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
                {"C": 0.5, "kernel": "rbf", "gamma": "auto"},
                {"C": 2.0, "kernel": "linear"}
            ]
        },
        {
            "name": "knn",
            "hyperparameters": [
                {"n_neighbors": 5, "weights": "uniform"},
                {"n_neighbors": 10, "weights": "distance"},
                {"n_neighbors": 7, "weights": "uniform", "metric": "manhattan"}
            ]
        }
    ]
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    config_count = 0
    for algo in algorithms:
        algo_dir = configs_dir / algo["name"]
        algo_dir.mkdir(exist_ok=True)
        
        for i, params in enumerate(algo["hyperparameters"], 1):
            config = base_config.copy()
            config["algorithm"] = {
                "name": algo["name"],
                "hyperparameters": params
            }
            config["experiment"]["run_name"] = f"{algo['name']}_{i}"
            config["mlflow"]["tags"]["algorithm"] = algo["name"]
            config["mlflow"]["tags"]["config_id"] = f"{algo['name']}_{i}"
            
            config_path = algo_dir / f"config_{i}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            config_count += 1
    
    print(f"Сгенерировано {config_count} конфигураций")
    return config_count

def run_all_experiments():

    config_files = list(glob.glob("configs/**/*.yaml", recursive=True))
    if not config_files:
        print("Конфиги не найдены, генерируем...")
        generate_configs()
        config_files = list(glob.glob("configs/**/*.yaml", recursive=True))
    
    print(f"Найдено {len(config_files)} конфигураций")
    results = []
    experiment_manager = ExperimentManager()
    
    for i, config_path in enumerate(config_files, 1):
        print(f"эксперимент {i}/{len(config_files)}: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            metrics = train_with_config(config)
            
            results.append({
                "config": config_path,
                "algorithm": config["algorithm"]["name"],
                "accuracy": metrics.get("accuracy", 0),
                "f1_score": metrics.get("f1_score", 0),
                "status": "success"
            })
            
            print(f"accuracy={metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            results.append({
                "config": config_path,
                "algorithm": "unknown",
                "accuracy": 0,
                "f1_score": 0,
                "status": f"failed: {str(e)}"
            })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiment_results.csv", index=False)
    
    if not df_results.empty:
        print(f"\nВсего экспериментов: {len(df_results)}")
        print(f"Успешных: {len(df_results[df_results['status'] == 'success'])}")
        print(f"Неудачных: {len(df_results[df_results['status'] != 'success'])}")
        
        best_by_algorithm = df_results[df_results['status'] == 'success'].groupby('algorithm')['accuracy'].max()
        print("\nЛучшая точность по алгоритмам:")
        for algo, acc in best_by_algorithm.items():
            print(f"  {algo}: {acc:.4f}")
        
        best_overall = df_results[df_results['status'] == 'success']['accuracy'].max()
        best_config = df_results[df_results['accuracy'] == best_overall].iloc[0]
        print(f"Лучший результат: {best_overall:.4f}")
        print(f"Алгоритм: {best_config['algorithm']}")
        print(f"Конфиг: {best_config['config']}")
    
    return df_results

if __name__ == "__main__":
    run_all_experiments()