import concurrent.futures
import yaml
import pandas as pd
from src.models.train_model import train_with_config
from datetime import datetime

def run_experiment_parallel(exp_config, common_settings):
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
    
    full_config["mlflow"]["tags"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        metrics = train_with_config(full_config)
        return {
            "experiment_name": exp_config["name"],
            "algorithm": exp_config["algorithm"],
            "accuracy": metrics.get("accuracy", 0),
            "f1_score": metrics.get("f1_score", 0),
            "roc_auc": metrics.get("roc_auc", 0),
            "status": "success",
        }
    except Exception as e:
        return {
            "experiment_name": exp_config["name"],
            "algorithm": exp_config["algorithm"],
            "accuracy": 0,
            "f1_score": 0,
            "roc_auc": 0,
            "status": f"failed: {str(e)}",
        }

def run_experiments_parallel(config_path: str = "configs/all_experiments.yaml", max_workers: int = 4):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiments = config["experiments"]
    common_settings = config["common_settings"]
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(run_experiment_parallel, exp_config, common_settings): exp_config 
            for exp_config in experiments
        }
        
        for future in concurrent.futures.as_completed(future_to_exp):
            exp_config = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Завершен: {result['experiment_name']}, Accuracy: {result['accuracy']:.4f}")
            except Exception as e:
                print(f"Ошибка в {exp_config['name']}: {str(e)}")
    
    return results

if __name__ == "__main__":
    results = run_experiments_parallel(max_workers=4)
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiments_results_parallel.csv", index=False)