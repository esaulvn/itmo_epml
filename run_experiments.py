import yaml
import mlflow
import pandas as pd
from src.models.train_model import train_with_config
from datetime import datetime

def run_all_experiments(config_path: str = "configs/all_experiments.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiments = config['experiments']
    common_settings = config['common_settings']
    
    print(f"Всего экспериментов: {len(experiments)}")
    
    results = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"Эксперимент {i}/{len(experiments)}: {exp_config['name']}")
        print(f"Алгоритм: {exp_config['algorithm']}")
        
        full_config = {
            'experiment': {
                'name': common_settings['mlflow']['experiment_name'],
                'run_name': exp_config['name']
            },
            'algorithm': {
                'name': exp_config['algorithm'],
                'hyperparameters': exp_config['hyperparameters']
            },
            'data': common_settings['data'],
            'mlflow': common_settings['mlflow']
        }
        
        full_config['mlflow']['tags']['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            metrics = train_with_config(full_config)
            
            results.append({
                'experiment_name': exp_config['name'],
                'algorithm': exp_config['algorithm'],
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0),
                'status': 'success'
            })
            
        except Exception as e:
            results.append({
                'experiment_name': exp_config['name'],
                'algorithm': exp_config['algorithm'],
                'accuracy': 0,
                'f1_score': 0,
                'roc_auc': 0,
                'status': f'failed: {str(e)}'
            })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('experiments_results.csv', index=False)
    
    successful = df_results[df_results['status'] == 'success']
    failed = df_results[df_results['status'] != 'success']
    
    if not successful.empty:

        best_overall = successful.loc[successful['accuracy'].idxmax()]
        print(f"Best result:")
        print(f"  Название: {best_overall['experiment_name']}")
        print(f"  Алгоритм: {best_overall['algorithm']}")
        print(f"  Accuracy: {best_overall['accuracy']:.4f}")
        print(f"  F1-score: {best_overall['f1_score']:.4f}")

    summary = {
        'total_experiments': len(experiments),
        'successful': len(successful),
        'failed': len(failed),
        'best_accuracy': successful['accuracy'].max() if not successful.empty else 0,
        'best_algorithm': best_overall['algorithm'] if not successful.empty else None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('experiments_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    try:
        experiment = mlflow.get_experiment_by_name(common_settings['mlflow']['experiment_name'])
        print(f"MLflow эксперимент: {experiment.name}")
        print(f"ID: {experiment.experiment_id}")
        print(f"Всего запусков: {len(successful)}")
        print(f"URL: {common_settings['mlflow']['tracking_uri']}")
    except:
        print("Error")
    
    return df_results

if __name__ == "__main__":
    run_all_experiments()