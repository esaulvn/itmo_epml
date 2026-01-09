from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from clearml import Task, Dataset, Model
from datetime import datetime, timedelta


class ClearMLExperimentManager:
    def __init__(self, project_name: Optional[str] = None):
        self.project_name = project_name
        
    def get_project_tasks(self, project_name: Optional[str] = None) -> List[Task]:
        project = project_name or self.project_name
        if not project:
            raise ValueError("Project name must be specified")
        
        tasks = Task.get_tasks(project_name=project)
        return tasks
    
    def get_task_by_id(self, task_id: str) -> Task:
        return Task.get_task(task_id=task_id)
    
    def get_best_run(
        self,
        project_name: Optional[str] = None,
        metric: str = "accuracy",
        ascending: bool = False,
        min_completed_runs: int = 1,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        project = project_name or self.project_name
        if not project:
            raise ValueError("Project name must be specified")
        
        tasks = Task.get_tasks(
            project_name=project,
            tags=tags or []
        )
        
        if not tasks:
            return None
        
        completed_tasks = [
            task for task in tasks 
            if task.status == 'completed' and task.get_last_scalar_series()
        ]
        
        if len(completed_tasks) < min_completed_runs:
            return None
        
        best_task = None
        best_metric_value = -float('inf') if not ascending else float('inf')
        
        for task in completed_tasks:
            try:
                scalar_series = task.get_last_scalar_series()
                
                for series_title in scalar_series:
                    if 'metrics' in series_title:
                        metrics_dict = scalar_series[series_title]
                        if metric in metrics_dict:
                            metric_value = metrics_dict[metric]['last']
                            
                            if not ascending: 
                                if metric_value > best_metric_value:
                                    best_metric_value = metric_value
                                    best_task = task
                            else: 
                                if metric_value < best_metric_value:
                                    best_metric_value = metric_value
                                    best_task = task
            except Exception as e:
                print(f"Error processing task {task.id}: {e}")
                continue
        
        if not best_task:
            return None
        
        return self._extract_task_info(best_task)
    
    def compare_experiments(
        self,
        project_names: List[str],
        metrics: List[str] = None,
        group_by_tags: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'roc_auc']
        
        results = []
        
        for project_name in project_names:
            tasks = Task.get_tasks(project_name=project_name)
            
            for task in tasks:
                if task.status != 'completed':
                    continue
                
                try:
                    task_info = self._extract_task_info(task)
            
                    scalar_series = task.get_last_scalar_series()
                    task_metrics = {}
                    
                    for series_title in scalar_series:
                        if 'metrics' in series_title:
                            metrics_dict = scalar_series[series_title]
                            for metric_name in metrics:
                                if metric_name in metrics_dict:
                                    task_metrics[metric_name] = metrics_dict[metric_name]['last']
                    
                    result = {
                        'project': project_name,
                        'task_name': task_info.get('task_name', ''),
                        'task_id': task_info.get('task_id', ''),
                        'status': task_info.get('status', ''),
                        'created': task_info.get('created', ''),
                        'tags': ', '.join(task_info.get('tags', [])),
                        'algorithm': task_info.get('algorithm', 'unknown'),
                        **task_metrics
                    }
                    
                    if group_by_tags:
                        for tag in group_by_tags:
                            result[f'tag_{tag}'] = tag in task_info.get('tags', [])
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing task {task.id}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def create_experiment_report(
        self,
        project_name: Optional[str] = None,
        output_path: str = 'experiment_report.csv'
    ) -> pd.DataFrame:
        project = project_name or self.project_name
        if not project:
            raise ValueError("Project name must be specified")
        
        tasks = Task.get_tasks(project_name=project)
        
        report_data = []
        
        for task in tasks:
            try:
                task_info = self._extract_task_info(task)
                
                parameters = task_info.get('parameters', {})
                
                metrics = {}
                if task.get_last_scalar_series():
                    scalar_series = task.get_last_scalar_series()
                    for series_title in scalar_series:
                        if 'metrics' in series_title:
                            metrics.update({
                                k: v['last'] for k, v in scalar_series[series_title].items()
                            })
                
                record = {
                    'task_id': task_info.get('task_id'),
                    'task_name': task_info.get('task_name'),
                    'status': task_info.get('status'),
                    'created': task_info.get('created'),
                    'completed': task_info.get('completed'),
                    'duration_minutes': task_info.get('duration_minutes'),
                    'tags': ', '.join(task_info.get('tags', [])),
                    'algorithm': task_info.get('algorithm', 'unknown'),
                    **{f'param_{k}': v for k, v in parameters.items()},
                    **{f'metric_{k}': v for k, v in metrics.items()}
                }
                
                report_data.append(record)
                
            except Exception as e:
                print(f"Error creating report for task {task.id}: {e}")
                continue
        
        df = pd.DataFrame(report_data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Report saved to {output_path}")
        
        return df
    
    def _extract_task_info(self, task: Task) -> Dict[str, Any]:
        parameters = task.get_parameters() or {}
        
        algorithm = 'unknown'
        if 'algorithm' in parameters:
            algorithm = parameters.get('algorithm', {}).get('name', 'unknown')
        
        tags = task.get_tags() or []
        for tag in tags:
            if tag in ['random_forest', 'logistic_regression', 'xgboost', 'svm', 'knn']:
                algorithm = tag
                break
        
        duration_minutes = None
        if task.data.started and task.data.completed:
            duration = task.data.completed - task.data.started
            duration_minutes = duration.total_seconds() / 60
        
        return {
            'task_id': task.id,
            'task_name': task.name,
            'project': task.project,
            'status': task.status,
            'created': task.data.created,
            'started': task.data.started,
            'completed': task.data.completed,
            'duration_minutes': duration_minutes,
            'tags': tags,
            'algorithm': algorithm,
            'parameters': parameters,
            'url': task.get_output_log_web_page()
        }


class ClearMLModelRegistry:
    def __init__(self, project_name: str = "Models"):
        self.project_name = project_name
    
    def register_model(
        self,
        model_path: str,
        model_name: str,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        description: str = ""
    ) -> Model:
        model = Model.create(
            model_name=model_name,
            tags=tags or [],
            description=description
        )
        
        model.update_weights(model_path)
        
        if metadata:
            model.set_metadata(metadata)
        
        model.publish()
        return model
    
    def get_model_versions(self, model_name: str) -> List[Model]:
        models = Model.query_models(
            project_name=self.project_name,
            model_name=model_name
        )
        return models
    
    def get_latest_model(self, model_name: str) -> Optional[Model]:
        models = self.get_model_versions(model_name)
        if not models:
            return None
        
        models.sort(key=lambda x: x.created, reverse=True)
        return models[0]
    
    def compare_models(
        self,
        model_name: str,
        metric: str = "accuracy"
    ) -> pd.DataFrame:
        models = self.get_model_versions(model_name)
        
        if not models:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model in models:
            metadata = model.get_metadata() or {}
            
            record = {
                'model_id': model.id,
                'model_name': model.name,
                'version': model.version,
                'created': model.created,
                'tags': ', '.join(model.tags),
                'description': model.comment or '',
                'framework': metadata.get('framework', 'unknown'),
                'algorithm': metadata.get('algorithm', 'unknown'),
                'metric': metadata.get('metrics', {}).get(metric, None)
            }
            
            if 'metrics' in metadata:
                for metric_name, metric_value in metadata['metrics'].items():
                    record[f'metric_{metric_name}'] = metric_value
            
            comparison_data.append(record)
        
        return pd.DataFrame(comparison_data)
    
    def promote_model(
        self,
        model_id: str,
        stage: str = "production",
        description: str = ""
    ) -> Model:
        model = Model(model_id=model_id)
        model.set_stage(stage)
        
        if description:
            model.comment = description
        
        print(f"Model {model.name} promoted to {stage}")
        return model

