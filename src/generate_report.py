import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.io as pio

def generate_report():
    Path("monitoring").mkdir(exist_ok=True)
    
    reports = []
    
    if Path("experiments_results.csv").exists():
        df1 = pd.read_csv("experiments_results.csv")
        reports.append({
            'source': 'original_experiments',
            'data': df1
        })
    
    if Path("hydra_experiments_results.csv").exists():
        df2 = pd.read_csv("hydra_experiments_results.csv")
        reports.append({
            'source': 'hydra_experiments',
            'data': df2
        })
    
    all_results = []
    for report in reports:
        df = report['data']
        df['source'] = report['source']
        all_results.append(df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        summary = {
            'total_experiments': len(combined_df),
            'successful_experiments': len(combined_df[combined_df['status'] == 'success']),
            'failed_experiments': len(combined_df[combined_df['status'] != 'success']),
            'best_accuracy': combined_df['accuracy'].max(),
            'average_accuracy': combined_df['accuracy'].mean(),
            'best_algorithm': combined_df.loc[combined_df['accuracy'].idxmax(), 'algorithm'],
            'generated_at': datetime.now().isoformat()
        }
        
        with open("monitoring/experiment_report.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        fig = px.scatter(combined_df, x='experiment_name', y='accuracy', 
                        color='algorithm', title='Experiment Results')
        fig.write_html("monitoring/summary_report.html")
        
        print(f"Report generated: {summary}")

if __name__ == "__main__":
    generate_report()