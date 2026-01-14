import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

METRICS_DIR = Path("metrics")
REPORTS_DIR = Path("docs/reports")
ASSETS_DIR = Path("docs/assets")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

records = []
for metrics_file in METRICS_DIR.glob("*_metrics.json"):
    model_name = metrics_file.stem.replace("_metrics", "")
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    record = {"model": model_name, **data}
    records.append(record)

if not records:
    print("No metrics files found in metrics/")
    exit()

df = pd.DataFrame(records)
df = df.round(4)

table_md = df.to_markdown(index=False)

plt.style.use("seaborn-v0_8")

plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="model", y="accuracy", palette="Blues_d")
plt.title("Accuracy comparison")
plt.xticks(rotation=45)
plt.tight_layout()
acc_path = ASSETS_DIR / "accuracy_comparison.png"
plt.savefig(acc_path)
plt.close()

plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="model", y="f1_score", palette="Greens_d")
plt.title("F1-score comparison")
plt.xticks(rotation=45)
plt.tight_layout()
f1_path = ASSETS_DIR / "f1_score_comparison.png"
plt.savefig(f1_path)
plt.close()

report_path = REPORTS_DIR / "experiments.md"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Отчет об экспериментах\n\n")
    f.write("## Сравнительная таблица моделей\n\n")
    f.write(table_md)
    f.write("\n\n## Визуализация результатов\n\n")
    f.write(f"![Accuracy Comparison]({acc_path.relative_to(REPORTS_DIR.parent)})\n\n")
    f.write(f"![F1-score Comparison]({f1_path.relative_to(REPORTS_DIR.parent)})\n\n")

print(f"Report generated: {report_path}")
