1. Клонировать репозиторий
```
git clone --branch hw_2 https://github.com/yourusername/enginiring-practices-ml.git
cd enginiring-practices-ml
```

2. Задать создание виртуального окружения внутри проекта в poetry, установить все зависимости
```
py -m poetry config virtualenvs.in-project true --local
py -m poetry install
py -m poetry install --with dev
```

3. Получить данные из dvc
```
py -m poetry run dvc pull
```

4. В новом терминале перейти в папку репозитория. Запустить MLflow UI для просмотра экспериментов через poetry
```
py -m poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host localhost
```
Открыть http://localhost:5000

5. Воспроизвести пайплайн
```
py -m poetry run dvc repro
```

6. Открыть app.clear.ml, выполнить в терминале команду и вставить ключ из своего аккаунта
```
poetry run clearml-init
```

5. Воспроизвести пайплайн для всех экспериментов и создать отчет
```
py -m poetry run dvc repro run_all_experiments_clearml
py -m poetry run dvc repro generate_report
```

для запуска только одного эксперимента

```
py -m poetry run dvc repro run_all_experiments_clearml
py -m poetry run dvc repro generate_report
```
