1. Клонировать репозиторий
```
git clone --branch hw_6 https://github.com/esaulvn/itmo_epml/
cd itmo_epml
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

4. Открыть app.clear.ml, выполнить в терминале команду и вставить ключ из своего аккаунта
```
poetry run clearml-init
```

5. Воспроизвести все эксперименнты и создать отчет
```
py -m poetry run dvc repro run_all_experiments_clearml
py -m poetry run dvc repro report
```

для запуска только одного эксперимента

```
py -m poetry run dvc repro run_single_experiment_clearml
py -m poetry run dvc repro report
```