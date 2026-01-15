1. Обработать данные
```
py -m poetry run dvc repro prepare

```

2. Воспроизвести один эксперимент

```
py -m poetry run dvc repro run_single_experiment_clearml
```


1. Воспроизвести все эксперименты
```
py -m poetry run dvc repro run_all_experiments_clearml

```

4. Оценить результаты выполения экспериментов

```
py -m poetry run dvc repro evaluate_clearml
```

4. Создать страницу с отчетом

```
py -m poetry run dvc repro report
```
