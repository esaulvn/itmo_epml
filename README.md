esaulvn_hw3_epml
==============================

# Настройка MLFlow:

*   Установлен MLFlow, используем базу SQLite, создается в корне репозитория при запуске. 

*   В файл с пайплайном dvc добавлена часть для запуска экспериментов

```
py -m poetry install
py -m poetry run dvc repro prepare run_experiments
```
*   Созданы файлы с конфигами в папке configs для моделей 
    -    random forest
    -    log regression
    -    xgboost
    -    svm
    -    knn

*   Для запуска MLFlow с аутентификацией запускаем в другом окошке терминала через src\utils\start_with_auth.py файл, который берет значения из .env переменной. Все так же делаем через виртуальное окружение в poetry

```
py -m poetry install
py -m poetry run python src/utils/start_with_auth.py
```

![alt text](image.png)

![alt text](image-1.png)

# Проведение экспериментов:

*   Проведены эксперименты с 5-ю типами моделей и 15 конфигурациями параметров

![alt text](image-2.png)

*   Артефакты логируются в папку metrics, модели сохраняются в папку models, лучший результат выносится в файл experiments.json
*   Каждой конфигурации параметров присвоено название, результаты сохраняются в папки внутри mlruns


# Интеграция с кодом:

*   MLFlow интегриован в Python код в src/models/train-model.py, там же реализован контекстных менеджер
*   Созданы утилиты для работы с экспериментами - в папке src/utils

# Отчет о проделанной работе:
*   Создан отчет в формате Markdown
