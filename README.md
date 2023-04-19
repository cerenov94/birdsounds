# birdsounds
* Dataset: https://www.kaggle.com/competitions/birdclef-2023/data
* Valid dataset : Собирался вручную
* Используется претрейн EfficientNet_b0, т.к. показывает лучшее качество.

## Starter config
configs.py
Стартовые конфигурации проекта
## Preprocessing
* imageconstructor.py 
* testDSconstructor.py
* test_audio_preprocess.py

Обработка аудио в мел спектрограммы и сохранением, сделано для того экономии времени.Отдельный файл для тестовой выборки по причине того, что файлы в тестовой выборке слиты в один длинный аудиофайл.

## Train
Run python main.py -r t

Содержит трейн датасет, модель , и функции для обучения.
