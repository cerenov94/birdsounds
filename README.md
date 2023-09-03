# birdsounds
* Dataset: https://www.kaggle.com/competitions/birdclef-2023/data
* Valid dataset : Собирался вручную
* Используется претрейн EfficientNet_b0, т.к. показывает лучшее качество.

## Starter config
configs.py
Стартовые конфигурации проекта.
## Preprocessing
* imageconstructor.py 
* testDSconstructor.py
* test_audio_preprocess.py

Обработка аудио в мел спектрограммы, сделано для экономии времени.Отдельный файл для тестовой выборки, файлы в тестовой выборке слиты в один длинный аудиофайл.

## Train
Run python main.py -r t

* Содержит класс -> трейн датасет 
* модель
* функции для обучения
