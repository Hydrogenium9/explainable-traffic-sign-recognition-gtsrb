# Explainable Traffic Sign Recognition — Streamlit Web Application

Готовое локальное приложение для классификации дорожных знаков GTSRB с помощью:

- Custom CNN;
- EfficientNetV2B0;
- ConvNeXtTiny.

Приложение показывает предсказанный класс, confidence, Top-3 результата и Grad-CAM explanation. По умолчанию выбрана ConvNeXtTiny, поскольку она получила лучшую точность и устойчивость в проведённых экспериментах.

## Структура

```text
Traffic_Sign_Web_App
├── app.py
├── class_names.py
├── config.py
├── gradcam.py
├── image_processing.py
├── metrics_loader.py
├── model_loader.py
├── prediction.py
├── styles.py
├── validate_installation.py
├── requirements.txt
├── requirements_without_tensorflow.txt
├── setup_app.bat
├── run_app.bat
├── run_app.ps1
├── models
├── results
├── samples
└── .streamlit
```

## Быстрый запуск на Windows

### Вариант 1 — отдельное окружение приложения

1. Распакуйте архив.
2. Запустите `setup_app.bat`.
3. Дождитесь установки TensorFlow и остальных библиотек.
4. После успешной проверки запустите `run_app.bat`.
5. Браузер обычно откроется автоматически по адресу `http://localhost:8501`.

Первая установка может занять продолжительное время, потому что TensorFlow — большой пакет.

### Вариант 2 — существующее окружение проекта

Если в родительской папке уже существует `.venv` с TensorFlow и Streamlit, `run_app.bat` попытается использовать его автоматически.

Для ручной установки недостающих библиотек без повторной установки TensorFlow:

```powershell
python -m pip install -r requirements_without_tensorflow.txt
python -m streamlit run app.py
```

## Проверка перед запуском

```powershell
python validate_installation.py
```

Проверка загружает все три модели и выполняет одно предсказание на первом sample image.

## Использование

1. На странице **Classifier** выберите модель.
2. Загрузите PNG, JPG, JPEG или PPM либо выберите bundled sample.
3. Просмотрите основной класс, confidence и Top-3.
4. Нажмите **Generate Grad-CAM**.
5. Сравните исходное изображение, heatmap и overlay.
6. На странице **Model evidence** доступны итоговые таблицы clean test, robustness, Grad-CAM и SHAP.

## Preprocessing

- Custom CNN получает вход в диапазоне `[0, 1]`.
- EfficientNetV2B0 и ConvNeXtTiny получают вход в диапазоне `[0, 255]`, поскольку preprocessing включён в сохранённые модели.
- Все изображения приводятся к RGB и размеру `224 × 224`.

## Ограничения

Это исследовательский прототип. Он не предназначен для управления автомобилем, навигации или принятия safety-critical решений. Модели обучены только на 43 классах GTSRB и могут ошибаться на знаках других стран, изображениях вне датасета, частично видимых знаках и при сильном domain shift.

## Windows и GPU

Native Windows TensorFlow запускается на CPU. Для TensorFlow GPU в современных версиях рекомендуется WSL2. Для локальной демонстрации приложения GPU не обязателен; ConvNeXtTiny просто будет загружаться и вычисляться медленнее.
