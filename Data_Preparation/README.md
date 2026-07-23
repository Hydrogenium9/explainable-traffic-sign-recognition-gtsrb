# Шаг 4: подготовка изображений и tf.data pipeline

## Что делает код

1. Читает `train_manifest.csv`, `validation_manifest.csv` и `Test.csv`.
2. Проверяет пути и метки.
3. Загружает изображения как RGB.
4. Изменяет размер до 224×224.
5. Применяет аугментацию только к training.
6. Нормализует пиксели.
7. Создаёт batched `tf.data.Dataset`.
8. Рассчитывает class weights.
9. Создаёт контрольные PDF-фигуры.
10. Проверяет форму, диапазон пикселей и метки.

Исходные изображения не изменяются и не копируются.

## Куда положить

Создайте папку:

`D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project\Codes\Data_Preparation`

Скопируйте туда все файлы из архива.

## Предварительное условие

Должны существовать:

- `Codes\Data_Cleaning\prepared_data\train_manifest.csv`
- `Codes\Data_Cleaning\prepared_data\validation_manifest.csv`
- `Datasets\traffic_signs\Test.csv`

## Рекомендуемый Python

Python 3.11.

Проверка установленных версий:

```powershell
py -0p
```

## Автоматический запуск

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

cd "D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project\Codes\Data_Preparation"

.\run_data_preparation.ps1
```

## Ручной запуск

```powershell
cd "D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project"

py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

cd "Codes\Data_Preparation"

python -m pip install --upgrade pip
pip install -r requirements.txt
python prepare_and_verify_data.py
```

## Результаты

Появится папка:

`Codes\Data_Preparation\outputs`

### Reports

- `data_pipeline_report.txt`
- `data_config.json`
- `class_weights.json`
- `class_counts_by_split.csv`

### Figures

- `augmentation_comparison.pdf`
- `training_batch_after_augmentation.pdf`
- `validation_batch_preprocessed.pdf`

Также создаются PNG-копии для быстрого просмотра.

## Использование позже в модели

```python
from config import DataConfig
from data_pipeline import build_all_datasets

config = DataConfig()
data = build_all_datasets(config)

train_ds = data["train_dataset"]
validation_ds = data["validation_dataset"]
test_ds = data["test_dataset"]
class_weights = data["class_weights"]
```

При обучении Custom CNN:

```python
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=30,
    class_weight=class_weights,
)
```

Метки остаются целыми числами 0..42, поэтому позже следует использовать
`SparseCategoricalCrossentropy`.
