# Data Cleaning для GTSRB

## Куда положить файлы

Скопируйте:

- `inspect_and_clean_dataset.py`
- `requirements.txt`

в папку:

`D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project\Codes\Data_Cleaning`

## Установка

Откройте PowerShell или Terminal:

```powershell
cd "D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project\Codes\Data_Cleaning"

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если PowerShell запрещает активацию, выполните:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Запуск

Полная проверка, включая SHA-256 дубликаты:

```powershell
python inspect_and_clean_dataset.py
```

Более быстрый запуск без поиска точных дубликатов:

```powershell
python inspect_and_clean_dataset.py --skip-hashes
```

## Результаты

Скрипт создаст:

`D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project\Codes\Data_Cleaning\outputs`

Основные файлы:

- `SUMMARY.txt`
- `reports\dataset_structure.csv`
- `reports\image_inventory.csv`
- `reports\corrupt_or_unreadable_images.csv`
- `reports\csv_consistency_issues.csv`
- `reports\train_class_distribution.csv`
- `reports\duplicate_groups.csv`
- `reports\cross_split_duplicates.csv`
- `reports\ignored_non_image_files.csv`
- `figures\train_class_distribution.pdf`
- `figures\image_size_distribution.pdf`
- `figures\sample_images_by_class.pdf`
- `figures\challenging_image_examples.pdf`

Скрипт работает в безопасном режиме и не удаляет изображения.
