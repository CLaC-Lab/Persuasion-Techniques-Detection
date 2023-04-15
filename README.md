# Persuasion-Techniques-Detection

## Setup

```
python3 -m pip install -r requirements.txt
```

## Preprocessing

### Expected File System

```
├── /Persuasion-Techniques-Detection
├── /semeval2023task3bundle-v4
│   ├── /data
│   ...
```

### Usage

```
python3 src/preprocessing.py
```

### Output

```
├── /Persuasion-Techniques-Detection
├── /semeval2023task3bundle-v4
│   ├── /data
│   ├── train_df_en.csv
|   ├── train_df_fr.csv
|   ├── train_df_ge.csv
|   ├── train_df_it.csv
|   ├── train_df_po.csv
|   ├── train_df_ru.csv
│   ├── dev_df_en.csv
|   ├── dev_df_fr.csv
|   ├── dev_df_ge.csv
|   ├── dev_df_it.csv
|   ├── dev_df_po.csv
|   ├── dev_df_ru.csv
```

## Augmentation

### Usage

```
python3 src/augmentation.py
```

### Output

```
├── /Persuasion-Techniques-Detection
├── /semeval2023task3bundle-v4
│   ├── /data
│   ├── train_df_en.csv
 ...
│   ├── dev_df_en.csv
  ...
│   ├── train_df_aug_en.csv
|   ├── train_df_aug_fr.csv
|   ├── train_df_aug_ge.csv
|   ├── train_df_aug_it.csv
|   ├── train_df_aug_po.csv
|   ├── train_df_aug_ru.csv
```

## Train and Predict Pipeline

### Usage

```
python3 src/bert_pipeline.py <TRAIN_LANG> <RUN_NUM>
```

Configure `MODEL_NAME` by hardcoding in `src/bert_pipeline.py` and `src/model.py` before run

#### Example

```
python3 src/bert_pipeline.py "en" "1"
```

### Output

```
├── /Persuasion-Techniques-Detection
├── /semeval2023task3bundle-v4
│   ├── /data
│   ├── train_df_en.csv
 ...
│   ├── dev_df_en.csv
 ...
│   ├── train_df_aug_en.csv
 ...
│   ├── /model_run
│   │   ├── /<MODEL_NAME>
│   │   │   ├── /trained_on_<TRAIN_LANG>
│   │   │   │   ├── /<RUN_NUM>
│   │   │   │   │   ├── <TRAIN_LANG>-pred-en.txt
│   │   │   │   │   ├── <TRAIN_LANG>-pred-fr.txt
│   │   │   │   │   ├── <TRAIN_LANG>-pred-ge.txt
│   │   │   │   │   ├── <TRAIN_LANG>-pred-it.txt
│   │   │   │   │   ├── <TRAIN_LANG>-pred-ru.txt
│   │   │   │   │   ├── <TRAIN_LANG>-pred-po.txt
```
