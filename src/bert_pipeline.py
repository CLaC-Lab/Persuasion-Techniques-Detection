# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/128oFNIxYrCIl0pQWJlOfDe7Z356NclKD
"""


import numpy as np
import pandas as pd

import transformers
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from model import BERTTagger
from data_model import TextDataModule, TextDataset
from labels import LABEL_COLUMNS

DATA_PATH = "drive/MyDrive/BERT/new-data"

### CONFIG ###
# ----------   ---------------   -------------  -----------
# file system config
MODEL = "roberta-base"
ID = "new-aug-drop-over-tra-en"
LANG = "en"

# training config
MAX_TOKEN_COUNT = 300
BATCH_SIZE = 8
N_EPOCHS = 10
# ----------   ---------------   -------------  -----------


def load_data():
  """
  load training data, by language
  """
  return pd.read_csv(f"{DATA_PATH}/{ID}/{LANG}/{LANG}.csv")


def train_model(train_split, valid_df):

  tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
  train_df = train_split[~(train_split[LABEL_COLUMNS]==0).all(axis=1)]

  steps_per_epoch=len(train_df) // BATCH_SIZE
  total_training_steps = steps_per_epoch * N_EPOCHS
  warmup_steps = total_training_steps // 5


  model = BERTTagger(
    n_classes=len(LABEL_COLUMNS),
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps
  )

  data_module = TextDataModule(
    train_df,
    valid_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
  )

  checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    )

  logger = TensorBoardLogger("lightning_logs", name="persuasive-text")

  early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=2)

  trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=30
  )

  trainer.fit(model, data_module)

  trained_model = BERTTagger.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    n_classes=len(LABEL_COLUMNS)
  )
  trained_model.eval()
  trained_model.freeze()

  return trained_model



def run(run_num):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model_run_path = f"{DATA_PATH}/{ID}/{LANG}/model/{MODEL}/{run_num}"

  df = load_data()
  train_split_df, valid_split_df = train_test_split(df, test_size=0.2)

  trained_model = train_model(train_split_df, valid_split_df)
  trained_model = trained_model.to(device)


  langs_to_pred = ["en", "fr", "ge", "it", "po", "ru"]
  for lang_to_pred in langs_to_pred:

    model_run_pred_path = f"{model_run_path}/{lang_to_pred}"
    gold_path = f"{model_run_pred_path}/{LANG}-gold-{lang_to_pred}.txt"
    pred_path = f"{model_run_pred_path}/{LANG}-pred-{lang_to_pred}.txt"

    test_df_gold = pd.read_csv(f"{DATA_PATH}/dev-sets-en/{lang_to_pred}-dev-sets-en.csv")

    for label in LABEL_COLUMNS:
      test_df_gold[label] = np.where(test_df_gold[label] == 1, label, "")
    test_df_gold["tags"] = test_df_gold[LABEL_COLUMNS].apply(lambda row: ','.join(row[row != ""].astype(str)), axis=1)

    test_df_gold = test_df_gold.drop(columns=LABEL_COLUMNS + ["text"])
    test_df_gold.to_csv(gold_path, sep ='\t', header=False, index=False)
    test_df = pd.read_csv(f"{DATA_PATH}/dev-sets-en/{lang_to_pred}-dev-sets-en.csv")

    for col in LABEL_COLUMNS:
      test_df[col].values[:] = 0

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    dataset = TextDataset(
      test_df.copy(),
      tokenizer,
      max_token_len=MAX_TOKEN_COUNT
    )

    predictions = []

    for item in tqdm(dataset):
      _, prediction = trained_model(
        item["input_ids"].unsqueeze(dim=0).to(device),
        item["attention_mask"].unsqueeze(dim=0).to(device)
      )

      predictions.append(prediction.flatten())

    predictions = torch.stack(predictions).detach().cpu()

    THRESHOLD = 0.5
    upper, lower = 1, 0
    predictions = np.where(predictions > THRESHOLD, upper, lower)

    final_preds_df = pd.DataFrame(predictions, columns=LABEL_COLUMNS)
    final_preds_df = pd.concat([test_df[['article', 'paragraph']].reset_index(drop=True), final_preds_df], axis=1)



    for label in LABEL_COLUMNS:
      final_preds_df[label] = np.where(final_preds_df[label] == 1, label, "")
    final_preds_df["tags"] = final_preds_df[LABEL_COLUMNS].apply(lambda row: ','.join(row[row != ""].astype(str)), axis=1)

    final_preds_df = final_preds_df.drop(columns=LABEL_COLUMNS)

    final_preds_df.to_csv(pred_path, sep ='\t', header=False, index=False)

"""### Results"""

SCORER = "drive/MyDrive/BERT/data/scorer-subtask-3.py"
LABELS_FILE = "drive/MyDrive/BERT/data/techniques_subtask3.txt"

def get_gold(lang_to_pred, run_num):
  path = f"{DATA_PATH}/{ID}/{LANG}/model/{MODEL}/{run_num}/{lang_to_pred}"
  return f"{path}/{LANG}-gold-{lang_to_pred}.txt"

def get_pred(lang_to_pred, run_num):
  path = f"{DATA_PATH}/{ID}/{LANG}/model/{MODEL}/{run_num}/{lang_to_pred}"
  return f"{path}/{LANG}-pred-{lang_to_pred}.txt"


RUN_NUM = 1
run(RUN_NUM)

# !echo $RUN_NUM
# !echo $MODEL

# PRED_LANG = "en"
# GOLD = get_gold(PRED_LANG, RUN_NUM)
# PRED = get_pred(PRED_LANG, RUN_NUM)

# !echo $LANG "pred" $PRED_LANG
# !python3 {SCORER} -g {GOLD} -p {PRED} -f {LABELS_FILE}