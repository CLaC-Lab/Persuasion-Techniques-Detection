# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/128oFNIxYrCIl0pQWJlOfDe7Z356NclKD
"""

import os
import sys
import subprocess

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


# ----------   ------- ### CONFIG ### -----  ----------- #

MODEL_NAME = "roberta-base"

# training config
MAX_TOKEN_COUNT = 300
BATCH_SIZE = 8
N_EPOCHS = 10

# ----------   ---------------   -------------  ----------- #


# utility function
def make_dir_if_none(filepath):
  try:
    os.mkdir(filepath)
  except:
    pass

def load_train_data(TRAIN_LANG):
  """
  load training data, by language
  """
  input_train_file = os.path.join("..", "semeval2023task3bundle-v4", f"train_df_aug_{TRAIN_LANG}.csv")
  return pd.read_csv(input_train_file)

def train_model(train_split, valid_df):

  # drop nones
  print("DROPPING NONES")
  train_df = train_split[~(train_split[LABEL_COLUMNS]==0).all(axis=1)]

  # initialize hyper parameters
  steps_per_epoch=len(train_df) // BATCH_SIZE
  total_training_steps = steps_per_epoch * N_EPOCHS
  warmup_steps = total_training_steps // 5

  # initialize model
  print("LOADING INITIAL MODEL")
  model = BERTTagger(
    n_classes=len(LABEL_COLUMNS),
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps,
    MODEL_NAME=MODEL_NAME
  )

  # initialize text tokenizer
  print("LOADING TOKENIZER")
  tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)


  # initialize data module
  print("LOADING DATAMODULE")
  data_module = TextDataModule(
    train_df,
    valid_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
  )

  # initialize checkpoint configuration
  checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    )

  # initialize logger
  logger = TensorBoardLogger("lightning_logs", name="persuasive-text")

  # initialize early stop conditions
  early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=2)

  # initialize trainer
  print("LOADING TRAINER")
  trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=30
  )

  # train model
  print("TRAINING")
  trainer.fit(model, data_module)

  # pull model from best checkpoint
  trained_model = BERTTagger.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    n_classes=len(LABEL_COLUMNS)
  )

  # set model params to eval mode
  trained_model.eval()
  trained_model.freeze()

  # return optimal, trained model, ready for prediction
  return trained_model

def get_prediction_file_path(model_run__pred_path, LANG, lang_to_pred):
  return os.path.join(model_run__pred_path, f"{LANG}-pred-{lang_to_pred}.txt")

def predict(trained_model, LANG, model_run__base_path):

  # set reference to device to load model
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # move model to device
  trained_model = trained_model.to(device)

  # iterate over each language to predict
  langs_to_pred = ["en", "fr", "ge", "it", "po", "ru"]
  for lang_to_pred in langs_to_pred:

    # set path of prediction of given language, for model run
    model_run__pred_path = f"{model_run__base_path}/{lang_to_pred}"

    make_dir_if_none(model_run__pred_path)

    # initialize test set path to read
    test_set_path = os.path.join("..", "semeval2023task3bundle-v4", f"dev_df_{lang_to_pred}")
    test_df = pd.read_csv(test_set_path)

    # set all test set labels to 0
    for col in LABEL_COLUMNS:
      test_df[col].values[:] = 0

    # initialize tokenizer for test set data
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    # initialize test set data module
    dataset = TextDataset(
      test_df.copy(),
      tokenizer,
      max_token_len=MAX_TOKEN_COUNT
    )

    # initialize initial model predictions list
    predictions = []

    # iterate over test set batches
    # (tqdm is just for the progress bar visual)
    for item in tqdm(dataset):

      # predict on batch of test set
      _, prediction = trained_model(
        item["input_ids"].unsqueeze(dim=0).to(device),
        item["attention_mask"].unsqueeze(dim=0).to(device)
      )

      # append predictions to list
      # flatten to excepted list shape
      predictions.append(prediction.flatten())

    # move initial predictions list to cpu
    predictions = torch.stack(predictions).detach().cpu()

    # set prediction threshold and output values
    THRESHOLD = 0.5
    upper, lower = 1, 0

    # convert initial predictions to 0/1
    predictions = np.where(predictions > THRESHOLD, upper, lower)

    # add label columns to final predictions
    final_preds_df = pd.DataFrame(predictions, columns=LABEL_COLUMNS)

    # re-add article, paragraph fields to predictions from original test set, for scorer
    final_preds_df = pd.concat([test_df[['article', 'paragraph']].reset_index(drop=True), final_preds_df], axis=1)

    # convert final predictions back to (initial) expected format of scorer
    for label in LABEL_COLUMNS:
      final_preds_df[label] = np.where(final_preds_df[label] == 1, label, "")

    # rename "labels" to "tags", expected by scorer
    final_preds_df["tags"] = final_preds_df[LABEL_COLUMNS].apply(lambda row: ','.join(row[row != ""].astype(str)), axis=1)
    final_preds_df = final_preds_df.drop(columns=LABEL_COLUMNS)

    # write predictions to disk
    pred_path = get_prediction_file_path(model_run__pred_path, LANG, lang_to_pred)
    final_preds_df.to_csv(pred_path, sep ='\t', header=False, index=False)

def run(TRAIN_LANG, RUN_NUM):

  print("LOADING DATA")
  # load test data
  df = load_train_data(TRAIN_LANG)

  # split to train and test
  train_split_df, valid_split_df = train_test_split(df, test_size=0.2)

  # train model
  print("TRAINING MODEL")
  trained_model = train_model(train_split_df, valid_split_df)

  # set base path directory for model run
  # (given model trained on a given language)
  model_run__base_path = os.path.join("model_run", MODEL_NAME, f"trained_on_{TRAIN_LANG}", RUN_NUM)
  make_dir_if_none(model_run__base_path)

  # run predictions with trained model
  predict(trained_model, TRAIN_LANG, model_run__base_path)

  # run scorer
  score(model_run__base_path, TRAIN_LANG)

def get_gold_label_file_path(lang_to_pred):
   return os.path.join("..", "semeval2023task3bundle-v4", "data", lang_to_pred, "dev-labels-subtask-3.txt")

def score(model_run__base_path, TRAIN_LANG):
  SCORER = os.path.join("..", "semeval2023task3bundle-v4", "scorers", "scorer-subtask-3.py")
  LABELS_FILE = os.path.join("..", "semeval2023task3bundle-v4", "scorers", "techniques_subtask3.txt")

  langs_to_pred = ["en", "fr", "ge", "it", "po", "ru"]
  for lang_to_pred in langs_to_pred:
    gold_labels_path = get_gold_label_file_path(lang_to_pred)
    prediction_file_path = get_prediction_file_path(model_run__base_path, TRAIN_LANG, lang_to_pred)

    print(f"""
    RUNNING SCORER:
    MODEL_NAME=[{MODEL_NAME}]
    TRAIN_LANG=[{TRAIN_LANG}]
    RUN_NUM=[{RUN_NUM}]
    LANG_TO_PRED=[{lang_to_pred}]
    """)

    subprocess.run([f"python3 {SCORER} -g {gold_labels_path} -p {prediction_file_path} -f {LABELS_FILE}"])

if __name__ == "__main__":
  TRAIN_LANG = None
  RUN_NUM = None

  try:
    TRAIN_LANG = sys.argv[1]
    RUN_NUM = sys.argv[2]
  except:
    if (TRAIN_LANG is None):
      print("enter TRAIN_LANG")
    if (RUN_NUM is None):
      print("enter RUN_NUM")

    sys.exit()


  print(f"""
  TRAINING:
  MODEL_NAME=[{MODEL_NAME}]
  TRAIN_LANG=[{TRAIN_LANG}]
  RUN_NUM=[{RUN_NUM}]
  """)

  run(TRAIN_LANG, RUN_NUM)