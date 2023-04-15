import os
import pandas as pd
import numpy as np

from labels import LABEL_COLUMNS


def read_process_write(split, lang):

   labels_input_file = os.path.join("..", "semeval2023task3bundle-v4", "data", lang, f"{split}-labels-subtask-3.txt")
   text_input_file = os.path.join("..", "semeval2023task3bundle-v4", "data", lang, f"{split}-labels-subtask-3.template")

   df_labels = pd.read_csv(labels_input_file, sep='\t',
                            header=None,
                            names={'article': int, 'paragraph': int, 'labels': str},
                            on_bad_lines='skip')
   df_text = pd.read_csv(text_input_file, sep='\t',
                          header=None,
                          names={'article': int, 'paragraph': int, 'text': str},
                          on_bad_lines='skip')

   df_labels['labels'] = np.where(df_labels['labels'].isna(), "", df_labels['labels'])

   df = df_text.merge(df_labels, how='inner', on=['article', 'paragraph'])

   for label in LABEL_COLUMNS:
      df[label] = np.where(df['labels'].str.contains(label), 1, 0)
   df = df.drop(columns="labels")

   output_file = os.path.join("..", "semeval2023task3bundle-v4", f"{split}_df_{lang}.csv")
   df.to_csv(output_file, index=False)

def main():

   LANGS = ['en', 'fr', 'ge', 'it', 'po', 'ru']

   for lang in LANGS:
      print(lang)
      read_process_write("train", lang)
      read_process_write("dev", lang)

if __name__ == "__main__":
   main()