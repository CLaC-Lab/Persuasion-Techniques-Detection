import sys
import getopt
import pandas as pd
import numpy as np

from labels import LABEL_COLUMNS

# may need to fix formatting bugs in original data files

def parse_args(argv):
   input_file = ''
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('preprocessing.py -i <input_file_path> -o <output_file_path>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('preprocessing.py -i <input_file_path> -o <output_file_path>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         input_file = arg
      elif opt in ("-o", "--ofile"):
         output_file = arg

   return input_file, output_file

def main(argv):
    input_file, output_file = parse_args(argv)

    df = pd.read_csv(input_file)
    for label in LABEL_COLUMNS:
        df[label] = np.where(df['labels'].str.contains(label), 1, 0)
    df = df.drop(columns="labels")

    df.to_csv(output_file)


if __name__ == "__main__":
   main(sys.argv[1])