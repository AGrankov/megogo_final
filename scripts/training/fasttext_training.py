import sys
sys.path.append('..')
import os
import argparse
import fasttext
import pandas as pd
from scripts.utils.utils import labeling_col, process_sequence


def main():
    parser = argparse.ArgumentParser(description='Train fasttext model')

    parser.add_argument('-i', '--input', default='../p_input/train_data_full.csv', help='input data file name')
    parser.add_argument('-m', '--model-name', default='../models/fasttext_skipgram_model', help='name of trained model')
    parser.add_argument('-s', '--sequences-filename', default='../p_input/movies_sequences.txt', help='file name of processed sequences')
    parser.add_argument('-dim', '--dimension', default=64, help='dimention of embeddings')
    parser.add_argument('-ws', '--window-size', default=5, help='window size for skipgram algo')
    parser.add_argument('-e', '--epochs', default=50, help='amount of epochs')
    parser.add_argument('-mc', '--min-count', default=5, help='min count of examples')
    parser.add_argument('-neg', '--neg', default=100, help='negative examples count')
    parser.add_argument('-is', '--is-silent', default=0, help='is silent')

    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=['session_start_datetime'])
    labeling_col(df, 'primary_video_id')
    movies_sequences = process_sequence(df, 'primary_video_id')

    with open(args.sequences_filename, 'w') as file:
        for seq in movies_sequences:
            file.write(' '.join([str(x) for x in seq]) + os.linesep)

    fasttext.skipgram(args.sequences_filename,
                      args.model_name,
                      dim=args.dimension,
                      ws=args.window_size,
                      epoch=args.epochs,
                      min_count=args.min_count,
                      neg=args.neg,
                      silent=args.is_silent
                      )


if __name__ == '__main__':
    main()
