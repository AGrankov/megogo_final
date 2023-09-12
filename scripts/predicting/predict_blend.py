import sys
sys.path.append('..')
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from scripts.utils.utils import labeling_col, process_sequence
from scripts.utils.torch_train_utils import progress_bar
from scripts.models.movie_features_seq_model import MovieRNNModel
from scripts.models.movie_mlp import MovieMLPModel
from scripts.dataset.user_movie_additional_dataset import UserMovieAdditionalDataset
from scripts.dataset.movie_mlp_dataset import MovieMLPDataset

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True


def load_vectors(input_file):
    vectors = {}
    with open(input_file) as file:
        file.readline()
        for line in file:
            line_list = line.strip().split()
            if not line_list[0].isdigit():
                continue
            movie_id = int(line_list[0])
            vec = np.array([float(_) for _ in line_list[1:]], dtype=float)
            if not movie_id in vectors:
                vectors[movie_id] = vec
    return vectors


def main():
    parser = argparse.ArgumentParser(description='Predict meaned values from two models')

    parser.add_argument('-i', '--input', default='../p_input/train_data_full.csv', help='input data file name')
    parser.add_argument('-mi', '--meta-information', default='../p_input/video_meta_data_full.csv', help='meta information file name')
    parser.add_argument('-ss', '--sample-submission', default='../input/sample_submission_full.csv', help='sample submission file name')
    parser.add_argument('-v', '--vectors', default='../models/fasttext_skipgram_model.vec', help='path to pretrained vectors')
    parser.add_argument('-dim', '--dimension', default=64, help='dimention of embeddings')
    parser.add_argument('-sl', '--sequence-length', default=50, help='length of sequences')

    parser.add_argument('-mp1', '--model-path-1', default='../models/torch_movies_seq_add_model_v6/', help='first model file name')
    parser.add_argument('-mp2', '--model-path-2', default='../models/torch_movies_mlp_model_v1/', help='second model file name')
    parser.add_argument('-bs', '--batch-size', default=256, help='size of batches')

    parser.add_argument('-o', '--output', default='../submissions/movies_embed_v24_movies_mlp_v2.csv.gz', help='output submission file name')

    args = parser.parse_args()


    df = pd.read_csv(args.input, parse_dates=['session_start_datetime'])
    movie_id_trans = labeling_col(df, 'primary_video_id')

    ss = pd.read_csv(args.sample_submission)
    res_users = set(ss.user_id.unique())
    df = df[df.user_id.isin(res_users)].reset_index(drop=True)

    movies_sequences = process_sequence(df, 'primary_video_id')
    wp = process_sequence(df, 'watching_percentage')

    df_meta = pd.read_csv(args.meta_information)
    int_cols = [
        'year', 'rating_imdb', 'rating_kinopoisk', 'score_by_popular',
        'score_by_recommended', 'age_limit', 'video_duration'
    ]
    ohe_cols = [
        'genres',
        'type',
        'country',
        'quality'
    ]
    ohe_cols_size = [50, 10, 72, 4]

    vectors = load_vectors(args.vectors)

    movies_dict = dict()
    for name, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        processed = []
        processed.append(row[int_cols].values)
        for oidx, oc in enumerate(ohe_cols):
            ar = np.zeros((ohe_cols_size[oidx]), dtype=np.uint8)
            if type(row[oc]) == str:
                ar[np.array([int(x) for x in row[oc].split(',')])] = 1
            if type(row[oc]) == int:
                ar[row[oc]] = 1
            processed.append(ar)
        if name in vectors:
            processed.append(vectors[name])
        else:
            processed.append(np.zeros((args.dimension), dtype=np.float32))
        movies_dict[name+1] = np.concatenate(processed)

    ohe_cols_size.append(args.dimension)

    testing_seq = []
    testing_additional = []

    wp_all = np.zeros((len(movies_sequences), len(movie_id_trans.classes_)), dtype=np.float16)

    def make_glob_from_data(seq, seq_data):
        res = np.zeros((len(movie_id_trans.classes_)), dtype=np.float16)
        for idx, sv in enumerate(seq):
            res[sv] += seq_data[idx]
        return res

    for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
        wp_all[si] = make_glob_from_data(seq, wp[si])

        ts = [x+1 for x in seq[-args.sequence_length:]]
        wts = wp[si][-args.sequence_length:]

        if len(ts) < args.sequence_length:
            ts = [0] * (args.sequence_length-len(ts)) + ts
            wts = [0] * (args.sequence_length-len(wts)) + wts

        adds = np.hstack((np.reshape(wts, (-1, 1)),))

        testing_seq.append(ts)
        testing_additional.append(adds)

    testing_seq = np.array(testing_seq)
    testing_additional = np.array(testing_additional)


    testset1 = UserMovieAdditionalDataset(testing_seq,
                                            testing_additional,
                                            movies_dict)

    model1 = MovieRNNModel(
        seq_len=args.sequence_length,
        features_dim=testset1[0].shape[1],
        result_classes=len(movie_id_trans.classes_),
        rnn_units=256,
        mid_dim=2048,
        mid_dim2=2048,
        pool_count=5,
        num_layers=2
    ).cuda()

    checkpoint = torch.load(os.path.join(args.model_path_1, "best_model_chkpt.t7"))
    model1.load_state_dict(checkpoint['net'])
    model1.eval()


    testset2 = MovieMLPDataset(wp_all)

    model2 = MovieMLPModel(
        features_dim=testset2[0].shape[0],
        result_classes=len(movie_id_trans.classes_),
        mid_dim=4096,
        mid_dim2=4096
    ).cuda()

    checkpoint = torch.load(os.path.join(args.model_path_2, "best_model_chkpt.t7"))
    model2.load_state_dict(checkpoint['net'])
    model2.eval()


    batches_count = len(testset1) // args.batch_size + 1

    test_predictions = []
    with torch.no_grad():
        for bidx in tqdm(range(batches_count)):
            inp1_arr = []
            inp2_arr = []
            for sidx in range(args.batch_size):
                if bidx*args.batch_size+sidx < len(testset1):
                    inp1_arr.append(testset1[bidx*args.batch_size+sidx])
                    inp2_arr.append(testset2[bidx*args.batch_size+sidx])

            input1 = torch.Tensor(np.array(inp1_arr)).cuda()
            y_pred_1 = model1(input1)

            input2 = torch.Tensor(np.array(inp2_arr)).cuda()
            y_pred_2 = model2(input2)

            y_pred = torch.mean(torch.stack([y_pred_1, y_pred_2]), dim=0)
            sort_preds, sort_indices = y_pred.sort(dim=1)
            sort_indices = sort_indices[:, -10:]
            test_predictions.extend(sort_indices.cpu().data.numpy()[:, ::-1])
    test_predictions = np.array(test_predictions)

    result_preds = []
    for idx in tqdm(range(len(test_predictions))):
        reversed = movie_id_trans.inverse_transform(test_predictions[idx])
        result_preds.append(' '.join([str(j) for j in reversed]))

    result_users = sorted(df.user_id.unique())
    predicted_df = pd.DataFrame({'user_id': result_users,
                                'primary_video_id': result_preds})

    empty_tensor = torch.Tensor(np.zeros((1,) + testset1[0].shape)).cuda()
    empty_predict1 = model1(empty_tensor)
    empty_tensor = torch.Tensor(np.zeros((1,) + testset2[0].shape)).cuda()
    empty_predict2 = model2(empty_tensor)
    empty_predict = torch.mean(torch.stack([empty_predict1, empty_predict2]), dim=0)
    _, sort_indices = empty_predict.sort(dim=1)
    sort_indices = sort_indices[:, -10:]
    res_empty = sort_indices.cpu().data.numpy()[:, ::-1]
    res_empty = ' '.join([str(j) for j in res_empty[0]])

    res_df = ss.copy()
    res_df = res_df.merge(predicted_df, how='left', on='user_id')
    res_df.loc[pd.isna(res_df['primary_video_id_y']), 'primary_video_id_y'] = res_empty
    res_df.drop('primary_video_id_x', 1, inplace=True)
    res_df.rename(columns={'primary_video_id_y': 'primary_video_id'}, inplace=True)

    res_df.to_csv(args.output, compression='gzip', index=False)
    # os.system('kaggle competitions submit megogochallenge -f {} -m "{}"'.format(args.output, 'stack'))


if __name__ == '__main__':
    main()
