import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd
import argparse
from scripts.utils.utils import labeling_col, normalize_int_col

train_filename = 'train_data_full.csv'
video_meta_filename = 'video_meta_data_full.csv'
video_genres_filename = 'video_genres_data_full.csv'


def main():
    parser = argparse.ArgumentParser(description='Preprocess initial dataset')

    parser.add_argument('-i', '--input', default='../input/', help='input data path')
    parser.add_argument('-o', '--output', default='../p_input/', help='output data path')

    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.input, train_filename))

    cols_for_drop = [
        'user_ip',
        'vod_type',
        'player_position_min',
        'player_position_max',
        'time_cumsum_max',
        'session_duration',
        'device_type',
        'device_os'
    ]

    train_df.drop(cols_for_drop, 1, inplace=True)
    train_df['session_start_datetime'] = pd.to_datetime(train_df['session_start_datetime'])

    sub_df = train_df[['video_id', 'primary_video_id', 'video_duration']]
    sub_df = sub_df.drop_duplicates(['video_id', 'primary_video_id'])

    train_df.drop('video_duration', 1, inplace=True)
    train_df.drop('video_id', 1, inplace=True)

    train_df = train_df.sort_values('session_start_datetime').reset_index(drop=True)
    train_df.to_csv(os.path.join(args.output, train_filename), index=False)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    vid_meta_df = pd.read_csv(os.path.join(args.input, video_meta_filename))

    cols_for_drop = [
        'title',
        'imdb_id',
        'kinopoisk_id',
        'mpaa_rating' # same as age_limit
    ]

    vid_meta_df.drop(cols_for_drop, 1, inplace=True)

    vid_meta_df = vid_meta_df.merge(sub_df, how='left', on=['video_id', 'primary_video_id'])
    vid_meta_df['video_duration'] = vid_meta_df['video_duration'].fillna(vid_meta_df['video_duration'].mean())

    mean_for_cols = [
        'year',
        'rating_imdb',
        'rating_kinopoisk',
        'score_by_popular',
        'score_by_recommended',
        'age_limit'
    ]

    mean_map_dict = dict()
    for col in mean_for_cols:
        mean_map_dict[col] = vid_meta_df[['primary_video_id', col]].groupby('primary_video_id')[col].mean().to_dict()

    vid_meta_df.drop('video_id', 1, inplace=True)
    vid_meta_df = vid_meta_df.drop_duplicates('primary_video_id')
    for col in mean_for_cols:
        vid_meta_df[col] = vid_meta_df['primary_video_id'].map(mean_map_dict[col])


    labeling_col(vid_meta_df, 'type')
    labeling_col(vid_meta_df, 'country')
    labeling_col(vid_meta_df, 'quality')

    vid_meta_df['year'] = vid_meta_df['year'] - vid_meta_df['year'].min()
    vid_meta_df['video_duration'] = np.log1p(vid_meta_df['video_duration'])
    vid_meta_df['score_by_recommended'] = np.log1p(vid_meta_df['score_by_recommended'])
    vid_meta_df['score_by_popular'] = np.log1p(vid_meta_df['score_by_popular'])
    cols_for_normalize = [
        'year',
        'rating_imdb',
        'rating_kinopoisk',
        'score_by_popular',
        'score_by_recommended',
        'age_limit',
        'video_duration',
    ]

    for col in cols_for_normalize:
        normalize_int_col(vid_meta_df, col)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    vid_genres_df = pd.read_csv(os.path.join(args.input, video_genres_filename))
    labeling_col(vid_genres_df, 'title')

    vid_genres_dict = vid_genres_df.groupby('primary_video_id').agg({
    'title': (lambda x: ",".join([str(j) for j in x]))
    }).to_dict()['title']

    vid_meta_df['genres'] = vid_meta_df['primary_video_id'].map(vid_genres_dict)
    vid_meta_df.to_csv(os.path.join(args.output, video_meta_filename), index=False)


if __name__ == '__main__':
    main()
