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
from scripts.dataset.user_movie_additional_dataset import UserMovieAdditionalDataset

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
    parser = argparse.ArgumentParser(description='Train sequence model based on pretrained Item2Vec and videos meta information')

    parser.add_argument('-i', '--input', default='../p_input/train_data_full.csv', help='input data file name')
    parser.add_argument('-mi', '--meta-information', default='../p_input/video_meta_data_full.csv', help='meta information file name')
    parser.add_argument('-v', '--vectors', default='../models/fasttext_skipgram_model.vec', help='path to pretrained vectors')
    parser.add_argument('-dim', '--dimension', default=64, help='dimention of embeddings')
    parser.add_argument('-sl', '--sequence-length', default=50, help='length of sequences')
    parser.add_argument('-sm', '--sequence-min', default=3, help='minimum sequence length')
    parser.add_argument('-vc', '--valid-count', default=2, help='validation example count per user')

    parser.add_argument('-mp', '--model-path', default='../models/', help='path to models directory')
    parser.add_argument('-m', '--model-name', default='movies_seq_add_model_v6', help='name of trained model')
    parser.add_argument('-bs', '--batch-size', default=4096, help='size of batches')
    parser.add_argument('-lr', '--learning-rate', default=0.001, help='learning rate')
    parser.add_argument('-e', '--epochs', default=8, help='epochs count')

    args = parser.parse_args()


    df = pd.read_csv(args.input, parse_dates=['session_start_datetime'])
    movie_id_trans = labeling_col(df, 'primary_video_id')

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

    training_seq = []
    training_additional = []
    training_targets = []

    validation_seq = []
    validation_additional = []
    validation_targets = []

    for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
        for idx in range(args.sequence_min, len(seq)):
            if wp[si][idx] > 0.5:
                ts = [x+1 for x in seq[idx-args.sequence_length:idx]]
                wts = wp[si][idx-args.sequence_length:idx]
                if len(ts) < args.sequence_length:
                    ts = [0] * (args.sequence_length-len(ts)) + ts
                    wts = [0] * (args.sequence_length-len(wts)) + wts
                adds = np.hstack((np.reshape(wts, (-1, 1)),))

                if idx >= len(seq)-args.valid_count:
                    validation_seq.append(ts)
                    validation_additional.append(adds)
                    validation_targets.append(seq[idx])
                else:
                    training_seq.append(ts)
                    training_additional.append(adds)
                    training_targets.append(seq[idx])

    training_seq = np.array(training_seq)
    validation_seq = np.array(validation_seq)

    training_additional = np.array(training_additional)
    validation_additional = np.array(validation_additional)

    training_targets = np.array(training_targets)
    validation_targets = np.array(validation_targets)


    model_path = os.path.join(args.model_path, "torch_{}".format(args.model_name))

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    trainset = UserMovieAdditionalDataset(training_seq,
                                          training_additional,
                                          movies_dict,
                                          training_targets)

    validset = UserMovieAdditionalDataset(validation_seq,
                                          validation_additional,
                                          movies_dict,
                                          validation_targets)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=6,
                                              pin_memory=True)

    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    model = MovieRNNModel(
        seq_len=args.sequence_length,
        features_dim=trainset[0][0].shape[1],
        result_classes=df.primary_video_id.unique().shape[0],
        rnn_units=256,
        mid_dim=2048,
        mid_dim2=2048,
        pool_count=5,
        num_layers=2
    ).cuda()

    best_loss = np.finfo(np.float32).max

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.6)

    def train(epoch, model, trainloader, log_file, optimizer, criterion):
        '''
        Train function for each epoch
        '''
        model.train()
        train_loss = 0

        print('Training Epoch {} optimizer LR {}'.format(epoch, optimizer.param_groups[0]['lr']))

        for batch_idx, (seq_data, targets) in enumerate(trainloader):
            seq_data = seq_data.cuda()
            targets = targets.long().cuda()

            outputs = model(seq_data)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_batch_loss = loss.item()
            train_loss += curr_batch_loss

            log_file.write('train,{epoch},'\
                           '{batch},{loss:.8f}\n'.format(epoch=epoch,
                                                         batch=batch_idx,
                                                         loss=curr_batch_loss))
            progress_bar(batch_idx,
                         len(trainloader),
                         'Loss: {l:.8f}'.format(l = train_loss/(batch_idx+1)))

    def validate(epoch, model, validloader, log_file, criterion, best_loss):
        '''
        Validate function for each epoch
        '''
        print('\nValidate Epoch: %d' % epoch)
        model.eval()
        eval_loss = 0

        with torch.no_grad():
            for batch_idx, (seq_data, targets) in enumerate(validloader):
                seq_data = seq_data.cuda()
                targets = targets.long().cuda()

                outputs = model(seq_data)
                loss = criterion(outputs, targets)

                curr_batch_loss = loss.item()
                eval_loss += curr_batch_loss

                log_file.write('valid,{epoch},'\
                               '{batch},{loss:.8f}\n'.format(epoch=epoch,
                                                             batch=batch_idx,
                                                             loss=curr_batch_loss))
                progress_bar(batch_idx,
                             len(validloader),
                             'Loss: {l:.8f}'.format(l = eval_loss/(batch_idx+1)))

        if eval_loss < best_loss:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'loss': eval_loss,
                'epoch': epoch,
            }
            session_checkpoint = os.path.join(model_path, "best_model_chkpt.t7")
            torch.save(state, session_checkpoint)
            best_loss = eval_loss

        return best_loss

    try:
        log_file_path = os.path.join(args.model_path, "{}.log".format(args.model_name))
        log_file = open(log_file_path, 'w')
        log_file.write('type,epoch,batch,loss,acc\n')

        for epoch in range(args.epochs):
            lr_sch.step(epoch)
            train(epoch, model, trainloader, log_file, optimizer, criterion)
            best_loss = validate(epoch, model, validloader, log_file, criterion, best_loss)
    except Exception as e:
        print (e.message)
        log_file.write(e.message)
    finally:
        log_file.close()


if __name__ == '__main__':
    main()
