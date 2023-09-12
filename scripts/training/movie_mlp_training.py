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
from scripts.models.movie_mlp import MovieMLPModel
from scripts.dataset.movie_mlp_dataset import MovieMLPDataset

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description='Train multi layers perceptron')

    parser.add_argument('-i', '--input', default='../p_input/train_data_full.csv', help='input data file name')
    parser.add_argument('-sm', '--sequence-min', default=3, help='minimum sequence length')
    parser.add_argument('-vc', '--valid-count', default=2, help='validation example count per user')

    parser.add_argument('-mp', '--model-path', default='../models/', help='path to models directory')
    parser.add_argument('-m', '--model-name', default='movies_mlp_model_v1', help='name of trained model')
    parser.add_argument('-bs', '--batch-size', default=4096, help='size of batches')
    parser.add_argument('-lr', '--learning-rate', default=0.001, help='learning rate')
    parser.add_argument('-e', '--epochs', default=20, help='epochs count')

    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=['session_start_datetime'])
    movie_id_trans = labeling_col(df, 'primary_video_id')

    movies_sequences = process_sequence(df, 'primary_video_id')
    wp = process_sequence(df, 'watching_percentage')

    wp_all = np.zeros((len(movies_sequences), len(movie_id_trans.classes_)), dtype=np.float16)
    train_pairs = []
    valid_pairs = []

    def make_glob_from_data(seq, seq_data):
        res = np.zeros((len(movie_id_trans.classes_)), dtype=np.float16)
        for idx, sv in enumerate(seq):
            res[sv] += seq_data[idx]
        return res

    for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
        wp_all[si] = make_glob_from_data(seq, wp[si])
        for idx in range(args.sequence_min, len(seq)):
            if idx >= len(seq)-args.valid_count:
                valid_pairs.append((si, zip(seq[idx:], wp[si][idx:])))
            else:
                train_pairs.append((si, zip(seq[idx:], wp[si][idx:])))

    model_path = os.path.join(args.model_path, "torch_{}".format(args.model_name))

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    trainset = MovieMLPDataset(wp_all, train_pairs)
    validset = MovieMLPDataset(wp_all, valid_pairs)

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

    model = MovieMLPModel(
        features_dim=trainset[0][0].shape[0],
        result_classes=df.primary_video_id.unique().shape[0],
        mid_dim=4096,
        mid_dim2=4096
    ).cuda()

    best_loss = np.finfo(np.float32).max

    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.6)

    def train(epoch, model, trainloader, log_file, optimizer, criterion):
        '''
        Train function for each epoch
        '''
        model.train()
        train_loss = 0

        print('Training Epoch {} optimizer LR {}'.format(epoch, optimizer.param_groups[0]['lr']))

        for batch_idx, (glob_data, targets) in enumerate(trainloader):
            glob_data = glob_data.cuda()
            targets = targets.cuda()

            outputs = model(glob_data)
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
            for batch_idx, (glob_data, targets) in enumerate(validloader):
                glob_data = glob_data.cuda()
                targets = targets.cuda()

                outputs = model(glob_data)
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
