import argparse
import os
import numpy as np
from tqdm import tqdm

from models.dsp_updrs import UPDRS_DSP
from models.simple_mlp import SimpleMLP

import data.timeseries.data_timeseries as data_timeseries
from utils import evaluation as eval_utils
from utils import data as data_utils


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--foo', default='bar', help='example')

    args = parser.parse_args() 
    return args


def leave_one_out_eval(model, data):
    '''
    leave-one-out train/evaluation over all samples. Excluding 1 subjects samples at a time.
    '''
    labeler_idx = 0
    combine_34 = False

    eval_preds, eval_targets = [], []
    subj_ids = np.unique(data.subj_ids)
    print(f'Leave-one-out Eval on {len(subj_ids)} subjects:')
    for subj_id in tqdm(subj_ids):
        eval_subj_data = data.get_subj_data([subj_id])
        train_subj_data = data.get_subj_data(subj_ids[subj_ids != subj_id])
        
        # remove samples with label == -1
        train_x, train_y = data_utils.remove_unlabeled(train_subj_data[0], train_subj_data[1], combine_34=combine_34)
        train_y = train_y[:,labeler_idx] # TEMP: only using 2nd label for now
        test_x, test_y = data_utils.remove_unlabeled(eval_subj_data[0], eval_subj_data[1], combine_34=combine_34)
        test_y = test_y[:,labeler_idx]    # TEMP: only using 2nd label for now

        # train on all samples except excluded subject, then eval on excluded subject
        model.train(train_x, train_y)
        test_pred = model(test_x)

        eval_preds.append(test_pred)
        eval_targets.append(test_y)
    
    eval_preds = np.hstack(eval_preds)
    eval_targets = np.hstack(eval_targets)
    metrics = eval_utils.get_metrics(eval_preds, eval_targets)
    print('\n--- Metrics: ---')
    for metric, value in metrics.items():
        if metric == 'conf_mat':
            print(f'{metric}: \n{value}')
        else:
            print(f'{metric}: {value:.3f}')


if __name__ == '__main__':
    args = parse_args()

    pwd = os.getcwd()
    data_path = pwd + '/data/timeseries/' + 'CAMERA_UPDRS/data_all.npz'

    # Define the model and data
    data = data_timeseries.data_timeseries(data_path)
    # model = UPDRS_DSP()
    model = SimpleMLP(sample_len=data.x.shape[1], in_channels=data.x.shape[2])

    # Evaluate the model
    leave_one_out_eval(model, data)

    
