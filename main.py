import argparse
import os
import numpy as np

from models.dsp_updrs import UPDRS_DSP
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
    eval_preds, eval_targets = [], []
    subj_ids = np.unique(data.subj_ids)
    for subj_id in subj_ids:
        eval_subj_data = data.get_subj_data(subj_id)
        
        # train over all samples except this subject's samples
        # NO TRAIN FOR NOW

        # eval on excluded subject's data
        eval_pred = model(eval_subj_data[0])
        eval_target = eval_subj_data[1]

        eval_pred, eval_target = data_utils.remove_unlabeled(eval_pred, eval_target)    # don't include samples with either label == -1
        eval_target = eval_target[:,1]  # TEMP: only using 2nd label for now

        eval_preds.append(eval_pred)
        eval_targets.append(eval_target)
        
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
    model = UPDRS_DSP()

    # Evaluate the model
    leave_one_out_eval(model, data)

    
