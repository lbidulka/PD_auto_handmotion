import argparse
import os

from models.dsp_updrs import UPDRS_DSP
import data.timeseries.data_timeseries as data_timeseries
from utils import evaluation as eval_utils
from utils import data as data_utils

def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--foo', default='bar', help='example')

    args = parser.parse_args() 
    return args


if __name__ == '__main__':
    args = parse_args()

    pwd = os.getcwd()
    data_path = pwd + '/data/timeseries/' + 'CAMERA_UPDRS/data_all.npz'


    # Define the model and data
    data = data_timeseries.data_timeseries(data_path)
    updrs_dsp = UPDRS_DSP()

    # evaluate over whole dataset
    preds = updrs_dsp(data.x)
    targets = data.y

    preds, targets = data_utils.remove_unlabeled(preds, targets)    # don't include samples with either label == -1
    targets = targets[:,1]

    metrics = eval_utils.get_metrics(preds, targets)

    print('\n--- Metrics: ---')
    for metric, value in metrics.items():
        if metric == 'conf_mat':
            print(f'{metric}: \n{value}')
        else:
            print(f'{metric}: {value:.3f}')
