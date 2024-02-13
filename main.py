import argparse
import os
import numpy as np
from tqdm import tqdm

# from models.dsp_updrs import UPDRS_DSP
# from models.simple_mlp import SimpleMLP
from models import dsp_updrs, simple_mlp, simple_cnn, ratio_mlp

import data.timeseries.data_timeseries as data_timeseries
from utils import evaluation as eval_utils
from utils import data as data_utils


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--task', default='multiclass', help='Task to perform: binclass or multiclass')

    args = parser.parse_args() 
    return args

def print_metrics(metrics):
    print('--- Metrics: ---')
    for rater in metrics.keys():
        print(f'Rater {rater}:')
        for metric, value in metrics[rater].items():
            if metric == 'conf_mat':
                print(f'    {metric}: \n{value}')
            else:
                print(f'    {metric}: {value:.3f}')
        print('')

def leave_one_out_eval(args, model, data):
    '''
    leave-one-out train/evaluation over all samples. Excluding 1 subjects samples at a time.
    '''
    combine_34 = False
    rej_either = False   # if True, reject samples if any label == -1. If False, reject if all labels == -1

    eval_preds, eval_targets = [], []
    subj_ids = np.unique(data.subj_ids)
    print(f'Leave-one-out Eval on {len(subj_ids)} subjects:')
    for subj_id in tqdm(subj_ids):
        eval_subj_data = data.get_subj_data([subj_id], model.use_ratio)
        train_subj_data = data.get_subj_data(subj_ids[subj_ids != subj_id], model.use_ratio)
        
        # remove samples with label == -1
        train_x, train_y = data_utils.remove_unlabeled(train_subj_data, 
                                                       combine_34=combine_34, rej_either=rej_either)
        test_x, test_y = data_utils.remove_unlabeled(eval_subj_data,
                                                     combine_34=combine_34)

        # Convert to binary classification if needed
        if args.task == 'binclass':
            train_y[train_y > 0] = 1.0
            test_y[test_y > 0] = 1.0
            train_y = train_y.reshape(-1, 1)
            test_y = test_y.reshape(-1, 1)

        # Train and evaluate
        if test_x.shape[0] != 0:
            model.init_model()
            model.train(train_x, train_y)
            test_pred = model(test_x)

            eval_preds.append(test_pred.reshape(-1))
            eval_targets.append(test_y)
    
    eval_preds = np.hstack(eval_preds)
    eval_targets = np.vstack(eval_targets)
    metrics = eval_utils.get_metrics(eval_preds, eval_targets, 
                                     task=args.task)
    
    print("\n--- Model ---")
    print_metrics(metrics)
    
    if args.task == 'binclass':
        # check against majority class predictor (1)
        maj_preds = np.ones_like(eval_targets)
        maj_metrics = eval_utils.get_metrics(maj_preds, eval_targets, 
                                             task=args.task)
        print("\n--- Majority Class Predictor (1) ---")
        print_metrics(maj_metrics)


if __name__ == '__main__':
    args = parse_args()
    pwd = os.getcwd()
    data_path = pwd + '/data/timeseries/' + 'CAMERA_UPDRS/handmotion_all.npz'

    eval_model = 'ratio_mlp'   # updrs_dsp, simple_mlp, simple_cnn, ratio_mlp

    # Define the model and data
    data = data_timeseries.data_timeseries(data_path)
    if eval_model == 'updrs_dsp':
        model = dsp_updrs.UPDRS_DSP()
    elif eval_model == 'simple_mlp':
        model = simple_mlp.SimpleMLP(sample_len=data.x.shape[1], in_channels=data.x.shape[2], 
                                     task=args.task,)
    elif eval_model == 'ratio_mlp':
        model = ratio_mlp.RatioSimpleMLP(sample_len=data.x.shape[1], in_channels=data.x.shape[2], 
                                     task=args.task,)
    elif eval_model == 'simple_cnn':
        model = simple_cnn.SimpleCNN(sample_len=data.x.shape[1], in_channels=data.x.shape[2], 
                                    task=args.task,)
    else:
        raise NotImplementedError

    # Evaluate the model
    leave_one_out_eval(args, model, data)

    
