import argparse
import os
import numpy as np
from tqdm import tqdm

# from models.dsp_updrs import UPDRS_DSP
# from models.simple_mlp import SimpleMLP
from models import dsp_updrs, simple_mlp, simple_cnn, ratio_mlp, feature_ml

import data.timeseries.data_timeseries as data_timeseries
from utils import evaluation as eval_utils
from utils import data as data_utils


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--task', default='multiclass', help='Task to perform: binclass or multiclass')
    parser.add_argument('--datasets', default='PD4T', help='Datasets to process (comma separated, no spaces)')   # CAMERA, PD4T

    args = parser.parse_args() 
    return args

def print_metrics(metrics):
    print('-- Metrics: --')
    for rater in metrics.keys():
        print(f'Rater {rater}:')
        for metric, value in metrics[rater].items():
            if metric == 'conf_mat':
                print(f'    {metric}: \n{value}')
            else:
                print(f'    {metric}: {value:.3f}')
        print('')

def CA_TCC_eval(args, model,):
    '''
    Train/val/test split evaluation using splits from CA-TCC experiment
    '''
    assert args.task == 'multiclass'

    combine_34 = False
    unscaled_ts = False  # if True, use unscaled timeseries data

    # Load data
    data_splits_path = '../CA-TCC/data/camUPDRS/'
    data = data_timeseries.data_timeseries(CATCC_splits_path=data_splits_path)
    train_x = data.train_frac['samples'].numpy()
    train_y = data.train_frac['labels'].numpy()
    val_x = data.val_frac['samples'].numpy()
    val_y = data.val_frac['labels'].numpy()
    test_x = data.test['samples'].numpy()
    test_y = data.test['labels'].numpy()
    
    # simulate multiple raters
    train_y = np.repeat(train_y.reshape(-1,1), 2, axis=1)
    val_y = np.repeat(val_y.reshape(-1,1), 2, axis=1)
    test_y = np.repeat(test_y.reshape(-1,1), 2, axis=1)

    # Train and eval
    model.init_model()
    model.train(train_x, train_y, x_val=val_x, y_val=val_y)
    model.model.eval()
    test_pred = model(test_x).numpy()

    # Metrics
    metrics = eval_utils.get_metrics(test_pred.reshape(-1,1), test_y, task=args.task)
    print(f'\n--- {model.name} ---')
    print_metrics(metrics)

def leave_one_out_eval(args, model, data):
    '''
    leave-one-out train/evaluation over all samples. Excluding 1 subjects samples at a time.
    '''
    combine_34 = False
    rej_either = False   # if True, reject samples if any label == -1. If False, reject if all labels == -1
    unscaled_ts = False  # if True, use unscaled timeseries data

    eval_preds, eval_targets = [], []
    subj_ids = np.unique(data.subj_ids)
    print(f'Leave-one-out Eval on {len(subj_ids)} subjects:')
    for subj_id in tqdm(subj_ids):
        eval_subj_data = data.get_subj_data([subj_id], model.use_ratio, unscaled=unscaled_ts)
        train_subj_data = data.get_subj_data(subj_ids[subj_ids != subj_id], model.use_ratio, unscaled=unscaled_ts)
        
        # remove samples with label == -1
        train_x, train_y = data_utils.remove_unlabeled(train_subj_data, 
                                                       combine_34=combine_34, rej_either=rej_either)
        test_x, test_y = data_utils.remove_unlabeled(eval_subj_data,
                                                     combine_34=combine_34)

        # Convert to binary classification if needed
        if args.task == 'binclass':
            train_y[train_y > 0] = 1.0
            test_y[test_y > 0] = 1.0

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
    
    print(f'\n--- {model.name} ---')
    print_metrics(metrics)
    
    if args.task == 'binclass':
        # check against majority class predictor (1)
        maj_preds = np.ones_like(eval_preds)
        maj_metrics = eval_utils.get_metrics(maj_preds, eval_targets, 
                                             task=args.task)
        print("\n--- Majority Class Predictor (1) ---")
        print_metrics(maj_metrics)
    elif args.task == 'multiclass':
        # get avg class counts
        class_cnts = []
        for rater in range(eval_targets.shape[1]):
            class_cnts.append(np.bincount(eval_targets[:,0].astype(int), minlength=4))                        
        class_cnts = np.mean(class_cnts, axis=0)

        # check against random predictor based on train label frequency
        freq_preds = np.random.choice([0, 1, 2, 3, 4], size=eval_targets.shape[0], 
                                     p=class_cnts/np.sum(class_cnts))
        freq_metrics = eval_utils.get_metrics(freq_preds, eval_targets, 
                                              task=args.task)
        print("\n--- Frequency Class Predictor ---")
        print_metrics(freq_metrics)


if __name__ == '__main__':
    args = parse_args()
    eval_model = 'feature_ml'   # updrs_dsp, simple_mlp, simple_cnn, ratio_mlp
    classifier='svr' # Classifier to use for feature_ml: svr, svm, rf, dt

    # Define model and data
    data = data_timeseries.data_timeseries(args.datasets)
    if eval_model == 'updrs_dsp':
        model = dsp_updrs.UPDRS_DSP(task=args.task,)
    elif eval_model == 'feature_ml':
        model = feature_ml.Feature_ML(task=args.task, classifier=classifier)
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

    # Train/Eval the model
    leave_one_out_eval(args, model, data)
    # CA_TCC_eval(args, model)

    
