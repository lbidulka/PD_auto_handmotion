import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import wandb
import types

from models import dsp_updrs, simple_mlp, simple_cnn, ratio_mlp, feature_ml, feature_mlp, ddnet, dist_ddnet, cnn_vae

import data.timeseries.data_timeseries as data_timeseries
from utils import evaluation as eval_utils
from utils import data as data_utils


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--task', default='multiclass', help='Task to perform: binclass or multiclass')
    parser.add_argument('--UPDRS_task', default='hand_movement', help='Task to process')  #hand_movement, finger_tapping
    parser.add_argument('--datasets', default='CAMERA,PD4T', help='Datasets to process (comma separated, no spaces)')   # CAMERA, PD4T
    parser.add_argument('--rand_baseline', default=False, help='Use random baseline?')   # True False

    parser.add_argument('--model', default='ddnet', help='Model to use')   # ddnet, dist_ddnet, feature_ml, cnn_vae, updrs_dsp, simple_mlp, simple_cnn, ratio_mlp, feature_mlp

    parser.add_argument('--wblog', default=False, help='Log to wandb?')   # True False
    parser.add_argument('--num_trials', default=5, help='Number of trials to run')   # 1, 5, 10
    parser.add_argument('--num_folds', default=15, help='Number of folds for N-fold evaluation')   # 5, 10
    
    parser.add_argument('--save_model', default=False, help='Save deep model?')   # True False
    parser.add_argument('--save_model_path', default='./checkpoints/', help='Path to save models')

    parser.add_argument('--device', default='cuda:1', help='Device to run on')   # cuda, cuda:0, cuda:1, cpu

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
                print(f'    {metric}: {value:.2f}')
        print('')

def set_seed(args):
    '''
    Set random seed
    '''
    if not hasattr(args, 'seed'):
        seed = random.randint(0, 1000000)
        args.seed = seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return 

def init_logger(args, model):
    '''
    Initialize wandb logger if desired
    '''
    if args.wblog:
        config = {
            'seed': args.seed,
            'datasets': args.datasets,
            'model': model.name,
        }
        if model.name == 'ddnet':
            config['m_lr'] = model.lr
            config['m_epochs'] = model.num_epochs
            config['m_loss_type'] = model.loss_type
            if model.loss_type == 'Focal':
                config['m_focal_gamma'] = model.focal_gamma

            config['m_frame_l'] = model.frame_l
            config['m_filters'] = model.filters
            config['m_m_branch'] = model.m_branch
            config['m_f_branch'] = model.f_branch
            config['m_in_kpts'] = model.input_kpts

        if model.scheduler_type is not None:
            config['m_scheduler_type'] = model.scheduler_type

        wandb.init(project='auto-UPDRS', config=config)

def N_fold_eval(args, model, data):
    '''
    N_fold train/evaluation over all samples. Excluding a few subjects for testing each time.
    '''
    if model.name == 'feature_ml':
        data_format = model.sample_format
        combine_34 = model.combine_34
    elif model.name in ['ddnet', 'dist_ddnet', 'cnn_vae']:
        data_format = model.sample_format
        combine_34 = model.combine_34
    else:
        data_format = 'scaled'  # 'scaled', 'unscaled', 'unscaled_kpt'
        combine_34 = False

    rej_unlabelled_annot = model.labeler_idx # if not None, reject samples if this annotator has label == -1
    rej_either = False   # if True, reject samples if any label == -1. If False, reject if all labels == -1
    binclass_idx = 0    # index to split multiclass into binary classification (ie label > binclass_idx is 1, else 0)
    keep_only_agreed_labels = False # if True, keep only samples where all labels are the same

    subj_ids = np.unique(data.subj_ids)
    subj_data = data.get_subj_data(subj_ids, use_ratio=model.use_ratio, format=data_format)
    _, _, _, rej_idxs = data_utils.remove_unlabeled(subj_data, 
                                                    combine_34=combine_34, 
                                                    keep_agree=keep_only_agreed_labels,
                                                    rej_either=rej_either, 
                                                    rej_annot=rej_unlabelled_annot)
    data.delete_idxs(rej_idxs)

    subj_ids = np.unique(data.subj_ids)
    subj_ids = np.random.permutation(subj_ids)
    eval_folds, eval_fold_dists = data_utils.make_subj_folds(subj_ids, args.num_folds, data, 
                                                             args.datasets, data_format, model.use_ratio, 
                                                             combine_34, model.labeler_idx)
    if args.wblog:
        wandb.log({'fold_dists': eval_fold_dists}, commit=False)

    # Set weights for loss if needed
    if hasattr(model, 'loss_type') and model.loss_type == 'Focal':
        class_cnt_idx = model.labeler_idx
        class_cnt = torch.bincount(torch.tensor(data.y[:, class_cnt_idx]).long())
        if combine_34:
            class_cnt[3] += class_cnt[4]
            class_cnt = class_cnt[:4]
    else:
        class_cnt = None

    # Run that sucker
    print(f'\n{args.num_folds}-fold Eval on {len(subj_ids)} subjects:')
    eval_preds, eval_targets, eval_ids = [], [], []
    for i in tqdm(range(args.num_folds)):
        eval_subjs = eval_folds[i]
        eval_subj_data = data.get_subj_data([eval_subjs], format=data_format, use_ratio=model.use_ratio, combine_34=combine_34)
        train_subj_data = data.get_subj_data(subj_ids[[id not in eval_subjs for id in subj_ids]], 
                                             format=data_format, use_ratio=model.use_ratio, combine_34=combine_34)
        train_x, train_y, train_subj_ids = train_subj_data[0], train_subj_data[1], train_subj_data[2]
        test_x, test_y, test_subj_ids = eval_subj_data[0], eval_subj_data[1], eval_subj_data[2]

        # Convert to binary classification if needed
        if args.task == 'binclass':
            train_mask = train_y > binclass_idx
            train_y[train_mask] = 1.0
            train_y[~train_mask] = 0.0
            test_mask = test_y > binclass_idx
            test_y[test_mask] = 1.0
            test_y[~test_mask] = 0.0

        # Train and evaluate
        if len(test_x) != 0:
            if class_cnt != None:
                model.init_model(class_cnts=class_cnt)
            else: 
                model.init_model()
            model.train(train_x, train_y, train_subj_ids=train_subj_ids)
            test_pred = model(test_x)
            if model.name != 'feature_ml':
                test_pred = test_pred.cpu().numpy()

            print(f'Fold {i+1}: ')
            metrics = eval_utils.get_metrics(test_pred, test_y, task=args.task)
            print_metrics(metrics)

            eval_preds.append(test_pred.reshape(-1))
            eval_targets.append(test_y)
            eval_ids.append(test_subj_ids)

        # Save model dict
        if args.save_model:
            if model.name != 'feature_ml':
                save_model_path = os.path.join(args.save_model_path, args.UPDRS_task)
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                model_dict = model.get_model_dict()
                model_dict['metrics'] = metrics
                model_dict['seed'] = args.seed

                model_name = f'{eval_model}_{args.UPDRS_task}_fold{i}.pt'
                print(f'Saving model: {model_name} to {save_model_path}')
                torch.save(model_dict, os.path.join(save_model_path, model_name))
    
    eval_preds = np.hstack(eval_preds)
    eval_targets = np.vstack(eval_targets)
    eval_ids = np.hstack(eval_ids)
    metrics = eval_utils.get_metrics(eval_preds, eval_targets, 
                                     task=args.task)
    
    wandb_annot_idx = model.labeler_idx
    if wandb.run is not None:
        wandb.log({
            'T_acc': metrics[wandb_annot_idx]['acc'],
            'T_acc_t2': metrics[wandb_annot_idx]['acc_t2'],
            'T_precision': metrics[wandb_annot_idx]['precision'],
            'T_recall': metrics[wandb_annot_idx]['recall'],
            'T_f1': metrics[wandb_annot_idx]['f1'],
            'T_conf_mat': metrics[wandb_annot_idx]['conf_mat'],
        })
    print(f'\n--- {model.name} ---')
    print_metrics(metrics)
    
    if args.rand_baseline:
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
            possible_labels = np.unique(eval_targets)
            freq_preds = np.random.choice(possible_labels, size=eval_targets.shape[0], 
                                        p=class_cnts/np.sum(class_cnts))
            freq_metrics = eval_utils.get_metrics(freq_preds, eval_targets, 
                                                task=args.task)
            print("\n--- Frequency Class Predictor ---")
            print_metrics(freq_metrics)
    
    return metrics


if __name__ == '__main__':
    args = parse_args()
    set_seed(args)
    all_metrics = {}
    for i in tqdm(range(args.num_trials)):
        print(f'\n--- Trial {i+1} / {args.num_trials}---')

        eval_model = args.model #'ddnet'   # updrs_dsp, ddnet, feature_ml, simple_mlp, simple_cnn, ratio_mlp, feature_mlp
        classifier = 'svr' # Classifier to use for feature_ml: svr, svm, rf, dt

        # Define model and data
        data = data_timeseries.data_timeseries(args.datasets, args.UPDRS_task)
        if eval_model == 'updrs_dsp':
            model = dsp_updrs.UPDRS_DSP(task=args.task,)
        elif eval_model == 'ddnet':
            model = ddnet.DDNet(task=args.task, datasets=args.datasets, UPDRS_task=args.UPDRS_task, device=args.device)
            data = data_timeseries.data_timeseries(args.datasets, args.UPDRS_task, kpts_uniform_len=model.frame_l)  # set kpts series len
        elif eval_model == 'dist_ddnet':
            model = dist_ddnet.DistDDNet(task=args.task, datasets=args.datasets, UPDRS_task=args.UPDRS_task, device=args.device)
            data = data_timeseries.data_timeseries(args.datasets, args.UPDRS_task, kpts_uniform_len=model.frame_l)  # set kpts series len
        elif eval_model == 'cnn_vae':
            model = cnn_vae.CnnVae(task=args.task, datasets=args.datasets, device=args.device, 
                                   length=256, nclasses=4, transition_channels=4)
        # FEATURE BASELINES
        elif eval_model == 'feature_ml':
            model = feature_ml.Feature_ML(task=args.task, classifier=classifier)
        elif eval_model == 'feature_mlp':
            model = feature_mlp.FeatureMLP(sample_len=data.x.shape[1], in_channels=data.x.shape[2], 
                                        task=args.task,)
        # NAIVE BASELINES
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

        init_logger(args, model)

        # Train/Eval the model
        metrics = {}
        metrics = N_fold_eval(args, model, data)
        all_metrics[i] = metrics.copy()

        wandb.finish()       

    # combine metrics (dict of all labelers metrics) into single avgd dict
    avg_metrics = {}#{k: {metric: np.mean([m[k][metric] for m in avg_metrics.values()]) for metric in metrics[0].keys()} for k in range(2)}
    for k in [i for i in all_metrics[0].keys() if isinstance(i, int)]:
        avg_metrics[k] = {metric: np.mean([m[k][metric] for m in all_metrics.values()]) for metric in metrics[0].keys()}
        avg_metrics[k]['conf_mat'] = np.mean([m[k]['conf_mat'] for m in all_metrics.values()], axis=0)
    
    print(f'\n--- {args.num_trials}-Run Avg RAW Metrics ---')
    print_metrics(avg_metrics)

    # normalize the confusion matrix
    for k in avg_metrics.keys():
        avg_metrics[k]['conf_mat'] = (avg_metrics[k]['conf_mat'] / np.sum(avg_metrics[k]['conf_mat'], axis=1)[:, None]).round(2)
    
    print(f'\n--- {args.num_trials}-Run Avg Metrics ---')
    print_metrics(avg_metrics)
    

    
