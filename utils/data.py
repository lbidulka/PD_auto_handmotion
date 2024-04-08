import os
import numpy as np
from scipy import io

from data.CAMERA_expert_labels import UPDRS_med_data_KW, UPDRS_med_data_SA
from data.PD4T_expert_labels import PD4T_handmotion_df
import utils.features as features


def make_subj_folds(subj_ids, N, data,
                    datasets,
                    data_format, use_ratio, combine_34,
                    annot_id=1):
    '''
    Splits the subj ids into N folds, ensureing that each fold has a relatively 
    balanced label distribution
    '''
    eval_folds = []
    eval_fold_labels = []
    eval_fold_dists = []
    for i in range(N):
        num_eval_subjs = len(subj_ids) // N
        eval_subjs = subj_ids[i*num_eval_subjs:(i+1)*num_eval_subjs]
        
        # For each fold of subj id's, we should check the label distribution
        # to ensure that the distribution is similar across all folds
        fold_data = data.get_subj_data(eval_subjs, format=data_format, use_ratio=use_ratio, combine_34=combine_34)
        fold_labels = fold_data[1][:, annot_id]
        label_dist = np.bincount(fold_labels.flatten().astype(int), minlength=4)        
        eval_folds.append(eval_subjs)
        eval_fold_labels.append(fold_labels)
        eval_fold_dists.append(label_dist)
    total_dist = np.sum(eval_fold_dists, axis=0)

    # TEMP: randomly reshuffle until distribution is roughly balanced
    print('Shuffling fold subjs until balanced label distribution is *roughly* achieved...')
    if datasets == 'CAMERA':
        off_avg_tol = [2, 2, 2, 2]
    elif datasets == 'PD4T':
        off_avg_tol = [30, 30, 30, 3.5]
    elif datasets == 'CAMERA,PD4T' or 'PD4T,CAMERA':
        off_avg_tol = [30, 30, 30, 3.5]
    else:
        raise ValueError(f"Invalid dataset: {datasets}")
    unacceptable_folds = True
    while unacceptable_folds:
        np.random.shuffle(subj_ids)
        eval_folds = []
        eval_fold_labels = []
        eval_fold_dists = []
        for i in range(N):
            num_eval_subjs = len(subj_ids) // N
            eval_subjs = subj_ids[i*num_eval_subjs:(i+1)*num_eval_subjs]
            
            fold_data = data.get_subj_data(eval_subjs, format=data_format, use_ratio=use_ratio, combine_34=combine_34)
            fold_labels = fold_data[1][:, annot_id]
            label_dist = np.bincount(fold_labels.flatten().astype(int), minlength=4)        
            eval_folds.append(eval_subjs)
            eval_fold_labels.append(fold_labels)
            eval_fold_dists.append(label_dist)
        
        # Check conditions
        if datasets == 'CAMERA':
            class_4_mask = np.array([True, True, True, True])
        elif datasets == 'PD4T':
            class_4_mask = np.array([True, True, True, False])
        elif datasets == 'CAMERA,PD4T' or 'PD4T,CAMERA':
            class_4_mask = np.array([True, True, True, True])
        else:
            raise ValueError(f"Invalid dataset: {datasets}")
        out_of_tol = np.any(np.abs(eval_fold_dists - (total_dist / N)) > off_avg_tol, where=class_4_mask)
        # if any folds have 0 in a class
        has_0cnt = np.any([np.all(eval_fold_dists, axis=1) == 0])

        unacceptable_folds = np.any([out_of_tol, has_0cnt])

        # # DEBUG: print out fold distributions
        # print('Label distribution across folds:')
        # for i, dist in enumerate(eval_fold_dists):
        #     print(f'Fold {i+1}: {dist}')
        # total_dist = np.sum(eval_fold_dists, axis=0)
        # print(f'Total distribution: {total_dist}')
        # print(f'Avg distribution: {total_dist / N}')

    print('Label distribution across folds:')
    for i, dist in enumerate(eval_fold_dists):
        print(f'Fold {i+1}: {dist}')
    
    # check total distribution
    total_dist = np.sum(eval_fold_dists, axis=0)
    print(f'Total distribution: {total_dist}')
    print(f'Avg distribution: {total_dist / N}')

    return eval_folds, eval_fold_dists

def equalize_class_samples(x_tensor_in, y_tensor_in, weight_annot_idx=1):
    '''
    Equalize the number of samples for each class in the dataset
    by repeating samples from the minority classes
    '''
    x_tensor = x_tensor_in.copy()
    y_tensor = y_tensor_in.copy()
    # largest count will be the final target count
    class_sample_count = np.array(
        [len(y_tensor[y_tensor[:,weight_annot_idx] == t]) for t in np.unique(y_tensor)])
    target_count = class_sample_count.max()
    # repeat each class to match target count
    for t in np.unique(y_tensor):
        class_count = len(y_tensor[y_tensor[:,weight_annot_idx] == t])
        if class_count < target_count:
            repeat = int(np.ceil(target_count / class_count))
            x_tensor = np.concatenate([x_tensor, x_tensor[y_tensor[:,weight_annot_idx] == t].repeat(repeat, 0)])
            y_tensor = np.concatenate([y_tensor, y_tensor[y_tensor[:,weight_annot_idx] == t].repeat(repeat, 0)])
    return x_tensor, y_tensor

def move_subj_samples(subjs, source, dest):
    '''
    Move all [subj] samples from source [x,y,subj_ids] to dest [x,y,subj_ids]
    '''
    for subj in subjs:
        subj_idxs = np.where(source[2] == subj)[0]
        x_subj = source[0][subj_idxs]
        y_subj = source[1][subj_idxs]
        subj_ids_subj = source[2][subj_idxs]

        x_dest = np.concatenate([dest[0], x_subj])
        y_dest = np.concatenate([dest[1], y_subj])
        subj_ids_dest = np.concatenate([dest[2], subj_ids_subj])
        x_source = np.delete(source[0], subj_idxs, axis=0)
        y_source = np.delete(source[1], subj_idxs, axis=0)
        subj_ids_source = np.delete(source[2], subj_idxs, axis=0)

    return x_source, y_source, subj_ids_source, x_dest, y_dest, subj_ids_dest

def balance_eval_split(x_train, x_test, y_train, y_test, 
                       subj_ids_train, subj_ids_test,
                       tol=0.2, weight_annot_idx=1):
    '''
    Ensure that test split has somewhat balanced classes
    '''    
    # ensure at least 1 sample in each class
    for t in np.unique(y_train):
        if len(y_test[y_test[:,weight_annot_idx] == t]) == 0:
            move_idx = np.where(y_train[:,weight_annot_idx] == t)[0][0]
            move_subj = subj_ids_train[move_idx]
            x_train, y_train, subj_ids_train, x_test, y_test, subj_ids_test = move_subj_samples([move_subj], 
                                                                                                [x_train, y_train, subj_ids_train], 
                                                                                                [x_test, y_test, subj_ids_test])

    train_class_cnt = np.array(
        [len(y_train[y_train[:,weight_annot_idx] == t]) for t in np.unique(y_train)])
    test_class_cnt = np.array(
        [len(y_test[y_test[:,weight_annot_idx] == t]) for t in np.unique(y_test)])

    # move maj test class samples from test to train, min test class samples from train to test
    test_class_diff = test_class_cnt.max() - test_class_cnt.min()
    if test_class_diff > (tol*test_class_cnt.max()):
        num_test_maj_move = 1
        num_test_min_move = 1

        maj_class = np.argmax(test_class_cnt)
        min_class = np.argmin(test_class_cnt)
        maj_class_idxs = np.where(y_test[:,weight_annot_idx] == maj_class)[0]
        min_class_idxs = np.where(y_train[:,weight_annot_idx] == min_class)[0]
        maj_class_ids = subj_ids_test[maj_class_idxs]
        min_class_ids = subj_ids_train[min_class_idxs]

        # move maj class samples from test to train
        x_train, y_train, subj_ids_train, x_test, y_test, subj_ids_test = move_subj_samples(maj_class_ids, 
                                                                                            [x_train, y_train, subj_ids_train], 
                                                                                            [x_test, y_test, subj_ids_test])

        # move min class samples from train to test
        x_train, y_train, subj_ids_train, x_test, y_test, subj_ids_test = move_subj_samples(min_class_ids, 
                                                                                            [x_train, y_train, subj_ids_train], 
                                                                                            [x_test, y_test, subj_ids_test])

    train_class_cnt = np.array(
        [len(y_train[y_train[:,weight_annot_idx] == t]) for t in np.unique(y_train)])
    test_class_cnt = np.array(
        [len(y_test[y_test[:,weight_annot_idx] == t]) for t in np.unique(y_test)])

    return x_train, x_test, y_train, y_test, subj_ids_train, subj_ids_test

def remove_unlabeled(subj_data, handednesses=None, 
                     combine_34=True, rej_either=True, rej_annot=None):
    '''
    Remove samples with label == -1
    '''
    x, y = subj_data[0], subj_data[1]
    if len(subj_data) > 2:
        subj_ids = subj_data[2]
    # y is a list of lists, and we want to rej any entry which contains a -1
    rej_idxs = []
    for i, labels in enumerate(y):
        if rej_either:
            if -1 in labels:
                rej_idxs.append(i)
        else:
            if rej_annot is not None:
                if labels[rej_annot] == -1:
                    rej_idxs.append(i)
            else:   
                if all([l == -1 for l in labels]):
                    rej_idxs.append(i)

    x_out = np.delete(x, rej_idxs, axis=0)
    y_out = np.delete(y, rej_idxs, axis=0)

    # combine label 3 and 4
    if combine_34: y_out[y_out == 4] = 3

    if subj_ids is None:
        return x_out, y_out, rej_idxs
    subj_ids_out = np.delete(subj_ids, rej_idxs, axis=0)
    if handednesses is None:
        return x_out, y_out, subj_ids, rej_idxs
    handednesses_out = np.delete(handednesses, rej_idxs, axis=0)
    return x_out, y_out, subj_ids_out, handednesses_out, rej_idxs

def load_raw_ts(file_list, args, trimming):
    '''
    Loads raw hand pose timeseries, trims them, and returns them as a list along 
    with annotation info
    '''
    task = 'hand_movement'

    raw_ts = []
    trim_ts = []
    labels = []
    subj_ids = []
    handednesses = []
    for file in file_list:
        # file_path = os.path.join(args.inputFolder, file)
        file_name = file.split('/')[-1]

        if args.dataset == 'PD4T':
            visit = file_name.split('_')[0]
            subj_id = file_name.split('_')[2]
            handedness = file_name.split('_')[3]
            subj_labels = get_PD4T_labels_from_df(subj_id, handedness[0], visit, task)
        elif args.dataset == 'CAMERA':
            temp = file.split('/')[-1]
            temp = temp.split('_')
            subj_id = temp[0]
            date = temp[1]
            handedness = temp[-2]
            subj_labels = get_CAMERA_labels_from_dicts(subj_id, handedness, date, task)
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")

        # At least one annotation is available
        if not all([l is None for l in subj_labels]):
            try:
                segments = trimming[subj_id][handedness]
            except:
                # default to use entire signal
                segments = {0: {'start': 0, 'end': -1}}
            # segments = {0: {'start': 0, 'end': -1}}

            print(file)
            print('labels: ', subj_labels)
            matfile = io.loadmat(file)
            
            fileType = 'raw'

            for segment_id in segments.keys():
                print('segment id: ' + str(segment_id))
                data_block = {}
                data_block['datatype'] = fileType

                if 'hand' in matfile:
                    tdata = matfile['hand']
                elif 'left' in matfile:
                    tdata = matfile['left']
                elif 'right' in matfile:
                    tdata = matfile['right']

                # settings
                start = segments[segment_id]['start']
                end = segments[segment_id]['end']
                print('start: '+str(start)+'; end: '+str(end))

                tdata_trim = tdata[start:end, :, :]
                
                MIN_SEQ_LEN = 32
                if tdata.shape[0] > MIN_SEQ_LEN:
                    raw_ts.append(tdata)
                    trim_ts.append(tdata_trim)
                    labels.append(subj_labels)
                    subj_ids.append(subj_id)
                    handednesses.append(handedness)
                else:
                    print(f"{subj_id}, {handedness}: Empty data block, skipping")
        else:
            print("Subject not in UPDRS_med_data: " + str(subj_id))
    
    return raw_ts, trim_ts, labels, subj_ids, handednesses

def get_PD4T_labels_from_df(subj_id, handedness, visit, task):
    '''
    Get the labels for a given subject, handedness, and date from the PD4T dataset annotations.
    Gives None if the subject is not in the annotations (there are no unlabeled samples)
    '''
    if task == 'hand_movement':
        PD4T_df = PD4T_handmotion_df
    else:
        raise ValueError(f"Invalid PD4T task: {task}")
    annot = PD4T_df.loc[(PD4T_df['patient_id'] == subj_id) & (PD4T_df['visit'] == visit) & (PD4T_df['handedness'] == handedness)]
    label = annot['UPDRS_score'].values
    if len(label) == 0:
        label = None
    else:
        label = label[0]
    return [label]

def get_CAMERA_labels_from_dicts(subj_id, handedness, date, task):
    '''
    Get the labels for a given subject, handedness, and date from the CAMERA dataset annotations.
    Gives -1 if it is unlabeled and None if the subject is not in the annotations
    '''
    # check which dicts the combo of subj_id, handedness, and task is in (if any)
    if subj_id in UPDRS_med_data_KW.keys():
        try:
            label_KW = UPDRS_med_data_KW[subj_id][task][f'{handedness}_open_close'][date]
        # Unlabeled annotation
        except: 
            label_KW = -1 
        if label_KW is None:
            label_KW = -1
    else:
        label_KW = -1

    if subj_id in UPDRS_med_data_SA.keys():
        try:
            label_SA = UPDRS_med_data_SA[subj_id][task][f'{handedness}_open_close'][date]
        # Unlabeled annotation
        except:
            label_SA = -1
        if label_SA is None:
            label_SA = -1

        # Special case, subject performed both actions and label is an anomaly
        if subj_id == '25779':
            label_SA = 2.0

    else:
        label_SA = -1
    
    return [float(label_KW), float(label_SA)]


    # if (subj_id in UPDRS_med_data_KW.keys()) or (subj_id in UPDRS_med_data_SA.keys()):
    #         try:
    #             label_KW = UPDRS_med_data_KW[subj_id]['hand_movement'][f'{handedness}_open_close'][date]
    #         except:
    #             label_KW = -1
    #         try:
    #             label_SA = UPDRS_med_data_SA[subj_id]['hand_movement'][f'{handedness}_open_close'][date]
    #         except:
    #             label_SA = -1
            
    #         if label_KW is None:
    #             label_KW = -1
    #         if label_SA is None:
    #             label_SA = -1

def auto_trim_dist_ts(dist_ts, kpts_ts, trim_mask, num_cycles=10, num_passes=1, smooth=True):
    '''
    Automatically trim the fingertip-palm distance timeseries to the desired number
    of action cycles
    '''
    too_short_mask = [False for ts in dist_ts]
    dist_ts_trims = [ts for ts in dist_ts]
    kpts_ts_trims = [ts for ts in kpts_ts]
    peaks = [[] for ts in dist_ts]
    for i in range(num_passes):
        for j, ts in enumerate(dist_ts_trims):
            # Only trim if mask is true
            if trim_mask[j]:
                # check number of peaks
                peak_idxs, peaks_vals = features.get_cycle_peaks(np.array([ts.mean(1)]), min_peak_dist=None, 
                                                                keep=10, savgol_win=5, 
                                                                prominence=0.15, smooth=smooth)
                peak_idxs = peak_idxs[0]
                peaks_vals = peaks_vals[0]
                start_idx = 0
                end_idx = -1
                # Too short?
                if len(peak_idxs) < num_cycles:
                    too_short_mask[j] = True
                    start_idx = 0
                    end_idx = 25
                else:
                    too_short_mask[j] = False
                    # Too long?
                    if len(peak_idxs) != num_cycles:
                        # trim end to end at trough after last peak, trim start to num_cycles cycles before end
                        end_idx = int(peak_idxs[-1] + np.diff(peak_idxs[-3:-1]).mean() / 2)
                        start_idx = int(peak_idxs[-num_cycles] -np.diff(peak_idxs[-num_cycles:-num_cycles+3]).mean() / 2)

                # trim kpt and dist series
                dist_ts_trims[j] = ts[start_idx:end_idx]
                kpts_ts_trims[j] = kpts_ts[j][start_idx:end_idx]
                peaks[j] = [peak_idxs - start_idx, peaks_vals]

    return dist_ts_trims, kpts_ts_trims, too_short_mask, peaks
