import os
import numpy as np
from scipy import io

from data.CAMERA_expert_labels import UPDRS_med_data_KW, UPDRS_med_data_SA
from data.PD4T_expert_labels import PD4T_handmotion_df
import utils.features as features

def remove_unlabeled(subj_data, subj_ids=None, handednesses=None, 
                     combine_34=True, rej_either=True):
    '''
    Remove samples with label == -1
    '''
    x, y = subj_data[0], subj_data[1]
    # y is a list of lists, and we want to rej any entry which contains a -1
    rej_idxs = []
    for i, labels in enumerate(y):
        if rej_either:
            if -1 in labels:
                rej_idxs.append(i)
        else:            
            if all([l == -1 for l in labels]):
                rej_idxs.append(i)

    x_out = np.delete(x, rej_idxs, axis=0)
    y_out = np.delete(y, rej_idxs, axis=0)

    # combine label 3 and 4
    if combine_34: y_out[y_out == 4] = 3

    if subj_ids is None:
        return x_out, y_out
    
    subj_ids_out = np.delete(subj_ids, rej_idxs, axis=0)
    handednesses_out = np.delete(handednesses, rej_idxs, axis=0)
    return x_out, y_out, subj_ids_out, handednesses_out

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

def auto_trim_dist_ts(dist_ts, trim_mask, num_cycles=10, num_passes=1,):
    '''
    Automatically trim the fingertip-palm distance timeseries to the desired number
    of action cycles
    '''
    too_short_mask = [False for ts in dist_ts]
    ts_trims = [ts for ts in dist_ts]
    peaks = [[] for ts in dist_ts]
    for i in range(num_passes):
        for j, ts in enumerate(ts_trims):
            # Only trim if mask is true
            if trim_mask[j]:
                # check number of peaks
                peaks_idxs, peaks_vals = features.get_cycle_peaks(np.array([ts]), min_peak_dist=None, 
                                                                keep_10=False, savgol_win=5, prominence=0.15)
                peaks_idxs = peaks_idxs[0]
                peaks_vals = peaks_vals[0]
                start_idx = 0
                end_idx = -1
                # Too short?
                if len(peaks_idxs) < num_cycles:
                    too_short_mask[j] = True
                    ts_trims[j] = ts[:25]
                else:
                    too_short_mask[j] = False
                    # Too long?
                    if len(peaks_idxs) != num_cycles:
                        # trim end to end at trough after last peak
                        end_idx = int(peaks_idxs[-1] + np.diff(peaks_idxs[-3:-1]).mean() / 2)
                        # trim start to num_cycles cycles before end
                        start_idx = int(peaks_idxs[-num_cycles] -np.diff(peaks_idxs[-num_cycles:-num_cycles+3]).mean() / 2)
                        
                        # trim start to start at trough before first peak
                        # start_idx = int(peaks_idxs[0] - np.diff(peaks_idxs[:3]).mean() / 2)
                        # start_idx = max(0, start_idx)
                        # end_idx = int(peaks_idxs[num_cycles-1] + np.diff(peaks_idxs[num_cycles-3:num_cycles-1]).mean() / 2)

                    # trim
                    ts_trims[j] = ts[start_idx:end_idx]
                    # ts_trims[j] = ts
                peaks[j] = [peaks_idxs - start_idx, peaks_vals]
                
    # DEBUG: trim the too short series to only 2 samples
    # ts_trims = [ts[:2] for ts, too_short in zip(ts_trims, too_short_mask) if too_short]  
    return ts_trims, too_short_mask, peaks
