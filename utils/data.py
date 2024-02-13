import os
import numpy as np
from scipy import io


def remove_unlabeled(x, y, subj_ids=None, handednesses=None, 
                     combine_34=True, rej_either=True):
    '''
    Remove samples with label == -1
    '''
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

def load_raw_ts(file_list, args, UPDRS_med_data_KW, UPDRS_med_data_SA, trimming):
    '''
    Loads raw hand pose timeseries, trims them, and returns them as a list along 
    with annotation info
    '''
    raw_ts = []
    trim_ts = []
    labels = []
    subj_ids = []
    handednesses = []

    for file in file_list:
        '''
        
        '''
        file_path = os.path.join(args.inputFolder, file)
        temp = file.split('_')
        person_id = temp[0]
        date = temp[1]
        handedness = temp[-2]

        if (person_id in UPDRS_med_data_KW.keys()) or (person_id in UPDRS_med_data_SA.keys()):
            try:
                label_KW = UPDRS_med_data_KW[person_id]['hand_movement'][f'{handedness}_open_close'][date]
            except:
                label_KW = -1
            try:
                label_SA = UPDRS_med_data_SA[person_id]['hand_movement'][f'{handedness}_open_close'][date]
            except:
                label_SA = -1
            
            if label_KW is None:
                label_KW = -1
            if label_SA is None:
                label_SA = -1

            try:
                segments = trimming[person_id][handedness]
            except:
                # default to use entire signal
                segments = {0: {'start': 0, 'end': -1}}
            # segments = {0: {'start': 0, 'end': -1}}

            print(file)
            print('labels KW/SA: ', label_KW, label_SA)
            matfile = io.loadmat(file_path)
            
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
                    labels.append([label_KW, label_SA])
                    subj_ids.append(person_id)
                    handednesses.append(handedness)
                else:
                    print(f"{person_id}, {handedness}: Empty data block, skipping")
        else:
            print("Subject not in UPDRS_med_data: " + str(person_id))
    
    return raw_ts, trim_ts, labels, subj_ids, handednesses