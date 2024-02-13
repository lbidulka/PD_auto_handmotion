import argparse
import os
import numpy as np

from data.CAMERA_expert_labels import UPDRS_med_data_KW, UPDRS_med_data_SA, too_short, trimming
import utils.features as features
import utils.data as data


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--inputFolder', default='../my_hand/features/mat_booth/hand_movement', help='the filename to process')
    parser.add_argument('--outputFolder', default='./data/timeseries/CAMERA_UPDRS/', help='output folder for processed data')

    args = parser.parse_args() 
    return args

if __name__ == '__main__':
    args = parse_args()
    
    file_list = os.listdir(args.inputFolder)
    file_list.sort()

    # Get raw timeseries data from pose files
    raw_ts, trim_ts, labels, subj_ids, handednesses = data.load_raw_ts(file_list, args, 
                                                                       UPDRS_med_data_KW, 
                                                                       UPDRS_med_data_SA, 
                                                                       trimming)
    
    # Convert from full hand kpts to 5 channel distance from finger to palm
    finger_dists = features.get_finger_palm_distance(raw_ts)
    finger_dists_trimmed = features.get_finger_palm_distance(trim_ts)

    # Upscale all trimmed samples to same length via interpolation (to length of longest sample)
    max_seq_len = max([data.shape[0] for data in finger_dists_trimmed])
    finger_dists_upscale = []
    upscale_ratios = []
    for i, dists_data in enumerate(finger_dists_trimmed):
        # interpolate each channel
        interp_data = []
        for j in range(dists_data.shape[1]):
            xvals = np.linspace(0, dists_data.shape[0], max_seq_len)
            interp_data.append(np.interp(xvals, np.arange(dists_data.shape[0]), dists_data[:,j]))
        upscale_ratios.append(xvals.shape[0] / dists_data.shape[0])
        interp_data = np.stack(interp_data, axis=1)
        finger_dists_upscale.append(interp_data)

    # Load up labels
    y = np.vstack(labels)

    # exclude samples which are too short
    for i, data in enumerate(finger_dists_upscale):
        if (subj_ids[i] in too_short.keys()) and (handednesses[i] in too_short[subj_ids[i]]):
            finger_dists_upscale.pop(i)
            subj_ids.pop(i)
            handednesses.pop(i)
            upscale_ratios.pop(i)
            y = np.delete(y, i, axis=0)

    # Save entire dataset to single file
    train_data_dict = {'samples': np.stack(finger_dists_upscale), 
                       'labels': y, 
                       'subj_ids': np.array(subj_ids), 
                       'handednesses': np.array(handednesses),
                       'upscale_ratios': np.array(upscale_ratios),
                       }
    np.savez(os.path.join(args.outputFolder, 'handmotion_all.npz'), **train_data_dict)
