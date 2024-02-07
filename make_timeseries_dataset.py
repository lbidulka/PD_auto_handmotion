import os
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from data.CAMERA_expert_labels import UPDRS_med_data_KW, UPDRS_med_data_SA, too_short, trimming
import utils.features as features
import utils.data as data

# UPDRS_med_data = UPDRS_med_data_SA


RAND_SEED = 42
np.random.seed(RAND_SEED)


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--inputFolder', default='features/mat_booth/hand_movement', help='the filename to process')

    args = parser.parse_args() 
    return args

def plot_dists(finger_dists, fig_save_path):
    '''
    DEBUG: plot the data, 4 samples per figure
    '''
    fig_samples = []
    for i, data in enumerate(finger_dists):
        fig_samples.append(data)
        # plot every 4 samples
        if len(fig_samples) == 4 or i == len(finger_dists)-1:
            fig, axs = plt.subplots(4,1)
            for j, sample in enumerate(fig_samples):
                axs[j].plot(sample, linewidth=0.5)
                axs[j].set_title(f"{subj_ids[i-3+j]}, {handednesses[i-3+j]}, score: {labels[i-3+j]}")
                axs[j].set_ylim([0,2])
                axs[j].set_xlim([0, len(sample)])
                axs[j].xaxis.set_major_locator(ticker.AutoLocator())
                axs[j].xaxis.set_minor_locator(ticker.AutoMinorLocator())
            # save the figure
            fig.tight_layout()
            plt.savefig(f"{fig_save_path}fig_{i-4}_{i}.png", bbox_inches='tight', dpi=300)
            plt.close()
            fig_samples = []


if __name__ == '__main__':
    args = parse_args()

    file_list = os.listdir(args.inputFolder)
    file_list.sort()
    
    figure_save_path = 'data/debug/feat_plots/'
    PLOT_FIGS = True
    PLOT_FIGS_INTERP = False

    test_subjs = ['17000', # 0,0
                  '33749', # 0,0
                  '35623', # 0,0
                  '23160', # 0,1
                  '31769', # 1,0
                  '30893', # 1,0

                  '24318', # 1,2
                  '36297', # 1,2
                  '28411', # 2,2
                  '21401', # 2,3

                  '33164', # 3,3
                  '38256', # 3,3
                  '38519', # 4,3
                  ]

    # Get raw timeseries data from pose files
    raw_ts, trim_ts, labels, subj_ids, handednesses = data.load_raw_ts(file_list, args, UPDRS_med_data_KW, UPDRS_med_data_SA, trimming)
    
    # Convert from full hand kpts to 5 channel distance from finger to palm
    finger_dists = features.get_finger_palm_distance(raw_ts)
    finger_dists_trimmed = features.get_finger_palm_distance(trim_ts)

    # Upscale all trimmed samples to same length via interpolation (to length of longest sample)
    max_seq_len = max([data.shape[0] for data in finger_dists_trimmed])
    finger_dists_upscale = []
    for i, data in enumerate(finger_dists_trimmed):
        # interpolate each channel
        interp_data = []
        for j in range(data.shape[1]):
            xvals = np.linspace(0, data.shape[0], max_seq_len)
            interp_data.append(np.interp(xvals, np.arange(data.shape[0]), data[:,j]))
        interp_data = np.stack(interp_data, axis=1)
        finger_dists_upscale.append(interp_data)

    # DEBUG: Plotting
    if PLOT_FIGS: plot_dists(finger_dists, figure_save_path + 'full/')
    if PLOT_FIGS_INTERP: plot_dists(finger_dists_upscale, figure_save_path + 'interp/')    


    # Load up labels
    y = np.vstack(labels)
    
    # exclude samples which are too short
    for i, data in enumerate(finger_dists_upscale):
        if (subj_ids[i] in too_short.keys()) and (handednesses[i] in too_short[subj_ids[i]]):
            finger_dists_upscale.pop(i)
            subj_ids.pop(i)
            handednesses.pop(i)
            y = np.delete(y, i, axis=0)

    # Split into train/test according to subject ID
    train_idx = []
    test_idx = []
    for i, subj in enumerate(subj_ids):
        if subj in test_subjs:
            test_idx.append(i)
        else:
            train_idx.append(i)
    x_train = np.stack(finger_dists_upscale)[train_idx]
    y_train = y[train_idx]
    subj_ids_train = np.array(subj_ids)[train_idx]
    handednesses_train = np.array(handednesses)[train_idx]
    x_test = np.stack(finger_dists_upscale)[test_idx]
    y_test = y[test_idx]
    subj_ids_test = np.array(subj_ids)[test_idx]
    handednesses_test = np.array(handednesses)[test_idx]
    
    # Save
    train_data_dict_multiclass = {'samples': np.copy(x_train), 
                     'labels': np.copy(y_train),
                     'subj_ids': np.copy(subj_ids_train),
                     'handednesses': np.copy(handednesses_train)}
    test_data_dict_multiclass = {'samples': np.copy(x_test),
                     'labels': np.copy(y_test),
                     'subj_ids': np.copy(subj_ids_test),
                     'handednesses': np.copy(handednesses_test)}
    
    y_train[y_train > 0] = 1
    y_test[y_test > 0] = 1

    train_data_dict = {'samples': x_train, 
                     'labels': y_train}
    test_data_dict = {'samples': x_test,
                     'labels': y_test}
        
    print("Train data: " + str(x_train.shape))
    print("Test data: " + str(x_test.shape))

    out_path = './data/outputs/' + '/CAMERA_UPDRS/'
    np.savez(out_path + 'train.npz', **train_data_dict)
    np.savez(out_path + 'test.npz', **test_data_dict)

    np.savez(out_path + 'train_multiclass.npz', **train_data_dict_multiclass)
    np.savez(out_path + 'test_multiclass.npz', **test_data_dict_multiclass)

