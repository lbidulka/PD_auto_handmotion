import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data.CAMERA_expert_labels import UPDRS_med_data_KW, UPDRS_med_data_SA, too_short, trimming
import utils.features as features
import utils.data as data


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--inputFolder', default='../my_hand/features/mat_booth/hand_movement', help='the filename to process')
    parser.add_argument('--outputFolder', default='./data/timeseries/CAMERA_UPDRS/', help='output folder for processed data')

    args = parser.parse_args() 
    return args

def plot_dists(finger_dists, labels, subj_ids, handednesses, fig_save_path, plot_subjs):
    '''
    DEBUG: plot the data, 4 samples per figure
    '''
    fig_samples = []
    fig_titles = []
    for i, data in enumerate(finger_dists):
        if subj_ids[i] in plot_subjs:
            fig_samples.append(data)
            fig_titles.append(f"{subj_ids[i]}, {handednesses[i]}, score: {labels[i]}")
            # plot every 4 samples
            if len(fig_samples) == 4 or i == len(finger_dists)-1:
                fig, axs = plt.subplots(4,1)
                for j, sample in enumerate(fig_samples):
                    axs[j].plot(sample, linewidth=0.5)
                    axs[j].set_title(fig_titles[j])
                    axs[j].set_ylim([0,2])
                    axs[j].set_xlim([0, len(sample)])
                    axs[j].xaxis.set_major_locator(ticker.AutoLocator())
                    axs[j].xaxis.set_minor_locator(ticker.AutoMinorLocator())
                # save the figure
                fig.tight_layout()
                plt.savefig(f"{fig_save_path}fig_{i-4}_{i}.png", bbox_inches='tight', dpi=300)
                plt.close()
                fig_samples = []
                fig_titles = []

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
    # increase max_seq_len to nearest multiple of 8
    max_seq_len = max_seq_len + (8 - max_seq_len % 8) if max_seq_len % 8 != 0 else max_seq_len
    finger_dists_upscale = []
    upscale_ratios = []
    finger_dists_trimmed_pad = []
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

    # DEBUG: Plotting
    figure_save_path = 'outputs/debug/feat_plots/'
    PLOT_FIGS = False
    PLOT_FIGS_INTERP = False
    PLOT_SUBJS = ['36532', '18198', '21696', '34492', '17599', '23284', '35246', '36407']   # Change as desired

    if PLOT_FIGS: plot_dists(finger_dists, labels, subj_ids, handednesses, 
                             figure_save_path + 'full/', PLOT_SUBJS)
    if PLOT_FIGS_INTERP: plot_dists(finger_dists_trimmed, labels, subj_ids, handednesses, 
                                    figure_save_path + 'interp/', PLOT_SUBJS)

    # exclude samples which are too short
    for i, data in enumerate(finger_dists_upscale):
        if (subj_ids[i] in too_short.keys()) and (handednesses[i] in too_short[subj_ids[i]]):
            finger_dists_upscale.pop(i)
            finger_dists_trimmed.pop(i)
            subj_ids.pop(i)
            handednesses.pop(i)
            upscale_ratios.pop(i)
            y = np.delete(y, i, axis=0)

    # Print label distribution
    print('\nINFO: ')
    print('Num samples: ', len(y))
    # print('Label Distribution: ', np.unique(y, return_counts=True))
    for i in range(y.shape[1]):
        print(f'Labeller {i} dist: {np.unique(y[:,i], return_counts=True)}')
    num_unique_subjs = len(np.unique(subj_ids))
    print('Num unique subjects: ', num_unique_subjs)

    # Setup ragged array of original samples
    def stack_ragged(array_list, axis=0):
        lengths = [np.shape(a)[axis] for a in array_list]
        idx = np.cumsum(lengths[:-1])
        stacked = np.concatenate(array_list, axis=axis)
        return stacked, idx
    finger_dists_stacked, finger_dists_stacked_idx = stack_ragged(finger_dists_trimmed)
    
    # Save entire dataset to single file
    out_filepath = os.path.join(args.outputFolder, 'handmotion_all.npz')
    print('\nSaving to file: ', out_filepath)
    train_data_dict = {'samples_scaled': np.stack(finger_dists_upscale), 
                       'samples_unscaled': finger_dists_stacked, 
                       'samples_unscaled_idxs': finger_dists_stacked_idx,
                       'labels': y, 
                       'subj_ids': np.array(subj_ids), 
                       'handednesses': np.array(handednesses),
                       'upscale_ratios': np.array(upscale_ratios),
                       }
    np.savez(out_filepath, **train_data_dict)
