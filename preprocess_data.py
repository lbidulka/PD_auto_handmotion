import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import scipy.signal as signal

from data.CAMERA_expert_labels import too_short, trimming
import utils.features as features
import utils.data as data


def parse_args():
    parser = argparse.ArgumentParser(description='My command-line tool')
    parser.add_argument('--dataset', default='PD4T', help='Dataset to process')   # CAMERA, PD4T
    parser.add_argument('--UPDRS_task', default='hand_movement', help='Task to process')  #hand_movement, finger_tapping
    parser.add_argument('--save_out', default=True, help='Save output to file?')   # True False

    parser.add_argument('--keep_only_agreed', default=False, help='Keep only samples where all labellers agree?')   # True False

    parser.add_argument('--smooth', default=True, help='Smooth the data?')   # True False

    args = parser.parse_args() 
    return args

def plot_dists(finger_dists, labels, subj_ids, handednesses, fig_save_path, plot_subjs, DEBUG_PEAKS=None):
    '''
    DEBUG: plot the data, 4 samples per figure
    '''
    print("Plotting...")
    fig_samples = []
    fig_titles = []
    DEBUG_SAMPLES = []
    for i, data in enumerate(finger_dists):
        if (subj_ids[i] in plot_subjs) or (len(plot_subjs) == 0):
            fig_samples.append(data)
            if DEBUG_PEAKS != None: 
                if DEBUG_PEAKS[i] != None:
                    DEBUG_SAMPLES.append(DEBUG_PEAKS[i])
                else:
                    DEBUG_SAMPLES.append(None)
            fig_titles.append(f"({i}) {subj_ids[i]}, {handednesses[i]}, score: {labels[i]}")
            # plot every 4 samples
            if len(fig_samples) == 4 or i == len(finger_dists)-1:
                fig, axs = plt.subplots(4,1)
                for j, sample in enumerate(fig_samples):
                    if DEBUG_PEAKS != None: 
                        if len(DEBUG_SAMPLES[j]) != 0:
                            axs[j].plot(DEBUG_SAMPLES[j][0], DEBUG_SAMPLES[j][1], 'ro', markersize=5)

                    axs[j].plot(sample, linewidth=0.5)
                    axs[j].set_title(fig_titles[j])
                    axs[j].set_ylim([0, 2.25])
                    axs[j].set_xlim([0, len(sample)])
                    axs[j].xaxis.set_major_locator(ticker.AutoLocator())
                    axs[j].xaxis.set_minor_locator(ticker.AutoMinorLocator())
                # save the figure
                fig.tight_layout()
                plt.savefig(f"{fig_save_path}fig_{i-4}_{i}.png", bbox_inches='tight', dpi=300)
                plt.close()
                fig_samples = []
                fig_titles = []
                DEBUG_SAMPLES = []

if __name__ == '__main__':
    args = parse_args()
    
    in_folder = f'data/{args.dataset}/pose_series/{args.UPDRS_task}/'
    out_folder = f'data/{args.dataset}/timeseries/'

    file_list = os.listdir(in_folder)
    file_list.sort()
    file_list = [os.path.join(in_folder, f) for f in file_list]

    # Get raw timeseries data from pose files
    raw_ts, trim_ts, labels, subj_ids, handednesses = data.load_raw_ts(file_list, args, trimming)
    
    # Simple filtering
    if args.smooth:
        if args.dataset == 'CAMERA':
            savgol_win = 7 #10 #7 #5
            savgol_ord = 0 #3
        elif args.dataset == 'PD4T':
            savgol_win = 3 #3  #5
            savgol_ord = 0 #3
        for i, ts in enumerate(trim_ts):
            # trim_ts[i] = signal.savgol_filter(ts, savgol_win, 3)
            for dim in range(ts.shape[1]):
                trim_ts[i][:, dim, 0] = signal.savgol_filter(ts[:, dim, 0], savgol_win, savgol_ord)
                trim_ts[i][:, dim, 1] = signal.savgol_filter(ts[:, dim, 1], savgol_win, savgol_ord)
                trim_ts[i][:, dim, 2] = signal.savgol_filter(ts[:, dim, 2], savgol_win, savgol_ord)

    # Convert from full hand kpts to 5 channel distance from finger to palm
    finger_dists = features.get_finger_palm_distance(raw_ts)
    finger_dists_trimmed = features.get_finger_palm_distance(trim_ts)

    # Do automatic trimming of samples if desired
    if args.dataset == 'PD4T':    
        AUTO_TRIM = True
        AUTO_TRIM_MASK = [True for subj_id in subj_ids]
    else:
        AUTO_TRIM = True
        if args.UPDRS_task == 'hand_movement':
            # get mask where ids are present in trimming. IE True at idx if subj_ids[idx] is in trimming
            AUTO_TRIM_MASK = [subj_id not in trimming.keys() for subj_id in subj_ids]
        else:
            AUTO_TRIM_MASK = [True for subj_id in subj_ids]
        # DEBUG_PEAKS = None
    if AUTO_TRIM:
        if args.UPDRS_task == 'hand_movement':
            peak_det_channels = [0,1,2]
        elif args.UPDRS_task == 'finger_tapping':
            peak_det_channels = [0,]

        finger_dists_trimmed, trim_ts, too_short_mask, DEBUG_PEAKS = data.auto_trim_dist_ts(finger_dists_trimmed, 
                                                                                              trim_ts,
                                                                                              AUTO_TRIM_MASK,
                                                                                              peak_det_channels,
                                                                                              smooth=False)
    else:
        too_short_mask = [False for i in range(len(subj_ids))]

    # Upscale all trimmed samples to same length via interpolation (to length of longest sample)
    # max_seq_len = max([data.shape[0] for data in finger_dists_trimmed])
    max_seq_len = 512
    # increase max_seq_len to nearest multiple of 8
    max_seq_len = max_seq_len + (8 - max_seq_len % 8) if max_seq_len % 8 != 0 else max_seq_len
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

    # DEBUG: Plotting
    figure_save_path = f'outputs/debug/feat_plots/{args.UPDRS_task}/'
    PLOT_FIGS = False
    PLOT_FIGS_INTERP = False
    # PLOT_SUBJS = ['36532', '18198', '21696', '34492', '17599', '23284', '35246', '36407']   # Change as desired
    # PLOT_SUBJS = ['16827', '28637']
    PLOT_SUBJS = []

    if PLOT_FIGS: plot_dists(finger_dists, labels, subj_ids, handednesses, 
                             figure_save_path + 'full/', PLOT_SUBJS)
    if PLOT_FIGS_INTERP: plot_dists(finger_dists_trimmed, labels, subj_ids, handednesses, 
                                    figure_save_path + 'interp/', PLOT_SUBJS, DEBUG_PEAKS)

    # DEBUG: remove PD4T bad samples
    PD4T_bad_sample_ids = [1, 4, 14, 15, 27, 44, 47, 53, 54, 59, 
                           63, 64, 65, 66, 68, 72, 79, 90, 91, 93, 
                           96, 99, 100, 101, 102, 103, 106, 108, 109, 
                           110, 118, 121, 125, 126, 128, 131, 134, 137, 138, 
                           141, 143, 146, 148, 150, 151, 152, 158, 159, 160,

                           163, 164, 165, 170, 171, 180, 182, 183, 186, 187, 
                           188, 190, 194, 195, 198, 202, 203, 204, 210, 211, 
                           213, 220, 271, 272, 273, 274, 276, 337, 343, 344, 
                           345, 403, 404, 405, 406, 483, 500, 503, 519, 571, 
                           572, 574, 595, 596, 597, 598, 623, 624, 648, 661, 

                           680, 681, 684, 685, 686, 687, 689, 690, 691, 700,
                           701, 702, 703, 708, 709, 710, 711, 728, 730, 731, 
                           742, 743, 747, 758, 759, 760, 770, 772, 806, 807, 
                           808, 809, 810, 811, 812, 813, 822, 823, 824, 825, 
                           830, 831, 833,]
    too_short_mask = [True if i in PD4T_bad_sample_ids else i for i in too_short_mask]

    # exclude samples which are too short
    for i, data in enumerate(finger_dists_upscale):
        if (subj_ids[i] in too_short.keys()) and (handednesses[i] in too_short[subj_ids[i]]):
            too_short_mask[i] = True

    trim_ts = [trim_ts[i] for i in range(len(trim_ts)) if not too_short_mask[i]]
    finger_dists_upscale = [finger_dists_upscale[i] for i in range(len(finger_dists_upscale)) if not too_short_mask[i]]
    finger_dists_trimmed = [finger_dists_trimmed[i] for i in range(len(finger_dists_trimmed)) if not too_short_mask[i]]
    subj_ids = [subj_ids[i] for i in range(len(subj_ids)) if not too_short_mask[i]]
    handednesses = [handednesses[i] for i in range(len(handednesses)) if not too_short_mask[i]]
    upscale_ratios = [upscale_ratios[i] for i in range(len(upscale_ratios)) if not too_short_mask[i]]
    y = y[~np.array(too_short_mask)]

    # Remove samples which have bad position w.r.t. the camera
    FILTER_ANGLE = 2.0
    FILTER_ANGLE_FRAC = 0.25
    for i, ts in enumerate(finger_dists_trimmed):
        # compute angle of hand to camera, if its tilted too far forward, we wont use it
        ts = trim_ts[i]
        palm_vector = (ts[:,0] + ts[:,5] + ts[:,17]) / 3
        palm_vector -= ts[:,0]
        palm_vector = palm_vector / np.linalg.norm(palm_vector, axis=1).reshape(-1,1)
        
        angle = np.arccos(palm_vector[:,2])
        angle_frac = (angle > FILTER_ANGLE).mean()
        if (angle_frac > FILTER_ANGLE_FRAC) or (len(ts) == 0):
            trim_ts.pop(i)
            finger_dists_upscale.pop(i)
            finger_dists_trimmed.pop(i)
            subj_ids.pop(i)
            handednesses.pop(i)
            upscale_ratios.pop(i)
            y = np.delete(y, i, axis=0)


    # if desired, keep only samples where all labellers agree
    if args.keep_only_agreed and (args.dataset != 'PD4T'):
        agree_mask = y[:, 0] == y[:, 1]
        trim_ts = [trim_ts[i] for i in range(len(trim_ts)) if agree_mask[i]]
        finger_dists_upscale = [finger_dists_upscale[i] for i in range(len(finger_dists_upscale)) if agree_mask[i]]
        finger_dists_trimmed = [finger_dists_trimmed[i] for i in range(len(finger_dists_trimmed)) if agree_mask[i]]
        subj_ids = [subj_ids[i] for i in range(len(subj_ids)) if agree_mask[i]]
        handednesses = [handednesses[i] for i in range(len(handednesses)) if agree_mask[i]]
        upscale_ratios = [upscale_ratios[i] for i in range(len(upscale_ratios)) if agree_mask[i]]
        y = y[agree_mask]

    # Print label distribution
    print('\nINFO: ')
    print('Num samples: ', len(y))
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
    kpts_stacked, kpts_stacked_idx = stack_ragged(trim_ts)
    
    # Save entire dataset to single file
    if args.save_out:
        out_filepath = os.path.join(out_folder, f'{args.UPDRS_task}_all.npz')
        print('\nSaving to file: ', out_filepath)
        train_data_dict = {
            'samples_scaled': np.stack(finger_dists_upscale), 
            'samples_unscaled': finger_dists_stacked, 
            'samples_kpts_unscaled': kpts_stacked,
            'samples_unscaled_idxs': finger_dists_stacked_idx,
            'labels': y, 
            'subj_ids': np.array(subj_ids), 
            'handednesses': np.array(handednesses),
            'upscale_ratios': np.array(upscale_ratios),
        }
        np.savez(out_filepath, **train_data_dict)
