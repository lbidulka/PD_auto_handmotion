import numpy as np
import scipy.signal as signal


def get_cycle_peaks(x):
    # find cycle peaks and get avg vals around at peak
    peaks_idxs = []
    peak_vals = []
    for i in range(x.shape[0]):
        smoothed = signal.savgol_filter(x[i].mean(axis=1), 25, 3)
        peaks_idxs.append(signal.find_peaks(smoothed, distance=60, )[0])
        # peaks.append(signal.find_peaks_cwt(x[i].mean(axis=1), widths=25,))
        # avg val around each peak
        peak_window = 2
        peak_vals.append([])
        for p in peaks_idxs[i]:
            peak_vals[i].append(x[i].mean(axis=1)[p-peak_window:p+peak_window].mean())
        peak_vals[i] = np.array(peak_vals[i])

        # take only top 10 peaks
        if len(peak_vals[i]) > 10:
            # get ranked idxs of peak vals
            peak_vals_top = np.copy(peak_vals[i])
            peak_vals_top = np.argsort(peak_vals[i])[::-1]
            rej_idxs = peak_vals_top[10:]
            # pop the rejected idxs
            peak_vals[i] = np.delete(peak_vals[i], rej_idxs)
            peaks_idxs[i] = np.delete(peaks_idxs[i], rej_idxs)

    return peaks_idxs, peak_vals  

def get_finger_palm_distance(raw_data):
    '''
    Takes in raw full hand kpt timeseries data and outputs 4 channel distance from fingers to palm (no thumb)
    inputs:
        raw_data: length N list of kpt series w/shape (num_frames, num_kpts, 3)
    '''
    norm_dists = []
    for ts in raw_data:
        # get palm center and size
        palm_centre = (ts[:,0,:] + ts[:,5,:] + ts[:,17,:]) / 3
        palm_size = np.linalg.norm(ts[:,0,:] - ts[:,9,:], axis=1).mean()
        # get distance from each finger to palm center (excluding thumb)
        fingertip_idxs = [8, 12 ,16, 20]
        dist = (ts[:,fingertip_idxs] - palm_centre.reshape(palm_centre.shape[0],1,palm_centre.shape[1]))
        dist = np.linalg.norm(dist, axis=2)
        norm_dists.append(dist / palm_size)
    return norm_dists

