import numpy as np
import scipy.signal as signal


def get_cycle_peaks(x, min_peak_dist=60, keep_10=True, savgol_win=10, prominence=0.25):
    # find cycle peaks and get avg vals around at peak
    peaks_idxs = []
    peak_vals = []
    for i in range(x.shape[0]):
        smoothed = signal.savgol_filter(x[i].mean(axis=1), savgol_win, 3)
        peaks_idxs.append(signal.find_peaks(smoothed, height=0.35, distance=min_peak_dist, prominence=prominence)[0])
        # avg val around each peak
        peak_window = 2
        peak_vals.append([])
        for p in peaks_idxs[i]:
            peak_vals[i].append(x[i].mean(axis=1)[p-peak_window:p+peak_window].mean())
        peak_vals[i] = np.array(peak_vals[i])
        
        # take only top 10 peaks
        if (len(peak_vals[i]) > 10) and keep_10:
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
        palm_centre = (ts[:,0] + ts[:,5] + ts[:,17]) / 3
        # palm_centre = ((ts[:,5] - ts[:,0]) + (ts[:,9] - ts[:,0]) + (ts[:,13] - ts[:,0]) + (ts[:,17] - ts[:,0])) / 8
        # palm_centre = (ts[:,0] + (ts[:,5]) + (ts[:,9]) + (ts[:,13]) + (ts[:,17])) / 5
        
        # palm_vector = ts[:,0] - ts[:,9]
        palm_vector = ((ts[:,5] - ts[:,0]) + (ts[:,9] - ts[:,0]) + (ts[:,13] - ts[:,0]) + (ts[:,17] - ts[:,0])) / 4
        palm_size = np.linalg.norm(palm_vector, axis=1).reshape(-1,1) #.mean()

        # get distance from each finger to palm center (excluding thumb)
        fingertip_idxs = [8, 12 ,16, 20]
        dist = (ts[:,fingertip_idxs] - palm_centre.reshape(palm_centre.shape[0],1,palm_centre.shape[1]))
        dist = np.linalg.norm(dist, axis=2)
        norm_dists.append(dist / palm_size)
        # norm_dists.append(dist)
    return norm_dists

def get_hesitations(peak_vals, peak_idxs, residual_thresh):
    '''
    Find hesitations in peak series, given peak vals and peak idxs and residual threshold
    (Following Ryans Definition in: "Clinically-informed Automated Assessment of Finger Tapping
    Videos in Parkinson’s Disease")
    '''
    poly_coeffs = np.polyfit(peak_idxs, peak_vals, 1)
    residuals = np.abs(peak_vals - (poly_coeffs[0]*peak_idxs + poly_coeffs[1]))
    hesitations = np.where(residuals > residual_thresh)[0]
    return hesitations

