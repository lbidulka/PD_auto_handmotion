import numpy as np
import scipy.signal as signal
import pycatch22


def get_cycle_peaks(x, min_peak_dist=60, keep=10, savgol_win=10, prominence=0.25):
    # find cycle peaks and get avg vals around at peak
    peak_idxs = []
    peak_vals = []
    for i in range(x.shape[0]):

        if i == 541:
            a=1
        smoothed = signal.savgol_filter((x[i]), savgol_win, 3)
        peak_idxs.append(signal.find_peaks(smoothed, height=0.35, distance=min_peak_dist, prominence=prominence)[0])
        # avg val around each peak
        peak_window = 2
        peak_vals.append([])
        for p in peak_idxs[i]:
            peak_vals[i].append(x[i][p-peak_window:p+peak_window].mean())
        peak_vals[i] = np.array(peak_vals[i])
        
        # take only top 10 peaks
        if (len(peak_vals[i]) > keep):
            # get ranked idxs of peak vals
            peak_vals_top = np.copy(peak_vals[i])
            peak_vals_top = np.argsort(peak_vals[i])[::-1]
            rej_idxs = peak_vals_top[keep:]
            # pop the rejected idxs
            peak_vals[i] = np.delete(peak_vals[i], rej_idxs)
            peak_idxs[i] = np.delete(peak_idxs[i], rej_idxs)

    return peak_idxs, peak_vals  

def get_cycle_valleys(x, peak_idxs):
    # find valley between peaks

    valley_idxs = []
    valley_vals = []
    for i in range(x.shape[0]):
        valley_idxs.append([])
        valley_vals.append([])
        peaks = peak_idxs[i]
        for j in range(len(peaks)-1):
            data = x[i, peaks[j]:peaks[j+1]]
            data_max = np.max(data)
            data = data_max - data
            data = np.expand_dims(data, 0)
            valley, valley_val = get_cycle_peaks(data, keep=1, savgol_win=25)
            if len(valley[0]) == 0:
                valley = int(len(data[0])/2)
                valley_val = data[0, valley]
            else:
                valley = valley[0][0]
                valley_val = valley_val[0][0]
            valley_val = data_max - valley_val
            valley_idxs[i].append(valley+peaks[j])
            valley_vals[i].append(valley_val)
    return valley_idxs, valley_vals

def adjust_peaks(input, peak_idxs, peak_vals, valley_idxs, valley_vals):
    new_peak_idxs=[]
    new_peak_vals=[]
    peak_width = []
    new_valley_idxs=[]
    new_valley_vals=[]
    valley_width = []
    for id in range(input.shape[0]):
        # adjust peak and valley centres, and get width for each peak and valley centers
        new_peak_idxs_sample = np.zeros(np.shape(peak_idxs[id]))
        new_peak_vals_sample = np.zeros(np.shape(peak_idxs[id]))
        peak_width_sample = np.zeros(np.shape(peak_idxs[id]))
        new_valley_idxs_sample = np.zeros(np.shape(valley_idxs[id]))
        new_valley_vals_sample = np.zeros(np.shape(valley_idxs[id]))
        valley_width_sample = np.zeros(np.shape(valley_idxs[id]))

        for i in range(len(peak_idxs[id])):
            x = peak_idxs[id][i]
            y = peak_vals[id][i]
            y_interval = 10

            if i > 0 and abs(y - valley_vals[id][i-1])<y_interval:
                y_interval = abs(y - valley_vals[id][i-1])
            
            if i < len(valley_vals[id]) and abs(y - valley_vals[id][i])<y_interval:
                y_interval = abs(y - valley_vals[id][i])

            if y_interval == 10:
                continue

            x_min = x
            x_max = x
            while x_min>=0 and abs(input[id][x_min]-y)<y_interval*0.1:
                x_min-=1
            x_min+=1

            while x_max<len(input[id]) and abs(input[id][x_max]-y)<y_interval*0.1:
                x_max+=1
            x_max-=1

            peak_width_sample[i] = x_max-x_min
            x_new = int((x_min+x_max)/2)
            y_new = input[id][x_new]
            new_peak_idxs_sample[i] = x_new
            new_peak_vals_sample[i] = y_new


        for i in range(len(valley_idxs[id])):
            x = valley_idxs[id][i]
            y = valley_vals[id][i]
            y_interval = 10

            if i >= 0 and abs(y - peak_vals[id][i])<y_interval:
                y_interval = abs(y - peak_vals[id][i])
            
            if i+1 < len(peak_vals[id]) and abs(y - peak_vals[id][i+1])<y_interval:
                y_interval = abs(y - peak_vals[id][i+1])

            if y_interval == 10:
                continue

            x_min = x
            x_max = x
            while x_min>=0 and abs(input[id][x_min]-y)<y_interval*0.1:
                x_min-=1
            x_min+=1

            while x_max<len(input[id]) and abs(input[id][x_max]-y)<y_interval*0.1:
                x_max+=1
            x_max-=1

            valley_width_sample[i] = x_max-x_min
            x_new = int((x_min+x_max)/2)
            y_new = input[id][x_new]
            new_valley_idxs_sample[i] = x_new
            new_valley_vals_sample[i] = y_new
        new_peak_idxs.append(new_peak_idxs_sample)
        new_peak_vals.append(new_peak_vals_sample)
        new_valley_idxs.append(new_valley_idxs_sample)
        new_valley_vals.append(new_valley_vals_sample)
        peak_width.append(peak_width_sample)
        valley_width.append(valley_width_sample)

    return new_peak_idxs, new_peak_vals, new_valley_idxs, new_valley_vals, peak_width, valley_width

def get_cycle_features(x, peak_idxs, peak_vals, valley_vals):
    effective_distance_completed = []
    total_distance_travelled = []
    cycle_times = []
    effective_average_speed = []
    total_average_speed = []
    smoothness = []

    for id in range(x.shape[0]):
        effective_distance_completed.append([])
        total_distance_travelled.append([])
        cycle_times.append([])
        effective_average_speed.append([])
        total_average_speed.append([])
        smoothness.append([])       

        for i in range(len(peak_vals[id])-1):
            # effective distance
            effective_distance_completed[id].append(peak_vals[id][i] - valley_vals[id][i]+peak_vals[id][i+1] - valley_vals[id][i])

            # total distance
            x_start, x_end = int(peak_idxs[id][i]), int(peak_idxs[id][i+1])
            d = 0
            for j in range(x_start, x_end+1):
                d+=abs(np.mean(x[id][j])-np.mean(x[id][j+1]))
            total_distance_travelled[id].append(d)

            # cycle time
            cycle_times[id].append(x_end - x_start)

            # effective average speed
            if cycle_times[id][-1]>0:
                effective_average_speed[id].append(effective_distance_completed[id][-1]/cycle_times[id][-1])
                total_average_speed[id].append(total_distance_travelled[id][-1]/cycle_times[id][-1])
            else:
                effective_average_speed[id].append(0)
                total_average_speed[id].append(0)
            
            # smoothness
            if total_distance_travelled[id][-1]>0:
                smoothness[id].append(effective_distance_completed[id][-1]/total_distance_travelled[id][-1])
            else:
                smoothness[id].append(0)

    return effective_distance_completed, total_distance_travelled, cycle_times, effective_average_speed, total_average_speed, smoothness

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

def get_hesitations_peaks_th(peak_vals, threshold):
    m = np.median(peak_vals)
    points_hes = 0
    for i, y in enumerate(peak_vals):
        if y<threshold or y<m-0.3:
            points_hes+=1
    return points_hes

def get_hesitations_valleys_th(valley_vals, threshold):
    m = np.median(valley_vals)
    points_hes = 0
    for i, y in enumerate(valley_vals):
        if y>threshold or y>m+0.3:
            points_hes+=1
    return points_hes

def get_fft_var(x):
    fft_var = []
    for id in range(x.shape[0]):
        fft_var.append(np.var(np.fft.fft(x[id])))
    
    return fft_var

def get_slowing(input):
    if len(input) >0:
        half = int(len(input)/2)
        result = np.mean(input[:half])/np.mean(input[half:])
    else:
        return 10
    if np.isnan(result):
        return 10
    else:
        return result

def get_fatigue_minmax(input):
    if len(input) >0:
        result =  1-np.min(input)/np.max(input)
    else:
        return 10
    if np.isnan(result):
        return 10
    else:
        return result
    
def get_catch22_features(input):
    results = []
    for i in range(input.shape[0]):
        result = pycatch22.catch22_all(input[i],catch24=True)
        results.append(result['values'])
    
    return results
