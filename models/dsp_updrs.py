import numpy as np
import scipy.signal as signal

import utils.features as features

# DSP based UPDRS severity classifier
class UPDRS_DSP():
    def __init__(self) -> None:
        self.amp_dec_thresh = 0.9

    def __call__(self, x,):
        self.get_features(x)
        preds = np.maximum(self.amp_dec, self.slowing)
        return preds

    def get_features(self, x):
        self.peak_idxs, self.peak_vals = self.get_peaks(x)
        self.amp_dec = self.get_amplitude_decrement(self.peak_vals)
        self.slowing = self.get_slowing(self.peak_idxs)
        self.num_hesitations = self.get_num_hesitations(x)
        return

    def get_num_hesitations(self, x):
        num_hesitations = None
        return num_hesitations

    def get_amplitude_decrement(self, peak_vals):
        '''
        Given input timeseries of 4-channel finger
        '''
        # get vals around each peak
        amp_decs = []
        for i in range(len(peak_vals)):

            # Check for amplitude decrement, by comparing first 3 peaks avg to last 3 peaks avg
            init_window = 3
            final_window = 3
            init_peak_avg = np.array(peak_vals[i][:init_window]).mean()
            final_peak_avg = np.array(peak_vals[i][-final_window:]).mean()

            # if ratio of final to initial peak avg is less than threshold, then amp decrement is present
            amp_dec = 0
            if final_peak_avg / init_peak_avg < self.amp_dec_thresh:
               for j in range(1, len(peak_vals[i])-3):
                    # find starting idx of amp dec
                    curr_avg = np.array(peak_vals[i][:j]).mean()
                    next_avg = np.array(peak_vals[i][j:]).mean()
                    if next_avg / curr_avg < self.amp_dec_thresh:
                        if j < 3:   # beginning
                            amp_dec = 3
                        elif j < 7: # middle
                            amp_dec = 2
                        else:       # end
                            amp_dec = 1
            
            amp_decs.append(amp_dec)

        return amp_decs
    
    def get_peaks(self, x):
        peaks, peak_vals = features.get_cycle_peaks(x)
        return peaks, peak_vals

    
    def get_slowing(self, peak_idxs):
        ''' 
        '''
        slowings = []
        for i in range(len(peak_idxs)):
            # find the time between peaks
            peak_t_diffs = np.diff(peak_idxs[i])

            # fit a line to the peak_t_diffs and get slope
            t_diff_line = np.polyfit(np.arange(len(peak_t_diffs)), peak_t_diffs, 1)
            t_trend = t_diff_line[0]

            # if slowing trend is positive, then slowing is present
            slowing = 0
            if t_trend >= 0.5:
                if t_trend < 1.0:
                    slowing = 1
                elif t_trend < 2.0:
                    slowing = 2
                else:
                    slowing = 3
            
            slowings.append(slowing)
        return slowings
    
    def train(self, x, y):
        '''
        Train the model on the given data (DUMMY, SINCE THIS IS DSP BASED)
        '''
        pass