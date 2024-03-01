import numpy as np
import scipy.signal as signal

import utils.features as features

# DSP based UPDRS severity classifier
class UPDRS_DSP():
    def __init__(self, task) -> None:
        self.name = 'updrs_dsp'
        self.task = task

        self.min_num_peaks = 7
        self.amp_dec_thresh = 0.9
        self.hesitation_resid_thresh = 0.2
        self.use_ratio = False

    def __call__(self, x,):
        self.get_features(x)
        preds = np.array([self.amp_dec, self.slowing, self.num_hesitations]).max(axis=0)
        if self.task == 'binclass':
            preds = (preds > 1).astype(int)
        return preds

    def get_features(self, x):
        self.peak_idxs, self.peak_vals = self.get_peaks(x)
        self.amp_dec = self.get_amplitude_decrement(self.peak_vals)
        self.slowing = self.get_slowing(self.peak_idxs)
        self.num_hesitations = self.get_num_hesitations(x)
        return

    def get_num_hesitations(self, x):

        '''
        alg:
        
        1. get peaks and peak idxs
        2. fit linear model to peak series
        3. find residuals of linear model for the peaks
        4. count num peaks with residuals > hesitation threshold
        '''
        hesitations_scores = []
        for i in range(x.shape[0]):
            if len(self.peak_idxs[i]) < self.min_num_peaks:
                hesitations_score = 0
            else:
                peak_idxs = self.peak_idxs[i]
                peak_vals = self.peak_vals[i]
                hesitations = features.get_hesitations(peak_vals, peak_idxs, self.hesitation_resid_thresh)
                num_hesitations = len(hesitations)
                # Score logic from MDS-UPDRS:
                if num_hesitations > 5:
                    hesitations_score = 3
                elif num_hesitations >= 3:
                    hesitations_score = 2
                elif num_hesitations >= 1:
                    hesitations_score = 1         
                else:
                    hesitations_score = 0
            hesitations_scores.append(hesitations_score)
        return hesitations_scores

    def get_amplitude_decrement(self, peak_vals):
        '''
        Given input timeseries of 4-channel finger
        '''
        # get vals around each peak
        amp_decs = []
        for i in range(len(peak_vals)):
            if len(peak_vals[i]) < self.min_num_peaks:
                amp_dec = 0
            else:
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
        peaks, peak_vals = features.get_cycle_peaks(x, keep_10=True, savgol_win=25)
        return peaks, peak_vals

    def get_slowing(self, peak_idxs):
        ''' 
        '''            
        slowings = []
        for i in range(len(peak_idxs)):
            if len(peak_idxs[i]) < self.min_num_peaks:
                slowing = 0
            else:
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
    
    def train(self, x, y, x_val=None, y_val=None,):
        '''
        Train the model on the given data (DUMMY, SINCE THIS IS DSP BASED)
        '''
        pass

    def init_model(self,):
        '''
        Initialize the model (DUMMY, SINCE THIS IS DSP BASED)
        '''
        pass