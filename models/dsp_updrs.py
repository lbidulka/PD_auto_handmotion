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
        for i in range(x.shape[0]):
            x[i] = x[i].mean(axis = 1)
        self.get_features(x)
        preds = np.array([self.amp_dec, self.slowing, self.num_hesitations]).max(axis=0)
        if self.task == 'binclass':
            preds = (preds > 1).astype(int)
        return preds

    def get_features(self, x):
        self.peak_idxs, self.peak_vals = self.get_peaks(x)
        self.amp_dec = self.get_amplitude_decrement(self.peak_vals)
        self.slowing = self.get_slowing(self.peak_idxs)
        self.num_hesitations = self.get_num_hesitations(self.peak_idxs, self.peak_vals)
        return

    def get_num_hesitations(self, peak_idxs, peak_vals):
        hesitations_scores = features.get_UPDRS_num_hesitations(peak_idxs, peak_vals, 
                                                                self.min_num_peaks, self.hesitation_resid_thresh)
        return hesitations_scores

    def get_amplitude_decrement(self, peak_vals):
        amp_decs = features.get_UPDRS_amplitude_decrement(peak_vals, self.min_num_peaks, self.amp_dec_thresh)
        return amp_decs
    
    def get_peaks(self, x):
        peaks, peak_vals = features.get_cycle_peaks(x, keep_10=True, savgol_win=25)
        return peaks, peak_vals

    def get_slowing(self, peak_idxs):
        slowings = features.get_UPDRS_slowing(peak_idxs, self.min_num_peaks)
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