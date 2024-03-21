import numpy as np
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn import tree
import utils.features as features

# DSP based UPDRS severity classifier
class Feature_ML():
    def __init__(self, task, classifier) -> None:
        self.name = 'feature_ml'
        self.task = task
        if classifier == 'svr':
            self.classifier = SVR(C=1.0, epsilon=0.2)
        elif classifier == 'dt':
            self.classifier = tree.DecisionTreeClassifier()
        elif classifier == 'linear_svm':
            self.classifier = LinearSVC(tol=1e-5)
        elif classifier == 'svm':
            self.classifier = SVC(gamma='auto')
        elif classifier == 'mlp':
            self.classifier = MLPRegressor(random_state=1, max_iter=500)
        self.clf = classifier
        self.scaler = StandardScaler()


        self.labeler_idx = 1

        self.min_num_peaks = 7
        self.amp_dec_thresh = 0.9
        self.hesitation_resid_thresh = 0.2
        self.use_ratio = False

    def __call__(self, x,):
        average_input = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            average_input[i,:] = x[i].mean(axis = 1)
        features_input = self.get_features(average_input)
        features_input = self.scaler.transform(features_input)

        features_input = features_input[:, self.selected_features]
        preds = self.classifier.predict(features_input)
        if self.clf == 'svr' or self.clf == 'mlp':
            for i in range(len(preds)):
                if preds[i]<0.5:
                    preds[i]=0
                elif preds[i]<1.5:
                    preds[i]=1
                elif preds[i]<2.5:
                    preds[i]=2
                elif preds[i]<3.5:
                    preds[i]=3
                else:
                    preds[i]=4


        return preds

    def get_features(self, x):      

        feature_vectors = features.get_catch22_features(x)

        self.peak_idxs, self.peak_vals = self.get_peaks(x)
        self.valley_idxs, self.valley_vals = self.get_valleys(x)
        
        # adjust peak and valley centres, get peak and valley width
        self.peak_idxs, self.peak_vals, self.valley_idxs, self.valley_vals, self.peak_width, self.valley_width = features.adjust_peaks(x, self.peak_idxs, self.peak_vals, self.valley_idxs, self.valley_vals)

        # maybe moving these functions to features.py since they are shared between dsp and ml
        self.amp_dec = self.get_amplitude_decrement(self.peak_vals)
        self.slowing = self.get_slowing(self.peak_idxs)
        self.num_hesitations = self.get_num_hesitations(x)
        
        # fft
        self.amp_fft_var = features.get_fft_var(x)

        # get cycle features
        self.effective_distance_completed, self.total_distance_travelled, self.cycle_times, self.effective_average_speed, self.total_average_speed, self.smoothness = features.get_cycle_features(x, self.peak_idxs, self.peak_vals, self.valley_vals)

        self.effective_distance_completed_mean = []
        self.effective_distance_completed_std = []
        self.total_distance_travelled_mean = []
        self.total_distance_travelled_std = []
        self.cycle_times_mean = []
        self.cycle_times_std = []
        self.effective_average_speed_mean = []
        self.effective_average_speed_std = []
        self.total_average_speed_mean = []
        self.total_average_speed_std = []
        self.smoothness_mean = []
        self.smoothness_std = []

        self.num_hesitations_peaks_th = []
        self.num_hesitations_valleys_th = []

        self.peak_slowing = []
        self.valley_slowing = []

        self.cycle_times_slowing = []
        self.effective_distance_completed_slowing = []
        self.total_distance_travelled_slowing = []
        self.effective_average_speed_slowing = []
        self.total_average_speed_slowing = []

        self.peak_fatigue = []
        self.valley_fatigue = []

        self.cycle_times_fatigue = []
        self.effective_distance_completed_fatigue = []
        self.total_distance_travelled_fatigue = []
        self.effective_average_speed_fatigue = []
        self.total_average_speed_fatigue = []

        for id in range(x.shape[0]):

            if len(self.effective_distance_completed[id])>0:

                self.effective_distance_completed_mean.append(np.mean(self.effective_distance_completed[id]))
                self.effective_distance_completed_std.append(np.std(self.effective_distance_completed[id]))
                feature_vectors[id].append(self.effective_distance_completed_mean[-1])
                feature_vectors[id].append(self.effective_distance_completed_std[-1])


                self.cycle_times_mean.append(np.mean(self.cycle_times[id]))
                self.cycle_times_std.append(np.std(self.cycle_times[id]))
                feature_vectors[id].append(self.cycle_times_mean[-1])
                feature_vectors[id].append(self.cycle_times_std[-1])

                self.total_distance_travelled_mean.append(np.mean(self.total_distance_travelled[id]))
                self.total_distance_travelled_std.append(np.std(self.total_distance_travelled[id]))
                feature_vectors[id].append(self.total_distance_travelled_mean[-1])
                feature_vectors[id].append(self.total_distance_travelled_std[-1])

                self.effective_average_speed_mean.append(np.mean(self.effective_average_speed[id]))
                self.effective_average_speed_std.append(np.std(self.effective_average_speed[id]))
                feature_vectors[id].append(self.effective_average_speed_mean[-1])
                feature_vectors[id].append(self.effective_average_speed_std[-1])

                self.total_average_speed_mean.append(np.mean(self.total_average_speed[id]))
                self.total_average_speed_std.append(np.std(self.total_average_speed[id]))
                feature_vectors[id].append(self.total_average_speed_mean[-1])
                feature_vectors[id].append(self.total_average_speed_std[-1])

                self.smoothness_mean.append(np.mean(self.smoothness[id]))
                self.smoothness_std.append(np.std(self.smoothness[id]))
                feature_vectors[id].append(self.smoothness_mean[-1])
                feature_vectors[id].append(self.smoothness_std[-1])

                feature_vectors[id].append(np.mean(self.peak_width[id]))
                feature_vectors[id].append(np.std(self.peak_width[id]))
                feature_vectors[id].append(np.mean(self.valley_width[id]))
                feature_vectors[id].append(np.std(self.valley_width[id]))

            else:
                for i in range(16):
                    feature_vectors[id].append(0)

            self.num_hesitations_peaks_th.append(features.get_hesitations_peaks_th(self.peak_vals[id], threshold=1))
            self.num_hesitations_valleys_th.append(features.get_hesitations_valleys_th(self.valley_vals[id], threshold=0.5))
            feature_vectors[id].append(float(self.num_hesitations_peaks_th[-1]))
            feature_vectors[id].append(float(self.num_hesitations_valleys_th[-1]))

            # self.peak_slowing.append(features.get_slowing(self.peak_vals[id]))
            # self.valley_slowing.append(features.get_slowing(self.valley_vals[id]))
            # feature_vectors[id].append(self.peak_slowing[-1])
            # feature_vectors[id].append(self.valley_slowing[-1])

            # self.cycle_times_slowing.append(features.get_slowing(self.cycle_times[id]))
            # self.effective_distance_completed_slowing.append(features.get_slowing(self.effective_distance_completed[id]))
            # self.total_distance_travelled_slowing.append(features.get_slowing(self.total_distance_travelled[id]))
            # self.effective_average_speed_slowing.append(features.get_slowing(self.effective_average_speed[id]))
            # self.total_average_speed_slowing.append(features.get_slowing(self.total_average_speed[id]))
            # feature_vectors[id].append(self.cycle_times_slowing[-1])
            # feature_vectors[id].append(self.effective_distance_completed_slowing[-1])
            # feature_vectors[id].append(self.total_distance_travelled_slowing[-1])
            # feature_vectors[id].append(self.effective_average_speed_slowing[-1])
            # feature_vectors[id].append(self.total_average_speed_slowing[-1])

            # self.peak_fatigue.append(features.get_fatigue_minmax(self.peak_vals[id]))
            # self.valley_fatigue.append(features.get_fatigue_minmax(self.valley_vals[id]))
            # feature_vectors[id].append(self.peak_fatigue[-1])
            # feature_vectors[id].append(self.valley_fatigue[-1])

            # self.cycle_times_fatigue.append(features.get_fatigue_minmax(self.cycle_times[id]))
            # self.effective_distance_completed_fatigue.append(features.get_fatigue_minmax(self.effective_distance_completed[id]))
            # self.total_distance_travelled_fatigue.append(features.get_fatigue_minmax(self.total_distance_travelled[id]))
            # self.effective_average_speed_fatigue.append(features.get_fatigue_minmax(self.effective_average_speed[id]))
            # self.total_average_speed_fatigue.append(features.get_fatigue_minmax(self.total_average_speed[id]))
            # feature_vectors[id].append(self.cycle_times_fatigue[-1])
            # feature_vectors[id].append(self.effective_distance_completed_fatigue[-1])
            # feature_vectors[id].append(self.total_distance_travelled_fatigue[-1])
            # feature_vectors[id].append(self.effective_average_speed_fatigue[-1])
            # feature_vectors[id].append(self.total_average_speed_fatigue[-1])

            feature_vectors[id].append(float(self.amp_dec[id]))
            feature_vectors[id].append(float(self.num_hesitations[id]))
            feature_vectors[id].append(float(self.slowing[id]))

            feature_vectors[id].append(self.amp_fft_var[id])
            
        return  np.asarray(feature_vectors)
    
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
        peaks, peak_vals = features.get_cycle_peaks(x, keep=10, savgol_win=25)
        return peaks, peak_vals
    
    def get_valleys(self,x):
        # find valleys between peaks
        valleys, valley_vals = features.get_cycle_valleys(x, self.peak_idxs)
        return valleys, valley_vals

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
    
    def information_gain_feature_selection(self, training_data, training_label, n_selected_feature = 20):
                # information gain
        ig = mutual_info_regression(training_data, training_label)
        
        # Create a dictionary of feature importance scores
        feature_scores = {}
        for i in range(len(ig)):
            feature_scores[i] = ig[i]
        # Sort the features by importance score in descending order
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        # print(sorted_features[:n_selected_feature])

        self.selected_features = []
        n_features = 0
        for feature, score in sorted_features:
            # print("Feature:", feature, "Score:", score)
            # if score>0.10:
                # selected_features.append(feature)
            self.selected_features.append(feature)
            n_features+=1
            if n_features>n_selected_feature:
                break

    def train(self, x, y, x_val=None, y_val=None,):
        average_input = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            average_input[i,:] = x[i].mean(axis = 1)
        features_input = self.get_features(average_input)

        # normalization
        self.scaler.fit(features_input)
        features_input = self.scaler.transform(features_input)

        label = y[:, self.labeler_idx]
        self.information_gain_feature_selection(features_input, label, n_selected_feature=20)
        features_input = features_input[:, self.selected_features]
        self.classifier = self.classifier.fit(features_input, label)

    def init_model(self,):
        '''
        Initialize the model (DUMMY, SINCE THIS IS DSP BASED)
        '''
        pass