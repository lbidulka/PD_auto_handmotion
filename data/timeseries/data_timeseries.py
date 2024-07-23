import numpy as np
import torch
import os

import utils.data as data_utils
import utils.features
import utils._dummy_dataset

class data_timeseries():
    def __init__(self, datasets=None, UPDRS_task=None, kpts_uniform_len=256) -> None:
        self.fingertip_kpts = [8, 12, 16, 20]  # all finger tips
        self.kpts_uniform_len = kpts_uniform_len

        self.dataset_framerates = {
            'PD4T': 25,
            'CAMERA': 60,
        }

        # self.dataset_path = dataset_path
        if datasets is not None:
            self.action = UPDRS_task
            self.datasets = datasets.split(',')
            self.load_dataset_files(self.datasets)
        
    def load_dataset_files(self, datasets):
        '''
        '''
        x, x_unscaled, x_kpts, y, subj_ids, handednesses, upscale_ratios, fps = [], [], [], [], [], [], [], []
        self.data = []
        for dataset in datasets:
            if dataset == 'dummy':
                dummy_dataset = utils._dummy_dataset.DummyDataset(seq_len=256, num_samples=50)
                self.x = dummy_dataset.data
                self.x_unscaled = dummy_dataset.x_unscaled
                self.x_kpts = dummy_dataset.x_kpts
                self.x_kpts_unscaled = dummy_dataset.x_kpts_unscaled

                self.y = dummy_dataset.labels
                self.subj_ids = dummy_dataset.subj_ids
                self.handednesses = dummy_dataset.handednesses
                self.upscale_ratios = dummy_dataset.upscale_ratios
                self.kpts_rescale_ratios = dummy_dataset.kpts_rescale_ratios
                self.dataset_framerates = dummy_dataset.dataset_framerates
                self.handcraft_feats = dummy_dataset.handcraft_feats         
                return       

            else:
                file_path = f'data/{dataset}/timeseries/{self.action}_all.npz'
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"ERR: Preprocessed Dataset file not found: {file_path}")
                self.data.append(np.load(file_path))
                x.append(self.data[-1]['samples_scaled'])
                x_unscaled.append(np.split(self.data[-1]['samples_unscaled'], 
                                        self.data[-1]['samples_unscaled_idxs'], 
                                        axis=0))
                x_kpts.append(np.split(self.data[-1]['samples_kpts_unscaled'], 
                                        self.data[-1]['samples_unscaled_idxs'], 
                                        axis=0))
            labels = self.data[-1]['labels']
            if dataset == 'PD4T':
                # repeat labels to emulate multiple raters
                labels = np.repeat(labels, 2, axis=1)
            y.append(labels)

            subj_ids.append(self.data[-1]['subj_ids'])
            handednesses.append(self.data[-1]['handednesses'])
            upscale_ratios.append(self.data[-1]['upscale_ratios'] * (self.dataset_framerates['CAMERA'] / self.dataset_framerates[dataset]))
            fps.append(self.dataset_framerates[dataset])

            
        self.x = np.vstack(x)
        self.x_unscaled = [ts for ts_list in x_unscaled for ts in ts_list]
        self.x_kpts_unscaled = [ts for ts_list in x_kpts for ts in ts_list]
        self.x_kpts, self.kpts_rescale_ratios = self.scale_to_uniform_len(self.x_kpts_unscaled, seq_len=self.kpts_uniform_len)

        # Adjust rescale ratios according to dataset framerates
        for i, fps in enumerate(fps):
            self.kpts_rescale_ratios[i] *= self.dataset_framerates['CAMERA'] / fps

        # convert x_kpts to np array by padding with last_idx
        max_len = max([ts.shape[0] for ts in self.x_kpts])
        x_kpts_pad = np.zeros((len(self.x_kpts), max_len+1, self.x_kpts[0].shape[1], 3))
        for i, ts in enumerate(self.x_kpts):
            x_kpts_pad[i, :ts.shape[0]] = ts
            x_kpts_pad[i, ts.shape[0]:] = ts.shape[0]

        self.x_kpts_unscaled = x_kpts_pad
        self.y = np.vstack(y)
        self.subj_ids = np.hstack(subj_ids)
        self.handednesses = np.hstack(handednesses)
        self.upscale_ratios = np.hstack(upscale_ratios)

        # compute handcrafted features
        self.handcraft_feats = self.get_handcraft_features(torch.tensor(self.x_kpts))

        return 
    
    def get_handcraft_features(self, P):
        '''
        Compute some handcrafted features given the input batch of pose series P

        args:
            P: (B, frame_l, joint_n, joint_d) tensor
        '''
        # Get finger-palm distances & cycle peaks/valleys/idxs
        dists = utils.features.get_finger_palm_distance(P.cpu())
        dists = [dist.mean(axis=1) for dist in dists]
        peak_idxs, peak_vals = utils.features.get_cycle_peaks(dists, keep=10, savgol_win=5, prominence=0.10, min_peak_dist=10)
        valley_idxs, valley_vals = utils.features.get_cycle_valleys(dists, peak_idxs, savgol_win=5)
        peak_idxs, peak_vals, valley_idxs, valley_vals, peak_width, valley_width = utils.features.adjust_peaks(dists, peak_idxs, peak_vals, valley_idxs, valley_vals)
        # peak_idxs = np.array(peak_idxs)
        # peak_vals = np.array(peak_vals)

        min_num_peaks = 7
        hesitation_resid_thresh = 0.2
        amp_dec_thresh = 0.9

        # UPDRS features
        num_hesitations = utils.features.get_UPDRS_num_hesitations(peak_idxs, peak_vals, 
                                                                min_num_peaks, hesitation_resid_thresh, score=False)
        amp_dec_idxs = utils.features.get_UPDRS_amplitude_decrement(peak_vals, min_num_peaks, 
                                                                    amp_dec_thresh, score=False)
        slowings = utils.features.get_UPDRS_slowing(peak_idxs, min_num_peaks, score=False)
        
        # Catch 22 features
        catch24_feats = utils.features.get_catch22_features(dists)
        
        # Cycle features
        cycle_feats = utils.features.get_cycle_features(dists, peak_idxs, peak_vals, valley_vals)
        effective_distance_completed_mean = [np.mean(eff_dist) for eff_dist in cycle_feats[0]]
        effective_distance_completed_std = [np.std(eff_dist) for eff_dist in cycle_feats[0]]
        total_distance_travelled_mean = [np.mean(dist) for dist in cycle_feats[1]]
        total_distance_travelled_std = [np.std(dist) for dist in cycle_feats[1]]
        cycle_times_mean = [np.mean(times) for times in cycle_feats[2]]
        cycle_times_std = [np.std(times) for times in cycle_feats[2]]
        total_average_speed_mean = [np.mean(spd) for spd in cycle_feats[3]]
        total_average_speed_std = [np.std(spd) for spd in cycle_feats[3]]
        smoothness_mean = [np.mean(s) for s in cycle_feats[5]]
        smoothness_std = [np.std(s) for s in cycle_feats[5]]

        # Other features
        amp_fft_var = utils.features.get_fft_var(dists)

        # combine all features into vector for each sample
        all_features = []
        for i in range(len(dists)):
            all_features.append([])
            if len(cycle_feats[0][i]) > 0:
                for c24_feat in catch24_feats[i]:
                    all_features[i].append(c24_feat)
                all_features[i].append(num_hesitations[i])
                all_features[i].append(amp_dec_idxs[i])
                all_features[i].append(slowings[i])
                all_features[i].append(effective_distance_completed_mean[i])
                all_features[i].append(effective_distance_completed_std[i])
                all_features[i].append(total_distance_travelled_mean[i])
                all_features[i].append(total_distance_travelled_std[i])
                all_features[i].append(cycle_times_mean[i])
                all_features[i].append(cycle_times_std[i])
                all_features[i].append(total_average_speed_mean[i])
                all_features[i].append(total_average_speed_std[i])
                all_features[i].append(smoothness_mean[i])
                all_features[i].append(smoothness_std[i])
                all_features[i].append(amp_fft_var[i])
            else: 
                all_features[i] = [0]*(14 + 24)
        all_features = np.array(all_features)
        features = torch.tensor(all_features, device=P.device).float()
        
        if torch.isnan(features).any():
            features[torch.isnan(features)] = 0
        return features
    
    def scale_to_uniform_len(self, x, seq_len=256):
        '''
        Scale samples in x to uniform length, either specified or max of all samples

        args:
        x: n_samples long list of (sample_len, num_kpts, 3) np arrays
        '''
        x_rescale, rescale_ratios = data_utils.scale_to_uniform_len(x, seq_len)
        return x_rescale, rescale_ratios

    def delete_idxs(self, idxs):
        '''
        Delete samples at specified indices
        '''
        self.x = np.delete(self.x, idxs, axis=0)
        self.x_unscaled = [ts for i, ts in enumerate(self.x_unscaled) if i not in idxs]
        self.x_kpts = np.delete(self.x_kpts, idxs, axis=0) #[ts for i, ts in enumerate(self.x_kpts) if i not in idxs]
        self.x_kpts_unscaled = np.delete(self.x_kpts_unscaled, idxs, axis=0)
        self.y = np.delete(self.y, idxs, axis=0)
        self.subj_ids = np.delete(self.subj_ids, idxs, axis=0)
        self.handednesses = np.delete(self.handednesses, idxs, axis=0)
        self.upscale_ratios = np.delete(self.upscale_ratios, idxs, axis=0)
        self.kpts_rescale_ratios = np.delete(self.kpts_rescale_ratios, idxs, axis=0)
        self.handcraft_feats = np.delete(self.handcraft_feats, idxs, axis=0)
    
    def get_subj_data(self, subj_ids, format='scaled',
                      use_ratio=False, combine_34=False):
        '''
        Get all samples for specified list of subjects, in specified data format
        '''
        subj_idxs = np.where(np.isin(self.subj_ids, subj_ids))[0]
        out_y = self.y[subj_idxs]
        if combine_34: out_y[out_y == 4] = 3

        if format == 'scaled':
            out_x = self.x_kpts[subj_idxs]
            palm = out_x[:, :, :1, :]
            tips = out_x[:, :, self.fingertip_kpts, :]
            out_x = np.linalg.norm(tips - palm, axis=-1)
            if use_ratio:
                subj_upscale_ratios = self.upscale_ratios[subj_idxs]
                out_x = np.append(out_x, 
                                np.repeat(subj_upscale_ratios.reshape(-1,1,1), 4, axis=2), 
                                axis=1)
        elif format == 'scaled_hf':
            out_x = self.x_kpts[subj_idxs]
            palm = out_x[:, :, :1, :]
            tips = out_x[:, :, self.fingertip_kpts, :]
            out_x = np.linalg.norm(tips - palm, axis=-1)
            # append ratio to last entry
            subj_rescale_ratios = self.upscale_ratios[subj_idxs]
            out_x = np.append(out_x, 
                            np.repeat(subj_rescale_ratios.reshape(-1,1,1), 4, axis=-1), 
                            axis=1)
            # pad and reshape handcraft features, then insert at 2nd last position
            hf = self.handcraft_feats[subj_idxs]
            hf = hf.reshape(hf.shape[0], -1, 1).repeat(4, 2)
            # hf = np.append(hf, pad, axis=1).reshape(-1, 1, out_x.shape[2], out_x.shape[3])
            # insert at 2nd last position
            out_x = np.concatenate([out_x[:,:-1], hf, out_x[:,-1:]], axis=1)
        elif format == 'unscaled':
            out_x = [ts for i, ts in enumerate(self.x_unscaled) if i in subj_idxs]

        elif format == 'unscaled_kpt':
            out_x = self.x_kpts_unscaled[subj_idxs] #[ts for i, ts in enumerate(self.x_kpts) if i in subj_idxs]
        elif format == 'scaled_kpt':
            out_x = self.x_kpts[subj_idxs]
            # append ratio to last entry
            subj_rescale_ratios = self.kpts_rescale_ratios[subj_idxs]
            out_x = np.append(out_x, 
                            np.repeat(np.repeat(subj_rescale_ratios.reshape(-1,1,1,1), 3, axis=-1), 21, axis=2), 
                            axis=1)
        elif format == 'scaled_kpt_hf':
            out_x = self.x_kpts[subj_idxs]
            # append ratio to last entry
            subj_rescale_ratios = self.kpts_rescale_ratios[subj_idxs]
            out_x = np.append(out_x, 
                            np.repeat(np.repeat(subj_rescale_ratios.reshape(-1,1,1,1), 3, axis=-1), 21, axis=2), 
                            axis=1)
            # pad and reshape handcraft features, then insert at 2nd last position
            hf = self.handcraft_feats[subj_idxs]
            pad = np.ones((hf.shape[0], out_x[0,0].reshape(-1).shape[0] - hf.shape[1]))*-99
            hf = np.append(hf, pad, axis=1).reshape(-1, 1, out_x.shape[2], out_x.shape[3])
            # insert at 2nd last position
            out_x = np.concatenate([out_x[:,:-1], hf, out_x[:,-1:]], axis=1)
        
        out_ids = self.subj_ids[subj_idxs]
        out_ids = np.array([int(id) for id in out_ids])

        return [out_x, out_y, out_ids]
    
