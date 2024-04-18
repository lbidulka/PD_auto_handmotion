import numpy as np
import torch
import os

class data_timeseries():
    def __init__(self, datasets=None, CATCC_splits_path=None) -> None:
        # self.dataset_path = dataset_path
        if datasets is not None:
            self.action = 'handmotion'
            self.datasets = datasets.split(',')
            self.load_dataset_files(self.datasets)
        elif CATCC_splits_path is not None:
            self.load_CATCC_splits(CATCC_splits_path)

    def load_CATCC_splits(self, root_path):
        '''
        '''
        self.train = torch.load(os.path.join(root_path, 'train.pt'))
        self.train_frac = torch.load(os.path.join(root_path, 'train_1perc.pt'))
        self.val = torch.load(os.path.join(root_path, 'val.pt'))
        self.val_frac = torch.load(os.path.join(root_path, 'val_1perc.pt'))
        self.test = torch.load(os.path.join(root_path, 'test.pt'))

        # swap last 2 axes
        self.train['samples'] = np.swapaxes(self.train['samples'], 1, 2)
        self.train_frac['samples'] = np.swapaxes(self.train_frac['samples'], 1, 2)
        self.val['samples'] = np.swapaxes(self.val['samples'], 1, 2)
        self.val_frac['samples'] = np.swapaxes(self.val_frac['samples'], 1, 2)
        self.test['samples'] = np.swapaxes(self.test['samples'], 1, 2)
        
    def load_dataset_files(self, datasets):
        '''
        '''
        x, x_unscaled, x_kpts, y, subj_ids, handednesses, upscale_ratios = [], [], [], [], [], [], []
        self.data = []
        for dataset in datasets:
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
            upscale_ratios.append(self.data[-1]['upscale_ratios'])
            
        self.x = np.vstack(x)
        self.x_unscaled = [ts for ts_list in x_unscaled for ts in ts_list]
        self.x_kpts_unscaled = [ts for ts_list in x_kpts for ts in ts_list]
        self.x_kpts, self.kpts_rescale_ratios = self.scale_to_uniform_len(self.x_kpts_unscaled)

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
        return 
    
    def scale_to_uniform_len(self, x, max_seq_len=256):
        '''
        Scale samples in x to uniform length, either specified or max of all samples

        args:
        x: n_samples long list of (sample_len, num_kpts, 3) np arrays
        '''
        # increase max_seq_len to nearest multiple of 8
        max_seq_len = max_seq_len + (8 - max_seq_len % 8) if max_seq_len % 8 != 0 else max_seq_len
        x_rescale = []
        rescale_ratios = []
        for i, sample in enumerate(x):
            # interpolate all dims of each channel
            interp_data = []
            for j in range(sample.shape[1]):
                start = [0 for i in range(sample.shape[2])]
                stop = [sample.shape[0] for i in range(sample.shape[2])]
                xvals = np.linspace(start, stop, max_seq_len)
                _interp_data = [np.interp(xvals[:,k], np.arange(sample.shape[0]), sample[:,j,k]) for k in range(sample.shape[2])]
                interp_data.append(np.stack(_interp_data, axis=1))
            rescale_ratios.append(xvals.shape[0] / sample.shape[0])
            interp_data = np.stack(interp_data, axis=1)
            x_rescale.append(interp_data)
        x_rescale = np.array(x_rescale)
        rescale_ratios = np.array(rescale_ratios)
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
    
    def get_subj_data(self, subj_ids, format='scaled',
                      use_ratio=False, combine_34=False):
        '''
        Get all samples for specified list of subjects, in specified data format
        '''
        subj_idxs = np.where(np.isin(self.subj_ids, subj_ids))[0]
        out_y = self.y[subj_idxs]
        if combine_34: out_y[out_y == 4] = 3

        if format == 'scaled':
            out_x = self.x[subj_idxs]
            if use_ratio:
                subj_upscale_ratios = self.upscale_ratios[subj_idxs]
                out_x = np.append(out_x, 
                                np.repeat(subj_upscale_ratios.reshape(-1,1,1), 4, axis=2), 
                                axis=1)
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
        
        out_ids = self.subj_ids[subj_idxs]
        out_ids = np.array([int(id) for id in out_ids])

        return [out_x, out_y, out_ids]
    
