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
        x, x_unscaled, y, subj_ids, handednesses, upscale_ratios = [], [], [], [], [], []
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
        self.y = np.vstack(y)
        self.subj_ids = np.hstack(subj_ids)
        self.handednesses = np.hstack(handednesses)
        self.upscale_ratios = np.hstack(upscale_ratios)
            
        # self.data = np.load(file_path)
        # self.x = self.data['samples_scaled']
        # self.x_unscaled = np.split(self.data['samples_unscaled'], 
        #                            self.data['samples_unscaled_idxs'], 
        #                            axis=0)
        # self.y = self.data['labels']
        # self.subj_ids = self.data['subj_ids']
        # self.handednesses = self.data['handednesses']
        # self.upscale_ratios = self.data['upscale_ratios']
        return 
    
    def get_subj_data(self, subj_ids, use_ratio=False, unscaled=False):
        '''
        Get all samples for specified list of subjects
        '''
        subj_idxs = np.where(np.isin(self.subj_ids, subj_ids))[0]
        out_y = self.y[subj_idxs]
        if unscaled:
            out_x = [ts for i, ts in enumerate(self.x_unscaled) if i in subj_idxs]
        else:
            out_x = self.x[subj_idxs]
            if use_ratio:
                subj_upscale_ratios = self.upscale_ratios[subj_idxs]
                out_x = np.append(out_x, 
                                np.repeat(subj_upscale_ratios.reshape(-1,1,1), 4, axis=2), 
                                axis=1)
        return [out_x, out_y]
    
