import numpy as np


class data_timeseries():
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        self.load_dataset_file(dataset_path)
        
    def load_dataset_file(self, file_path):
        self.data = np.load(file_path)
        self.x = self.data['samples_scaled']
        self.x_unscaled = np.split(self.data['samples_unscaled'], 
                                   self.data['samples_unscaled_idxs'], 
                                   axis=0)
        self.y = self.data['labels']
        self.subj_ids = self.data['subj_ids']
        self.handednesses = self.data['handednesses']
        self.upscale_ratios = self.data['upscale_ratios']
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
    
