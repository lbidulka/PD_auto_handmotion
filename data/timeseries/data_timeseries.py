import numpy as np


class data_timeseries():
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        data = self.load_dataset_file(dataset_path)
        self.x, self.y = data[0], data[1], 
        self.subj_ids, self.handednesses = data[2], data[3]
        self.upscale_ratios = data[4]
        
    def load_dataset_file(self, file_path):
        data = np.load(file_path)
        x = data['samples']
        y = data['labels']
        subj_ids = data['subj_ids']
        handednesses = data['handednesses']
        upscale_ratios = data['upscale_ratios']
        return x, y, subj_ids, handednesses, upscale_ratios
    
    def get_subj_data(self, subj_ids, use_ratio=False):
        '''
        Get all samples for specified list of subjects
        '''
        subj_idxs = np.where(np.isin(self.subj_ids, subj_ids))[0]

        # get data for these subjects
        subj_upscale_ratios = self.upscale_ratios[subj_idxs]
        out_x = self.x[subj_idxs]
        out_y = self.y[subj_idxs]
        # append ratios to x if needed
        if use_ratio:
            out_x = np.append(out_x, np.repeat(subj_upscale_ratios.reshape(-1,1,1), 4, axis=2), axis=1)
        return [out_x, out_y]
    
