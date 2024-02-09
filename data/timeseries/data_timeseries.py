import numpy as np


class data_timeseries():
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        self.x, self.y, self.subj_ids, self.handednesses = self.load_dataset_file(dataset_path)
        
    def load_dataset_file(self, file_path):
        data = np.load(file_path)
        x = data['samples']
        y = data['labels']
        subj_ids = data['subj_ids']
        handednesses = data['handednesses']
        return x, y, subj_ids, handednesses
    
    def get_subj_data(self, subj_ids):
        '''
        Get all samples for specified list of subjects
        '''
        subj_idxs = np.where(np.isin(self.subj_ids, subj_ids))[0]

        # get data for these subjects
        subj_data = self.x[subj_idxs]
        subj_labels = self.y[subj_idxs]
        return [subj_data, subj_labels]
    
