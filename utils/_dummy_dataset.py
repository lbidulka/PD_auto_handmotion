import numpy as np


class DummyDataset:
    '''
    Simple toy sequence dataset for testing algorithms

    Data consists of sin waves with different frequencies and amplitudes
    '''
    def __init__(self, seq_len=64, num_samples=1000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_classes = 3
        self._generate_data()

        # DEBUG: plot some samples from each class
        import matplotlib.pyplot as plt
        num_plots = 3
        _outplot_path = '_debug_outputs/vae/' + 'dummy_dataset.png'
        fig, axs = plt.subplots(1, num_plots, figsize=(20, 10))
        for c in range(self.num_classes):
            idx = np.where(self.labels == c)[0]
            for i in range(min(num_plots, len(idx))):
                axs[c].plot(self.data[idx[i]].mean(axis=1))
                axs[c].set_title('Class %d' % c)
                axs[c].set_ylim(0, 1)
                axs[c].set_xlim(0, self.seq_len)
        plt.savefig(_outplot_path)
        foo = 5
            

    def _generate_data(self):
        '''
        Generate the sin data, with different frequencies and amplitudes for each class
        '''
        # amplitude and frequency ranges
        self.class_params = {
            0: {'amp': [0.0, 0.0], 'freq': [1 / self.seq_len, 1 / self.seq_len]},
            1: {'amp': [0.25, 0.25], 'freq': [2 / self.seq_len, 2 / self.seq_len]},
            2: {'amp': [0.5, 0.5], 'freq': [4 / self.seq_len, 4 / self.seq_len]}
        }

        # generate data
        self.data = np.zeros((self.num_samples, self.seq_len))
        self.x_unscaled = [[0] for i in range(self.num_samples)]
        self.x_kpts = np.zeros((self.num_samples, self.seq_len, 21, 3))
        self.x_kpts_unscaled = np.zeros((self.num_samples, self.seq_len, 21, 3))
        self.labels = np.zeros((self.num_samples, 2))
        self.subj_ids = np.zeros(self.num_samples)
        self.handednesses = np.zeros(self.num_samples)
        self.upscale_ratios = np.ones(self.num_samples)
        self.kpts_rescale_ratios = np.ones(self.num_samples)
        self.dataset_framerates = np.ones(self.num_samples) * 64
        self.handcraft_feats = np.zeros((self.num_samples, 8))
        for i in range(self.num_samples):
            # random class
            c = np.random.randint(0, self.num_classes)
            self.labels[i, :] = c
            self.subj_ids[i] = i
            # random amplitude and frequency in class range
            amp = np.random.uniform(self.class_params[c]['amp'][0], self.class_params[c]['amp'][1])
            freq = np.random.uniform(self.class_params[c]['freq'][0], self.class_params[c]['freq'][1])
            self.data[i] = amp * np.sin(np.arange(self.seq_len) * freq * 2 * np.pi) + 0.5
        
        # put data at fingertip kpts 
        fingertip_kpts = [8, 12, 16, 20]
        self.x_kpts[:, :, fingertip_kpts, 0] = self.data[..., None]
        # repeat data to make it 4 channel
        self.data = self.data[:,:,None].repeat(4, 2)
        


# foo = DummyDataset()