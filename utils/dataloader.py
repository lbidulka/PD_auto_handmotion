import torch
import numpy as np

import utils.data as data_utils

class CustomTensorDataset(torch.utils.data.Dataset):
    '''TensorDataset with support of transforms.
    '''
    def __init__(self, tensors, transforms=None, transforms_p=None, use_ratio=False,
                 seq_len=-1):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms
        self.transforms_p = transforms_p
        self.use_ratio = use_ratio
        self.seq_len = seq_len

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transforms:
            for transform, transform_p in zip(self.transforms, self.transforms_p):
                if torch.rand(1) < transform_p:
                    # dont apply transform to ratio (final entry)
                    if self.use_ratio:
                        x[:self.seq_len], y = transform(x[:self.seq_len], y)
                    else:
                        x, y = transform(x, y)
        return x, y
    
    def __len__(self):
        return self.tensors[0].size(0)
    
# Simple transforms
def scale_rand(x, y):
    return x * torch.empty(1).uniform_(0.9, 1.1).to(x.device), y

def scale_rand_np(x, y):
    return x * np.random.uniform(0.95, 1.0), y

def noise_rand(x, y):
    return x + torch.empty(x.shape).normal_(0, 0.02).to(x.device), y

def noise_rand_np(x, y):
    return x + np.random.normal(0, 0.02, x.shape), y

def trim_rand(x, y, trim=[10, 10]):
    '''
    Randomly trim the input signal (DOES NOT RESCALE TO ORIGINAL LENGTH)
    '''
    trim_start = np.random.randint(1, trim[0])
    trim_end = np.random.randint(1, trim[1])
    x_rescale, _ = data_utils.scale_to_uniform_len([x[trim_start:-trim_end].cpu().numpy()], x.shape[0], False)
    x_rescale = torch.tensor(x_rescale[0]).to(x.device)
    return x_rescale, y

# UPDRS transforms
def amp_decrement(x, y):
    '''
    Apply an amplitude decrement to the input signal, only for y = 0 or 1
    Amplitude decrement is a linear decrease in amplitude to a final 1-decrement 
    '''
    decrement = 0.2
    if (y == 0).any():
        # Add amplitude decrement beginning at 3/4 of the signal
        dec_array = torch.ones_like(x)
        dec_array[int(x.shape[0]*3/4):] = torch.linspace(1, 1-decrement, int(x.shape[0]/4)).reshape(-1,1).repeat(1,4)
        y[y==0] = 1
        return x * dec_array, y
    # elif (y == 1).any():
    #     # Add amplitude decrement beginning at 1/2 of the signal
    #     dec_array = torch.ones_like(x)
    #     dec_array[int(x.shape[0]/2):] = torch.linspace(1, 1-decrement, int(x.shape[0]/2)).reshape(-1,1).repeat(1,4)
    #     y[y==1] = 2
    #     return x * dec_array, y
    else:
        return x, y