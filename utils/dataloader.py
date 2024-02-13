import torch

class CustomTensorDataset(torch.utils.data.Dataset):
    '''TensorDataset with support of transforms.
    '''
    def __init__(self, tensors, transforms=None, use_ratio=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms
        self.use_ratio = use_ratio

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transforms:
            for transform in self.transforms:
                # x = transform(x)
                # dont apply transform to ratio (final entry)
                if self.use_ratio:
                    x[:-1] = transform(x[:-1])
                else:
                    x = transform(x)
        y = self.tensors[1][index]
        return x, y
    
    def __len__(self):
        return self.tensors[0].size(0)
    
# Simple transforms
def scale_rand(x):
    return x * torch.empty(1).uniform_(0.9, 1.1)

def noise_rand(x):
    return x + torch.empty(x.shape).normal_(0, 0.1)