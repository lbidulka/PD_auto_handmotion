import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import medfilt
import scipy.ndimage.interpolation as inter
import numpy as np

from .base_deepnet import Base_DeepNet
import utils.dataloader as loader
# import utils.loss
import utils.focal_loss
import utils.features
import utils.data as data_utils

from torch.autograd import Variable
import torch
import importlib
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

class Conv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, is_conv=True):
        super(Conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding 
        self.pool_op = torch.nn.AvgPool1d(2, ) if is_conv \
                  else torch.nn.Upsample(scale_factor=2, mode='linear')
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = torch.nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.99)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool_op(x)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, in_length, nclasses, latent_size, encoder_out_channels,
                 device):
        super(Encoder, self).__init__()
        self.device = device

        self.in_channels = in_channels
        self.in_length = in_length
        self.nclasses = nclasses
        self.latent_size = latent_size
        self.encoder_out_channels = encoder_out_channels
        length = self.in_length
        self.bn0 = torch.nn.BatchNorm1d(self.in_channels, eps=0.001, momentum=0.99)
        # Layer 1
        in_channels = self.in_channels
        out_channels = 32
        kernel_size = 21
        padding = kernel_size // 2
        self.conv_block_1 = Conv_block(in_channels, out_channels, kernel_size, padding)
        length = length // 2
        # Layer 2
        in_channels = out_channels
        out_channels = 32
        kernel_size = 21
        padding = kernel_size // 2
        self.conv_block_2 = Conv_block(in_channels, out_channels, kernel_size, padding)
        length = length // 2

        # Layer 3
        in_channels = out_channels
        last_featuremaps_channels = 64
        kernel_size = 21
        padding = kernel_size // 2
        self.conv_block_3 = Conv_block(in_channels, last_featuremaps_channels, kernel_size, padding)
        length = length // 2

        in_channels = last_featuremaps_channels
        out_channels = nclasses
        kernel_size = 20
        padding = kernel_size // 2
        self.conv_final = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.gp_final = torch.nn.AvgPool1d(length)

        # encoder
        in_channels = last_featuremaps_channels
        out_channels = self.encoder_out_channels
        kernel_size = 21
        padding = kernel_size // 2
        self.adapt_pool = torch.nn.AvgPool1d(2); length = length // 2
        self.adapt_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.encode_mean = torch.nn.Linear(length*out_channels, self.latent_size)
        self.encode_logvar = torch.nn.Linear(length*out_channels, self.latent_size)
        self.relu = torch.nn.ReLU()
        length = 1

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.in_length)
        x = self.bn0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        cv_final = self.conv_final(x)
        oh_class = self.gp_final(cv_final)
        x = self.adapt_pool(x)
        x = self.adapt_conv(x)
        x = x.view(x.size(0), -1)
        mean = self.relu(self.encode_mean(x)) 
        logvar = self.relu(self.encode_logvar(x))
        return [oh_class.view(oh_class.size(0), self.nclasses), 
                mean, logvar, 
                self._sample_latent(mean, logvar)]

    def _sample_latent(self, mean, logvar): # z ~ N(mean, var (sigma^2))   
        z_std = torch.from_numpy(np.random.normal(0, 1, size=mean.size())).float()
        sigma = torch.exp(logvar).to(self.device)
        return mean + sigma * Variable(z_std, requires_grad=False).to(self.device)

class Decoder(torch.nn.Module):
    def __init__(self, length, in_channels, nclasses, latent_size, device):
        super(Decoder, self).__init__()
        self.device = device

        self.in_channels = in_channels
        self.length = length
        self.latent_size = latent_size
        length = self.length  
        length = length // 2 // 2 // 2 
        # Adapt Layer
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.adapt_nn = torch.nn.Linear(latent_size, self.in_channels*length)
        # Layer 1
        in_channels = self.in_channels
        out_channels = 64
        kernel_size = 20
        padding = kernel_size // 2
        self.deconv_block_1 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
        length = length * 2
        # Layer 2
        in_channels = out_channels
        out_channels = 32
        kernel_size = 20
        padding = kernel_size // 2
        self.deconv_block_2 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
        length = length * 2

        # Layer 3
        in_channels = out_channels
        out_channels = 32
        kernel_size = 20
        padding = kernel_size // 2
        self.deconv_block_3 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
        length = length * 2

        in_channels = out_channels
        out_channels = 1
        kernel_size = 20
        padding = kernel_size // 2
        self.decode_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, z):

        x = self.relu(self.adapt_nn(z)).to(self.device)
        x = x.view(x.size(0), self.in_channels, self.length // 2 // 2 // 2)
        x = self.deconv_block_1(x)
        x = self.deconv_block_2(x)
        x = self.deconv_block_3(x)
        x = self.decode_conv(x)
        out = self.tanh(x)
        return out

class SSD(torch.nn.Module):
    def __init__(self):
        super(SSD, self).__init__()
    def forward(self, x_decoded, x):
        loss = torch.sum(torch.pow(x - x_decoded, 2))
        return loss / x_decoded.size(0)

class Variational_loss(torch.nn.Module):
    def __init__(self):
        super(Variational_loss, self).__init__()
    def forward(self, x_decoded, x, mu, logvar, length):
        if len(x.shape) > 2:
            x = x.squeeze()
        return SSD()(x_decoded.squeeze()[:,:length], x[:,:length]) + torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1))

class VAE_loss(torch.nn.Module):
    def __init__(self, weights):
        super(VAE_loss, self).__init__()
        self.classification_loss = torch.nn.CrossEntropyLoss(weights)
        self.variational_loss = Variational_loss()
        self.c = 0.01
    def forward(self, x_decoded, x, mu, logvar, oh_class, y, length):

        a = self.classification_loss(oh_class, y)
        b = self.variational_loss(
            x_decoded, 
            x, 
            mu, logvar, length)*self.c
        return a + b, a, b

class CnnVae(torch.nn.Module):
    def __init__(self, task, datasets, device, length, nclasses, latent_size, transition_channels, class_weights=None,):
        super(CnnVae, self).__init__()

        self.name = 'cnn_vae'
        self.task = task
        self.datasets = datasets
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ratio = False
        self.combine_34 = True
        self.sample_format = 'scaled'   # input data format: 'scaled_kpt', 'unscaled_kpt', 'scaled', 'unscaled'

        self.length = length
        self.nclasses = nclasses
        self.latent_size = latent_size
        self.transition_channels = transition_channels

        self.val_frac = 0.20

        self.dropout = torch.nn.Dropout1d(0.2)

        self.transforms = [loader.noise_rand, loader.scale_rand]
        self.transforms_p = [0.9, 0.9]
        self.batch_size = 128
        self.labeler_idx = 1


    def init_model(self):
        self.encoder = Encoder(1, self.length, self.nclasses, self.latent_size, self.transition_channels, self.device)
        self.decoder = Decoder(self.length, self.transition_channels, self.nclasses, self.latent_size, self.device)
        self.to(self.device)

    def forward(self, x, train=False):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if len(x.shape) > 2:
            x = torch.mean(x, dim=2)
        if x.device != self.device:
            x = x.to(self.device)
        oh_class, mu, logvar, z = self.encoder(x)
        x_decoded = self.decoder(z)
        if train:
            return oh_class, mu, logvar, z, x_decoded
        else:
            preds = torch.argmax(oh_class, dim=1)
            return preds

    def setup_dataset(self, x, y, subj_ids=None):
        '''
        '''
        # Create dataset
        if not isinstance(x, torch.Tensor):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.from_numpy(y).long() if self.task == 'multiclass' else torch.from_numpy(y).float()
        else:
            y_tensor = y

        # Split the tensors into Train/val, ensuring that each subj is only in one set
        val_size = int(x_tensor.shape[0] * self.val_frac)
        train_size = x_tensor.shape[0] - val_size

        if subj_ids is not None:
            # Split by subj_ids
            subj_ids_unique = torch.from_numpy(subj_ids).long().unique()
            val_ids = subj_ids_unique[torch.randperm(len(subj_ids_unique))[:int(len(subj_ids_unique) * self.val_frac)]]
            train_ids = torch.tensor([id for id in subj_ids_unique if id not in val_ids])

            train_idxs = torch.tensor([i for i, id in enumerate(subj_ids) if id in train_ids])
            val_idxs = torch.tensor([i for i, id in enumerate(subj_ids) if id in val_ids])

            x_train = x_tensor[train_idxs].numpy()
            y_train = y_tensor[train_idxs].numpy()
            x_val = x_tensor[val_idxs].numpy()
            y_val = y_tensor[val_idxs].numpy()

            x_train, x_val, y_train, y_val, train_ids, val_ids = data_utils.balance_eval_split(x_train, x_val, y_train, y_val, 
                                                                                                subj_ids[train_idxs], subj_ids[val_idxs],
                                                                                                weight_annot_idx=self.labeler_idx,
                                                                                                tol=1.0)
            # x_train, y_train = data_utils.equalize_class_samples(x_train, y_train)
            x_train = np.mean(x_train, axis=2)
            x_val = np.mean(x_val, axis=2)

            y_train = y_train[:, self.labeler_idx]
            y_val = y_val[:, self.labeler_idx]


            x_train = torch.from_numpy(x_train).float()
            x_val = torch.from_numpy(x_val).float()
            y_train = torch.from_numpy(y_train).long() if self.task == 'multiclass' else torch.from_numpy(y_train).float()
            y_val = torch.from_numpy(y_val).long() if self.task == 'multiclass' else torch.from_numpy(y_val).float()
            train_ids = torch.from_numpy(train_ids).unique()
            val_ids = torch.from_numpy(val_ids).unique()

            # # setup loss w/ class weights if needed
            # if self.loss_type == 'Focal':
            #     class_weights_idx = 1   # TEMP: have to choose label to use for counting
            #     class_weights = torch.bincount(y_train[:, class_weights_idx].flatten())
            #     self.class_weights = 1 / class_weights
            #     focal_alpha = self.class_weights * (self.clc / torch.linalg.norm(self.class_weights, ord=1))   # norm 
            #     self._build_criterion(focal_alpha.to(self.device))

            trainset = loader.CustomTensorDataset(tensors=(x_train, y_train), 
                                                    transforms=self.transforms, 
                                                    transforms_p=self.transforms_p, 
                                                    use_ratio=self.use_ratio)
            valset = loader.CustomTensorDataset(tensors=(x_val, y_val), 
                                                    use_ratio=self.use_ratio)
        else:
            trainset, valset = torch.utils.data.random_split(loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), 
                                                                                        transforms=self.transforms,
                                                                                        transforms_p=self.transforms_p, 
                                                                                        use_ratio=self.use_ratio), 
                                                             [train_size, val_size])

        # Setup weighted random sample for trainset
        # if self.task == 'binclass':
        #     class_sample_count = torch.tensor(
        #         [(y_tensor == 0).sum(), (y_tensor == 1).sum()])
        #     ratio = class_sample_count[1] / class_sample_count[0]
        #     self.class0_reweight = ratio
        #     samples_weight = weight[y_tensor.long()]
        #     sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        #     self.shuffle = False
        # else:
        #     sampler = None
        sampler = None

        return trainset, valset, sampler

    def test(self, model, loader):
        acc = []
        Loss = torch.nn.CrossEntropyLoss()
        loss = 0

        with torch.no_grad():
            for batch_id, (x, y) in enumerate(loader):
                x = Variable(x).float().to(self.device)
                y = Variable(y).to(self.device)
                out = model(x)
                y_pred = out[0]
                loss += Loss(y_pred, y)
                _, index = torch.max(y_pred, -1)
                acc.append((index == y).cpu().data.numpy())
            acc = np.concatenate(acc).mean()
        return acc, loss        

    def train(self, x, y, 
              train_subj_ids=None,
              x_val=None, y_val=None):
        '''
        '''

        if x_val is None:
            trainset, valset, sampler = self.setup_dataset(x, y, train_subj_ids)
        else:
            x_tensor = torch.from_numpy(x).float()
            x_val_tensor = torch.from_numpy(x_val).float()
            y_tensor = torch.from_numpy(y).long()
            y_val_tensor = torch.from_numpy(y_val).long() if self.task == 'multiclass' else torch.from_numpy(y_val).float()
            transforms = [loader.scale_rand, loader.noise_rand,] # loader.amp_decrement]
            transforms_p = [0.5, 0.5,] # 0.1]
            trainset = loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), 
                                                  transforms=transforms, 
                                                  transforms_p=transforms_p, 
                                                  use_ratio=self.use_ratio)
            valset = loader.CustomTensorDataset(tensors=(x_val_tensor, y_val_tensor), 
                                                # transforms=transforms, 
                                                # transforms_p=transforms_p, 
                                                use_ratio=self.use_ratio)
            sampler = None

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, 
                                                  shuffle=True, sampler=sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, 
                                                shuffle=False, drop_last=False)
        weights = np.zeros(4,)
        for i in range(4):
            weights[i] = len(np.where(y[:,1] == i)[0])
        normalized_weights = weights / np.max(weights)
        class_weight = {i : normalized_weights[i] for i in range(len(normalized_weights))}
        weights = torch.from_numpy(np.array(list(class_weight.values()))).float().to(self.device)

        parameters = []
        layers = (self.encoder.conv_final, self.encoder.gp_final)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.encoder.parameters():
            if param.requires_grad == True:
                parameters.append(param)
        for param in self.decoder.parameters():
            if param.requires_grad == True:
                parameters.append(param)

        self.vae_parameters = iter(parameters)
        optim_vae = torch.optim.Adam(self.vae_parameters)
        Loss = Variational_loss()

        # warm up
        print("\nWarm up -----------")
        # learning_rates = [0.00001] * 1000 + [0.000001] * 1000 
        # learning_rates = [1e-5] * 1000 + [1e-6] * 1000 
        learning_rates = [1e-4] * 100 + [1e-5] * 100  #+ [1e-6] * 25
        for epoch, lr in enumerate(learning_rates):
            train_loss = 0
            optim_vae.param_groups[0]['lr'] = lr
            for i, (x, y) in enumerate(train_loader):
                x = Variable(x).float().to(self.device)
                y = Variable(y.long()).to(self.device)

                x = self.dropout.forward(x)
                oh_class, mu, logvar, z, x_decoded = self.forward(x, train=True)
                loss = Loss(x_decoded.to(self.device), x, mu.to(self.device), logvar.to(self.device), self.length) # x_decoded, x, mu, oh_class, y

                optim_vae.zero_grad()
                train_loss += loss
                loss.backward()
                optim_vae.step()

            if epoch%100==0:
                print('Epoch: ', epoch)
                train_loader_len = len(train_loader.dataset.tensors[0])
                print('Train loss: ', train_loss.item() / train_loader_len)

                val_loss = 0
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        x = Variable(x).float().to(self.device)
                        y = Variable(y.long()).to(self.device)

                        oh_class, mu, logvar, z, x_decoded = self.forward(x, train=True)
                        loss = Loss(x_decoded.to(self.device), x, mu.to(self.device), logvar.to(self.device), self.length) # x_decoded, x, mu, oh_class, y
                        val_loss +=loss
                    val_loader_len = len(val_loader.dataset.tensors[0])
                    print('Val loss:', val_loss.item() / val_loader_len)

        parameters = []
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.encoder.parameters():
            if param.requires_grad == True:
                parameters.append(param)
        for param in self.decoder.parameters():
            if param.requires_grad == True:
                parameters.append(param)
        self.all_parameters = iter(parameters)

        optim_all = torch.optim.Adam(self.all_parameters)
        Loss = VAE_loss(weights)

        print("\nMain Training -----------")
        best_val_metric = -math.inf
        # learning_rates = [0.00001] * 500 
        # learning_rates = [1e-5] * 500 
        learning_rates = [1e-5] * 50
        # learning_rates = [1e-4] * 50 + [1e-5] * 10
        for epoch, lr in enumerate(learning_rates):
            train_loss = 0

            for i, (x, y) in enumerate(train_loader):
                x = Variable(x).float().to(self.device)
                y = Variable(y.long()).to(self.device)

                x = self.dropout.forward(x)
                oh_class, mu, logvar, z, x_decoded = self.forward(x, train=True)
                loss, class_loss, var_loss = \
                    Loss(x_decoded.to(self.device), x, mu.to(self.device), logvar.to(self.device), oh_class.to(self.device), y, self.length) # x_decoded, x, mu, oh_class, y
                optim_all.zero_grad()
                train_loss += loss.item()
                loss.backward()
                optim_all.step()
                # if not i % 50:
                #     print('training encoder only\n')
                #     oh_class, _, _ = self.encoder(x)
                #     enc_aux_loss = torch.nn.CrossEntropyLoss()(oh_class.to(self.device), y)
                #     # enc_aux_loss = torch.nn.CrossEntropyLoss(weights)(oh_class.to(self.device), y)
                #     optim.zero_grad()
                #     enc_aux_loss.backward()
                #     optim.step()


            if epoch%10==0:
                train_loader_len = len(train_loader.dataset.tensors[0])
                print('Epoch:', epoch)
                print('Loss:', loss.data.item() / train_loader_len)
                print('Class loss:', class_loss.data.item() / train_loader_len)
                print('Recon loss:', var_loss.data.item() / train_loader_len)
                print('Train Accuracy: ', self.test(self.encoder, train_loader))
                
            val_acc, val_loss = self.test(self.encoder, val_loader)
            if epoch%10==0:
                print('Validation Acc, Loss:', val_acc, val_loss)
            val_loader_len = len(val_loader.dataset.tensors[0])
            val_loss = val_loss.item() / val_loader_len

            if best_val_metric < val_loss:
                print(" -----> New best val model")
                best_val_metric = val_loss
                best_epoch = epoch
                best_model = self.state_dict()

                # Load best model after training
        if best_model is not None:
            self.load_state_dict(best_model)
            print('Best epoch:', best_epoch)
            print('Best val metric:', best_val_metric)