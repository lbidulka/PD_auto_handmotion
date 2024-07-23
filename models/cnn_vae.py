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


class CnnVae(torch.nn.Module):
    def __init__(self, task, datasets, device, length, nclasses, transition_channels, class_weights=None,):
        super(CnnVae, self).__init__()

        self.name = 'cnn_vae'
        self.task = task
        self.datasets = datasets
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ratio = False
        self.combine_34 = True
        self.sample_format = 'scaled_hf'   # input data format: 'scaled_kpt', 'unscaled_kpt', 'scaled', 'unscaled'

        self.sequence_len = length
        self.nclasses = nclasses
        self.transition_channels = transition_channels

        self.layer_type = 'mlp'  # 'conv', 'mlp'
        if self.layer_type == 'conv':
            self.latent_size = 50
            self.hidden_dims = [32, 32, 64] #[32, 32, 64]     # (reversed for decoder)
            self.val_frac = 0.3
            self.batch_size = 32 #128
            self.epochs_warmup = 250     # 200
            self.epochs_main = 50       # 50
            self.lr_warmup = 1e-4 #5e-4       # 1e-4
            self.lr_main = 1e-5 #5e-4         # 1e-5
        elif self.layer_type == 'mlp':
            self.latent_size = 32
            self.hidden_dims = [64, 32, 32] #[32, 32, 64]     # (reversed for decoder)
            self.val_frac = 0.2
            self.batch_size = 32 #128
            self.epochs_warmup = 250     # 200
            self.epochs_main = 50       # 50
            self.lr_warmup = 1e-3 #5e-4       # 1e-4
            self.lr_main = 1e-5 #5e-4         # 1e-5

        self.dropout_type = 'none'        # simple_dropout, none,

        self.transforms = [loader.noise_rand, loader.scale_rand]
        self.transforms_p = [0.9, 0.9]
        self.labeler_idx = 1

        self.classifier_type = 'low_mlp'      # 'low_conv', 'low_mlp', 'latent_linear', 'latent_mlp'

    def init_model(self):
        self.encoder = Encoder(1, self.sequence_len, self.nclasses, self.latent_size, 
                               self.transition_channels, self.classifier_type, self.hidden_dims, self.layer_type,
                               self.device)
        self.decoder = Decoder(self.sequence_len, self.nclasses, self.latent_size, 
                               self.transition_channels, self.hidden_dims[::-1], self.layer_type,
                               self.device)
        
        if self.dropout_type == 'simple_dropout':
            dropout_frac = 0.2
            self.dropout = torch.nn.Dropout(dropout_frac)
        elif self.dropout_type == 'none':
            self.dropout = torch.nn.Identity()

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
                                                    use_ratio=self.use_ratio,)
            valset = loader.CustomTensorDataset(tensors=(x_val, y_val), 
                                                    use_ratio=self.use_ratio)
        else:
            trainset, valset = torch.utils.data.random_split(loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), 
                                                                                        transforms=self.transforms,
                                                                                        transforms_p=self.transforms_p, 
                                                                                        use_ratio=self.use_ratio,
                                                                                        seq_len=self.sequence_len), 
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

    def test(self, model, loader, Loss=None):
        acc = []
        if Loss is None:
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
        loss /= len(loader.dataset.tensors[0])
        return acc, loss.item()  

    def train(self, x, y, 
              train_subj_ids=None,
              x_val=None, y_val=None):
        '''
        '''

        x = x[:, :, :2]

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
        # layers = (self.encoder.conv_final, self.encoder.gp_final)
        layers = self.encoder.classifier
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
        best_val_metric = math.inf
        # learning_rates = [0.00001] * 1000 + [0.000001] * 1000 
        # learning_rates = [1e-5] * 1000 + [1e-6] * 1000 
        # learning_rates = [1e-4] * (self.epochs_warmup//2) + [1e-5] * (self.epochs_warmup//2)  #+ [1e-6] * 25
        # learning_rates = [self.lr_warmup] * (self.epochs_warmup//2) + [self.lr_warmup/2] * (self.epochs_warmup//2)  #+ [1e-6] * 25
        learning_rates = [self.lr_warmup] * self.epochs_warmup
        for epoch, lr in enumerate(learning_rates):
            train_loss = 0
            optim_vae.param_groups[0]['lr'] = lr
            for i, (x, y) in enumerate(train_loader):
                x = Variable(x).float().to(self.device)[:, :self.sequence_len]
                y = Variable(y.long()).to(self.device)

                x_masked = self.dropout.forward(x)
                oh_class, mu, logvar, z, x_decoded = self.forward(x_masked, train=True)
                loss = Loss(x_decoded.to(self.device), x, mu.to(self.device), logvar.to(self.device), self.sequence_len) # x_decoded, x, mu, oh_class, y

                optim_vae.zero_grad()
                train_loss += loss
                loss.backward()
                optim_vae.step()

            if epoch%50==0:
                train_loader_len = len(train_loader.dataset.tensors[0])
                # print(f' Train loss: {train_loss.item() / train_loader_len:.3f}')

                val_loss = 0
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        x = Variable(x).float().to(self.device)[:, :self.sequence_len]
                        y = Variable(y.long()).to(self.device)

                        oh_class, mu, logvar, z, x_decoded = self.forward(x, train=True)
                        loss = Loss(x_decoded.to(self.device), x, mu.to(self.device), logvar.to(self.device), self.sequence_len) # x_decoded, x, mu, oh_class, y
                        val_loss += loss
                    val_loader_len = len(val_loader.dataset.tensors[0])
                    # val_loss = val_loss.item() / val_loader_len
                    # print(f' Val loss: {val_loss:.4f}')

                print(f'Ep {epoch}:  train_loss: {train_loss.item():.3f}, val_loss: {val_loss:.4f}')
                
                if best_val_metric > val_loss:
                    print(f" -----> (Ep {epoch}) New best val_loss = {val_loss:.4f}")
                    best_val_metric = val_loss
                    best_epoch = epoch
                    best_model = self.state_dict()

        # Load best model after warmup
        if best_model is not None:
            self.load_state_dict(best_model)
            print('\nBest epoch:', best_epoch)
            print(f'Best val metric: {best_val_metric:.5f}')


        # DEBUG: make some plots of raw, and reconstructed samples (from last batch)
        _debug_plt_outpath = '_debug_outputs/vae/' + 'reconstruction.png'
        num_samples = 4
        x = x.cpu().detach().numpy()
        x_decoded = x_decoded.squeeze().cpu().detach().numpy()
        fig, axs = plt.subplots(2, num_samples, figsize=(20, 10))
        for i in range(num_samples):
            axs[0, i].plot(x[i, :])
            axs[1, i].plot(x_decoded[i, :self.sequence_len])
            # titles
            axs[0, i].set_title(f'Raw (y = {y[i]})')
            axs[1, i].set_title(f'Reconstructed (y = {y[i]})')
            # y range
            axs[0, i].set_ylim(0, 1)
            axs[1, i].set_ylim(0, 1)
        plt.savefig(_debug_plt_outpath)
        # -------------------------------------------------------------------------

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
        best_val_metric = math.inf
        # learning_rates = [0.00001] * 500 
        # learning_rates = [1e-5] * 500 
        learning_rates = [self.lr_main] * self.epochs_main
        # learning_rates = [1e-4] * 50 + [1e-5] * 10
        for epoch, lr in enumerate(learning_rates):
            train_loss = 0

            for i, (x, y) in enumerate(train_loader):
                x = Variable(x).float().to(self.device)
                y = Variable(y.long()).to(self.device)

                x_masked = self.dropout.forward(x)
                oh_class, mu, logvar, z, x_decoded = self.forward(x_masked, train=True)
                loss, class_loss, var_loss = \
                    Loss(x_decoded.to(self.device), x, mu.to(self.device), logvar.to(self.device), oh_class.to(self.device), y, self.sequence_len) # x_decoded, x, mu, oh_class, y
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

            val_acc, val_loss = self.test(self.encoder, val_loader, Loss.classification_loss)
            if epoch%10==0:
                train_loader_len = len(train_loader.dataset.tensors[0])
                train_acc, train_loss = self.test(self.encoder, train_loader, Loss.classification_loss)

                print(f'Ep {epoch}:  train_loss: {train_loss}, train_acc: {train_acc:.2f}, val_acc: {val_acc:.2f}, val_loss: {val_loss:.4f}')

                # print('Epoch:', epoch)
                # print(f' Loss: {loss.data.item() / train_loader_len:.3f}')
                # print(f' Class loss: {class_loss.data.item() / train_loader_len:.5f}')
                # print(f' Recon loss: {var_loss.data.item() / train_loader_len:.5f}')                
                # print(f' Train Acc, loss: {train_acc:.2f}, {train_loss:.5f}')
                # print(f' Validation Acc, Loss: {val_acc:.2f}, {val_loss:.5f}')
                
            # val_loader_len = len(val_loader.dataset.tensors[0])

            # if best_val_metric > val_loss:
            if best_val_metric > val_loss:
                print(f" -----> (Ep {epoch}) New best val_loss = {val_loss:.5f}")
                best_val_metric = val_loss
                best_epoch = epoch
                best_model = self.state_dict()

                # Load best model after training
        if best_model is not None:
            self.load_state_dict(best_model)
            print('\nBest epoch:', best_epoch)
            print(f'Best val metric: {best_val_metric:.5f}')

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
    def __init__(self, in_channels, in_length, nclasses, latent_size, encoder_out_channels, classifier_type, hidden_dims, layer_type,
                 device):
        super(Encoder, self).__init__()
        self.device = device
        self.classifier_type = classifier_type
        self.layer_type = layer_type

        self.in_channels = in_channels
        self.in_length = in_length
        self.nclasses = nclasses
        self.latent_size = latent_size
        self.encoder_out_channels = encoder_out_channels
        length = self.in_length
        self.bn0 = torch.nn.BatchNorm1d(self.in_channels, eps=0.001, momentum=0.99)

        self.hidden_dims = hidden_dims

        # for loop to declare layers
        modules = []
        out_channels = self.in_channels
        if self.layer_type == 'mlp':
            modules.append(torch.nn.Flatten())
            in_channels = self.in_channels * self.in_length
        for i in range(len(hidden_dims)):
            if self.layer_type == 'conv':
                in_channels = out_channels
                out_channels = hidden_dims[i]
                kernel_size = 21
                padding = kernel_size // 2
                modules.append(Conv_block(in_channels, out_channels, kernel_size, padding))
                length = length // 2
            elif self.layer_type == 'mlp':
                block = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_dims[i]),
                    torch.nn.BatchNorm1d(hidden_dims[i]),
                    torch.nn.ReLU()
                )
                in_channels = hidden_dims[i]
                modules.append(block)
        self.enc_blocks = torch.nn.Sequential(*modules)
        
        last_featuremaps_channels = hidden_dims[-1]

        # Classifier
        if self.classifier_type == 'low_conv':
            in_channels = last_featuremaps_channels
            out_channels = nclasses
            kernel_size = 20
            padding = kernel_size // 2
            self.classifier = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                torch.nn.AvgPool1d(length)
            )
            self.conv_final = self.classifier[0]
            self.gp_final = self.classifier[1]
        elif self.classifier_type == 'latent_linear':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.latent_size, nclasses)
            )
        elif self.classifier_type == 'latent_mlp':
            linear_size = 64
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.latent_size, linear_size),
                torch.nn.BatchNorm1d(num_features=linear_size),
                torch.nn.ReLU(),
                torch.nn.Linear(linear_size, nclasses)
            )
        elif self.classifier_type == 'low_mlp':
            linear_size = 64
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dims[-1], linear_size),
                torch.nn.BatchNorm1d(num_features=linear_size),
                torch.nn.ReLU(),
                torch.nn.Linear(linear_size, nclasses)
            )

        # encoder
        in_channels = last_featuremaps_channels
        out_channels = self.encoder_out_channels
        kernel_size = 21
        padding = kernel_size // 2
        self.adapt_pool = torch.nn.AvgPool1d(2); length = length // 2
        self.adapt_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

        if self.layer_type == 'conv':
            self.encode_mean = torch.nn.Linear(length*out_channels, self.latent_size)
            self.encode_logvar = torch.nn.Linear(length*out_channels, self.latent_size)
        elif self.layer_type == 'mlp':
            self.encode_mean = torch.nn.Linear(hidden_dims[-1], self.latent_size)
            self.encode_logvar = torch.nn.Linear(hidden_dims[-1], self.latent_size)

        self.relu = torch.nn.ReLU()
        length = 1

    def forward(self, x):
        if x.shape[1] != self.in_length:
            # separate out the handcraft features
            hf = x[:, self.in_length:]
            x = x[:, :self.in_length]

        x = x.view(-1, self.in_channels, self.in_length)
        x = self.bn0(x)
        x = self.enc_blocks(x)
        post_enc = x

        if self.layer_type == 'conv':
            x = self.adapt_pool(x)
            x = self.adapt_conv(x)
            x = x.view(x.size(0), -1)

        mean = self.relu(self.encode_mean(x)) 
        logvar = self.relu(self.encode_logvar(x))
        z = self._reparameterize(mean, logvar)

        if self.classifier_type in ['low_conv', 'low_mlp']:
            oh_class = self.classifier(post_enc)
        elif self.classifier_type in ['latent_linear', 'latent_mlp']:
            oh_class = self.classifier(z)

        return [oh_class.view(oh_class.size(0), self.nclasses), 
                mean, logvar, 
                z]

    def _reparameterize(self, mean, logvar): # z ~ N(mean, var (sigma^2))   
        # z_std = torch.from_numpy(np.random.normal(0, 1, size=mean.size())).float()
        # sigma = torch.exp(logvar).to(self.device)
        # return mean + sigma * Variable(z_std, requires_grad=False).to(self.device)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

class Decoder(torch.nn.Module):
    def __init__(self, length, nclasses, latent_size, in_channels, hidden_dims, layer_type, device):
        super(Decoder, self).__init__()
        self.device = device
        self.layer_type = layer_type

        self.in_channels = in_channels
        self.sequence_len = length
        self.latent_size = latent_size
        length = self.sequence_len  
        length = length // 2 // 2 // 2 

        self.hidden_dims = hidden_dims

        # Adapt Layer
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.adapt_nn = torch.nn.Linear(latent_size, self.in_channels*length)

        # for loop to declare layers
        modules = []
        out_channels = self.in_channels
        if self.layer_type == 'mlp':
            modules.append(torch.nn.Flatten())
            in_channels = self.in_channels * length
        for i in range(len(hidden_dims)):
            if self.layer_type == 'conv':
                in_channels = out_channels
                out_channels = hidden_dims[i]
                kernel_size = 20
                padding = kernel_size // 2
                modules.append(Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False))
                length = length * 2
            elif self.layer_type == 'mlp':
                block = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_dims[i]),
                    torch.nn.BatchNorm1d(hidden_dims[i]),
                    torch.nn.ReLU()
                )
                in_channels = hidden_dims[i]
                modules.append(block)
        self.dec_blocks = torch.nn.Sequential(*modules)

        in_channels = out_channels
        out_channels = 1
        kernel_size = 20
        padding = kernel_size // 2
        if self.layer_type == 'conv':
            self.dec_final = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        elif self.layer_type == 'mlp':
            self.dec_final = torch.nn.Linear(hidden_dims[-1], self.sequence_len)

    def forward(self, z):

        x = self.relu(self.adapt_nn(z)).to(self.device)
        x = x.view(x.size(0), self.in_channels, self.sequence_len // 2 // 2 // 2)
        x = self.dec_blocks(x)
        x = self.dec_final(x)
        # out = self.tanh(x)
        out = self.sigmoid(x)
        # out = x
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

        recons_loss = torch.nn.functional.mse_loss(x_decoded.squeeze(1)[:,:length], x[:,:length])
        kld_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()

        # return SSD()(x_decoded.squeeze(1)[:,:length], x[:,:length]) + torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1))
        return recons_loss + kld_loss

class VAE_loss(torch.nn.Module):
    def __init__(self, weights, class_loss_type='Focal'):
        super(VAE_loss, self).__init__()
        if class_loss_type == 'CrossEntropy':
            self.classification_loss = torch.nn.CrossEntropyLoss(weights)
        elif class_loss_type == 'Focal':
            self.classification_loss = utils.focal_loss.FocalLoss(gamma=2)
        self.variational_loss = Variational_loss()
        self.c = 0.01

    def forward(self, x_decoded, x, mu, logvar, oh_class, y, length):

        a = self.classification_loss(oh_class, y)
        b = self.variational_loss(
            x_decoded, 
            x, 
            mu, logvar, length)*self.c
        return a + b, a, b

