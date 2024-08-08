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

from ..base_deepnet import Base_DeepNet
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

from models.simmtm.Configs import Config
configs = Config()
from models.simmtm.loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss
from utils.augmentations import data_transform_masked4cl
import copy 

class TFC(torch.nn.Module):
    def __init__(self, task, datasets, device, length, class_weights=None,):
        super(TFC, self).__init__()

        self.training_mode = 'pre_train'

        self.name = 'sim_mtm'
        self.kernel_size = configs.kernel_size
        self.task = task
        self.datasets = datasets
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.combine_34 = True
        self.sample_format = 'unscaled'   # input data format: 'scaled_kpt', 'unscaled_kpt', 'scaled', 'unscaled'

        self.use_ratio = False
        self.sequence_len = length
        self.labeler_idx = 1
        self.val_frac = 0.2
        self.transforms = []#[loader.noise_rand, loader.scale_rand]
        self.transforms_p = []#[0.9, 0.9]

        self.loss_type = 'Focal'
        self.focal_gamma = 1.5    #1.5


    def init_model(self, class_cnts):

        self.class_cnts = class_cnts
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.dense = nn.Sequential(
            nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(configs)
        self.aggregation = AggregationRebuild(configs)
        self.head = nn.Linear(1280, 178)
        self.mse = torch.nn.MSELoss()

        self.classifier = target_classifier(configs)
        self.to(self.device)

    def init_classifier(self,):
        self.classifier = target_classifier(configs)
        self.to(self.device)

    def forward(self, x_in_t, mode=None):
        x = self.conv_block1(x_in_t)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        h = x.reshape(x.shape[0], -1)
        z = self.dense(h)
        predictions = self.classifier(h)

        if mode == 'train':
            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
            rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)
            pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))

            loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss = self.awl(loss_cl, loss_rb)

            return loss, loss_cl, loss_rb, pred_x, predictions
        elif mode == 'inference':
            return predictions
        else:
            predictions = torch.argmax(predictions, dim=1)

            return predictions 

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
                                                    use_ratio=self.use_ratio,
                                                    seq_len=self.sequence_len)
            valset = loader.CustomTensorDataset(tensors=(x_val, y_val), 
                                                    use_ratio=self.use_ratio,
                                                    seq_len=self.sequence_len)
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

    def model_pretrain(self, model_optimizer, model_scheduler, train_loader, configs, device, epoch):
        total_loss = []
        total_cl_loss = []
        total_rb_loss = []

        # maybe change the train() name in the future.
        self.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            model_optimizer.zero_grad()
            data = torch.unsqueeze(data, 1)
            data_masked_m, mask = data_transform_masked4cl(data, configs.masking_ratio, configs.lm, configs.positive_nums)
            data_masked_om = torch.cat([data, data_masked_m], 0)

            data, labels, data_masked_om = data.float().to(device), labels.float().to(device), data_masked_om.float().to(
                device)

            # Produce embeddings of original and masked samples
            loss, loss_cl, loss_rb, x_decoded, prediction = self.forward(data_masked_om, mode = 'train')

            loss.backward()
            model_optimizer.step()

            total_loss.append(loss.item())
            total_cl_loss.append(loss_cl.item())
            total_rb_loss.append(loss_rb.item())

        if epoch%100==0:
            # debug reconstruction
            _debug_plt_outpath = '_debug_outputs/vae/' + 'reconstruction.png'
            num_samples = 4
            data = data.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            x_decoded = x_decoded.squeeze().cpu().detach().numpy()
            fig, axs = plt.subplots(2, num_samples, figsize=(20, 10))
            for i in range(num_samples):
                axs[0, i].plot(data[i, 0, :])
                axs[1, i].plot(x_decoded[i, :])
                # titles
                axs[0, i].set_title(f'label (y = {labels[i]})')
                axs[1, i].set_title(f'losses (loss = {np.linalg.norm(data[i, 0, :]-x_decoded[i, :])}')
                # y range
                axs[0, i].set_ylim(0, 1)
                axs[1, i].set_ylim(0, 1)
            plt.savefig(_debug_plt_outpath)
            plt.close()

        total_loss = torch.tensor(total_loss).mean()
        total_cl_loss = torch.tensor(total_cl_loss).mean()
        total_rb_loss = torch.tensor(total_rb_loss).mean()

        model_scheduler.step()

        return total_loss, total_cl_loss, total_rb_loss

    def model_finetune(self, train_dl, device, model_optimizer, model_scheduler):
        self.train()

        total_loss = []
        total_acc = []

        alpha = None
        if self.class_cnts is not None:
            alpha = (1/ self.class_cnts) / torch.linalg.norm((1 / self.class_cnts).float(), ord=1)
            alpha = alpha.to(self.device)
        criterion = utils.focal_loss.FocalLoss(gamma=self.focal_gamma, alpha=alpha)

        for data, labels in train_dl:
            model_optimizer.zero_grad()
            data = torch.unsqueeze(data, 1)
            data, labels = data.float().to(device), labels.long().to(device)

            # Produce embeddings
            loss_pretrain, loss_cl, loss_rb, x_decoded, predictions = self.forward(data, mode='train')

            loss = criterion(predictions, labels)
            # loss+=loss_pretrain*0.1
            acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
            total_acc.append(acc_bs)
            total_loss.append(loss.item())

            loss.backward()
            model_optimizer.step()


        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc

        model_scheduler.step()

        return total_loss, total_acc

    def model_test(self, test_dl, device):
        self.eval()

        total_loss = []
        total_acc = []

        criterion = nn.CrossEntropyLoss()


        with torch.no_grad():
            labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
            for data, labels in test_dl:
                data = torch.unsqueeze(data, 1)
                data, labels = data.float().to(device), labels.long().to(device)

                # Add supervised classifier: 1) it's unique to fine-tuning. 2) this classifier will also be used in test
                predictions = self.forward(data, mode='inference')

                loss = criterion(predictions, labels)

                acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()

                total_acc.append(acc_bs)
                total_loss.append(loss.item())

        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc

        return total_loss, total_acc


    def trainer(self, x, y, 
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

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=configs.batch_size, 
                                                  shuffle=True, sampler=sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=configs.batch_size, 
                                                shuffle=False, drop_last=False)

        
        for param in self.classifier.logits.parameters():
            param.requires_grad = False
        for param in self.classifier.logits_simple.parameters():
            param.requires_grad = False

        params_group = [{'params': self.parameters()}]
        model_optimizer = torch.optim.Adam(params_group, lr=configs.pretrain_lr, betas=(configs.beta1, configs.beta2),
                                        weight_decay=0)
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.pretrain_epoch)


        # print("\nMain Training -----------")
        best_val_metric = math.inf

        for epoch in range(1, configs.pretrain_epoch + 1):

            total_loss, total_cl_loss, total_rb_loss = self.model_pretrain(model_optimizer, model_scheduler, train_loader,
                                                                  configs, self.device, epoch)

            if epoch%10 == 0:
                print(f'Pre-training Epoch: {epoch}\t Train Loss: {total_loss:.4f}\t CL Loss: {total_cl_loss:.4f}\t RB Loss: {total_rb_loss:.4f}\n')


            if epoch%10 == 0:
                # print("\nClassifier Training -----------")

                # initialize new classifier
                # self.init_classifier()
                for param in self.classifier.logits.parameters():
                    param.requires_grad = True
                for param in self.classifier.logits_simple.parameters():
                    param.requires_grad = True

                ft_model_optimizer = torch.optim.Adam(self.parameters(), lr=configs.finetune_lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
                ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.finetune_epoch)
                
                for ft_epoch in range(1, configs.finetune_epoch + 1):

                    train_loss, train_acc = self.model_finetune(train_loader, 
                                                                self.device,
                                                                ft_model_optimizer,
                                                                ft_scheduler)
                    
                    # if ft_epoch%10 == 0:
                    val_loss, valid_acc = self.model_test(val_loader,self.device)


                    if best_val_metric > val_loss:
                        print(f'Ep {epoch} ft_Ep {ft_epoch}:  train_loss: {train_loss.item():.3f}, val_loss: {val_loss:.4f} ,train_acc: {train_acc.item():.3f}, valid_acc: {valid_acc.item():.3f}')
                        print(f" -----> (Ep {epoch} ft_Ep {ft_epoch}) New best val_loss = {val_loss:.4f}")
                        best_val_metric = val_loss
                        best_epoch = (epoch, ft_epoch)
                        best_model = copy.deepcopy(self.state_dict())

                for param in self.classifier.logits.parameters():
                    param.requires_grad = True
                for param in self.classifier.logits_simple.parameters():
                    param.requires_grad = True

                params_group = [{'params': self.parameters()}]
                model_optimizer = torch.optim.Adam(params_group, lr=configs.pretrain_lr, betas=(configs.beta1, configs.beta2),
                                                weight_decay=0)
                model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.pretrain_epoch)


        # Load best model after warmup
        if best_model is not None:
            self.load_state_dict(best_model)
            print('\nBest epoch:', best_epoch)
            print(f'Best val metric: {best_val_metric:.5f}')


class target_classifier(nn.Module):  # Classification head
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(1280, 64)
        self.dropout = nn.Dropout(0.5)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        emb = self.dropout(emb)
        pred = self.logits_simple(emb)
        return pred

