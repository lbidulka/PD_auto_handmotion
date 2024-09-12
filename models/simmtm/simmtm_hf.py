import torch.nn as nn
import torch
import math
import numpy as np
import sklearn.metrics
import numpy as np


import utils.dataloader as loader
# import utils.loss
import utils.focal_loss
import utils.features
import utils.data as data_utils

import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from models.simmtm.Configs import Config
configs = Config()
from models.simmtm.loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss
from utils.augmentations import data_transform_masked4cl
import copy 

class TFC(torch.nn.Module):
    def __init__(self, task, datasets, device, length, class_weights=None,):
        super(TFC, self).__init__()

        self.training_mode = 'pre_train'

        self.name = 'sim_mtm_hf'
        self.n_HF_feats = 47 #47
        self.kernel_size = configs.kernel_size
        self.task = task
        self.datasets = datasets
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.combine_34 = True
        self.sample_format = 'scaled_kpt_hf' #'unscaled'   # input data format: 'scaled_kpt', 'unscaled_kpt', 'scaled', 'unscaled'

        self.sequence_len = length
        self.labeler_idx = 1
        self.val_frac = 0.25
        self.transforms = []#loader.noise_rand, loader.scale_rand]#[loader.noise_rand, loader.scale_rand]
        self.transforms_p = []#0.9, 0.9]#[0.9, 0.9]

        self.loss_type = 'Focal'
        self.focal_gamma = 2    #1.5

        self.f_scaler = StandardScaler()    # handcrafted feature normalizer


    def init_model(self, class_cnts):

        self.class_cnts = class_cnts
        # self.conv_block1 = nn.Sequential(
        #     nn.Conv1d(configs.input_channels, 2, kernel_size=configs.kernel_size,
        #               stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
        #     nn.BatchNorm1d(2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     nn.Dropout(configs.dropout)
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv1d(2, 4, kernel_size=8, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(4),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        # )

        # self.conv_block3 = nn.Sequential(
        #     nn.Conv1d(4, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(configs.final_out_channels),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        # )
        
        self.conv_block1 = cnn_block(configs.input_channels, 16, kernel=configs.kernel_size, stride=configs.stride, padding=(configs.kernel_size // 2), 
                                     skips=configs.CNN_skip_connections,)
        self.conv_block2 = cnn_block(16, 32, kernel=8, stride=1, padding=4, 
                                     skips=configs.CNN_skip_connections, )
        self.conv_block3 = cnn_block(32, configs.final_out_channels, kernel=8, stride=1, padding=4, 
                                     skips=configs.CNN_skip_connections, )

        self.encoder = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
        )


        self.dense = nn.Sequential(
            nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128)
        )

        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(configs)
        self.aggregation = AggregationRebuild(configs)
        self.head = nn.Linear(configs.final_out_channels * configs.CNNoutput_channel, self.sequence_len)
        self.loss = torch.nn.MSELoss()

        self.classifier = target_classifier(configs, self.n_HF_feats + 1)
        self.to(self.device)

        alpha = None
        if self.class_cnts is not None:
            alpha = (1/ self.class_cnts) / torch.linalg.norm((1 / self.class_cnts).float(), ord=1)
            alpha = alpha.to(self.device)
        self.classifier_criterion = utils.focal_loss.FocalLoss(gamma=self.focal_gamma, alpha=alpha)

    def init_classifier(self,):
        self.classifier = target_classifier(configs, self.n_HF_feats + 1)
        self.to(self.device)

    def forward(self, data, mode=None):
        # Parse data
        if mode is None:
            x, hf = self.setup_hf_feats(data)
            x_in_t = self.make_dists_from_kpts(x[:, :self.sequence_len])
            x_in_t = torch.tensor(x_in_t).mean(axis=2).float().to(self.device).unsqueeze(1)
            hf = torch.tensor(hf).float().to(self.device)
        else:
            x_in_t, hf, = data
            self.f_scaler.fit(hf.cpu())
            hf = torch.tensor(self.f_scaler.transform(hf.cpu())).to(self.device).float()
        
        # Model forward
        x = self.encoder(x_in_t)
        h = x.reshape(x.shape[0], -1)
        z = self.dense(h)

        if mode == 'pretrain':
            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
            rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)
            pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))

            loss_rb = self.loss(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss = self.awl(loss_cl, loss_rb)
            return loss, loss_cl, loss_rb, pred_x

        elif mode == 'train':
            predictions, _ = self.classifier(h, hf)

            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
            rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)
            pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))

            loss_rb = self.loss(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss = self.awl(loss_cl, loss_rb)

            return loss, loss_cl, loss_rb, pred_x, predictions
        
        elif mode == 'inference':
            predictions, _ = self.classifier(h, hf)
            return predictions
        
        else:
            predictions, embedding = self.classifier(h, hf)
            predictions = torch.argmax(predictions, dim=1)
            return predictions, embedding

    def setup_dataset(self, x, y, hf, subj_ids=None):
        '''
        '''
        # Create dataset
        if not isinstance(x, torch.Tensor):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x

        if not isinstance(hf, torch.Tensor):
            hf_tensor = torch.from_numpy(hf).float()
        else:
            hf_tensor = hf

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
            hf_train = hf_tensor[train_idxs].numpy()
            y_train = y_tensor[train_idxs].numpy()
            x_val = x_tensor[val_idxs].numpy()
            hf_val = hf_tensor[val_idxs].numpy()
            y_val = y_tensor[val_idxs].numpy()

            x_train = np.mean(x_train, axis=2)
            x_train = np.concatenate((x_train, hf_train), axis=1)
            x_val = np.mean(x_val, axis=2)
            x_val = np.concatenate((x_val, hf_val), axis=1)

            x_train, x_val, y_train, y_val, train_ids, val_ids = data_utils.balance_eval_split(x_train, x_val, y_train, y_val, 
                                                                                                subj_ids[train_idxs], subj_ids[val_idxs],
                                                                                                weight_annot_idx=self.labeler_idx,
                                                                                                tol=1.0)
            # x_train, y_train = data_utils.equalize_class_samples(x_train, y_train)


            y_train = y_train[:, self.labeler_idx]
            y_val = y_val[:, self.labeler_idx]

            # finetune with less data
            # use less label for finetune
            train_size = len(y_train)
            data_used_ft = int(train_size * configs.finetune_frac)
            ind_used_ft = range(train_size)[:data_used_ft]
            x_finetune = x_train[ind_used_ft]
            y_finetune = y_train[ind_used_ft]

            x_finetune = torch.from_numpy(x_finetune).float()
            y_finetune = torch.from_numpy(y_finetune).long() if self.task == 'multiclass' else torch.from_numpy(y_finetune).float()

            # datasize = len(y_train)
            # data_used = int(datasize*0.1)
            # x_train = x_train[:data_used]
            # y_train = y_train[:data_used]

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
                                                    seq_len=self.sequence_len)
            
            finetuneset = loader.CustomTensorDataset(tensors=(x_finetune, y_finetune), 
                                                    transforms=self.transforms, 
                                                    transforms_p=self.transforms_p, 
                                                    seq_len=self.sequence_len)
            
            valset = loader.CustomTensorDataset(tensors=(x_val, y_val), seq_len=self.sequence_len)
        else:
            trainset, valset = torch.utils.data.random_split(loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), 
                                                                                        transforms=self.transforms,
                                                                                        transforms_p=self.transforms_p, 
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

        return trainset, valset, finetuneset, sampler

    def model_pretrain(self, model_optimizer, model_scheduler, train_loader, configs, device, epoch):
        total_loss = []
        total_cl_loss = []
        total_rb_loss = []

        # maybe change the train() name in the future.
        self.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            model_optimizer.zero_grad()
            hf = data[:, self.sequence_len:]
            data = data[:, :self.sequence_len]
            data = torch.unsqueeze(data, 1)
            data_masked_m, mask = data_transform_masked4cl(data, configs.masking_ratio, configs.lm, configs.positive_nums)
            data_masked_om = torch.cat([data, data_masked_m], 0)

            data, labels, data_masked_om = data.float().to(device), labels.float().to(device), data_masked_om.float().to(
                device)
            hf = hf.float().to(device)

            # Produce embeddings of original and masked samples
            loss, loss_cl, loss_rb, x_decoded = self.forward([data_masked_om, hf,], mode = 'pretrain')

            loss.backward()
            model_optimizer.step()

            total_loss.append(loss.item())
            total_cl_loss.append(loss_cl.item())
            total_rb_loss.append(loss_rb.item())

        if (epoch % configs.debug_recon_eps_printout) == 0:
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
            plt.title(f'Epoch {epoch} Reconstruction')
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

        labels_all, preds_all = [], []
        for data, labels in train_dl:
            model_optimizer.zero_grad()
            hf = data[:, self.sequence_len:]
            data = data[:, :self.sequence_len]
            data = torch.unsqueeze(data, 1)
            data, labels = data.float().to(device), labels.long().to(device)
            hf = hf.float().to(device)

            # Produce embeddings
            loss_pretrain, loss_cl, loss_rb, x_decoded, predictions = self.forward([data, hf,], mode='train')

            loss = self.classifier_criterion(predictions, labels)

            labels_all.append(labels.cpu().numpy())
            preds_all.append(predictions.detach().argmax(dim=1).cpu().numpy())

            loss.backward()
            model_optimizer.step()
            total_loss.append(loss.item())

        labels_numpy_all = np.concatenate(labels_all)
        pred_numpy_all = np.concatenate(preds_all)
        total_acc = sklearn.metrics.balanced_accuracy_score(labels_numpy_all, pred_numpy_all)
        total_loss = torch.tensor(total_loss).mean()  # average loss

        model_scheduler.step()

        return total_loss, total_acc

    def model_test(self, test_dl, device):
        self.eval()

        # criterion = nn.CrossEntropyLoss()

        total_loss = []
        with torch.no_grad():
            labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
            labels_all, preds_all = [], []
            for data, labels in test_dl:
                hf = data[:, self.sequence_len:]
                data = data[:, :self.sequence_len]
                data = torch.unsqueeze(data, 1)
                data, labels = data.float().to(device), labels.long().to(device)
                hf = hf.float().to(device)
                # Add supervised classifier: 1) it's unique to fine-tuning. 2) this classifier will also be used in test
                predictions = self.forward([data, hf,], mode='inference')

                loss = self.classifier_criterion(predictions, labels)
                # loss = criterion(predictions, labels)

                labels_all.append(labels.cpu().numpy())
                preds_all.append(predictions.detach().argmax(dim=1).cpu().numpy())
                total_loss.append(loss.item())
        
        labels_numpy_all = np.concatenate(labels_all)
        pred_numpy_all = np.concatenate(preds_all)
        total_acc = sklearn.metrics.balanced_accuracy_score(labels_numpy_all, pred_numpy_all)

        total_loss = torch.tensor(total_loss).mean()  # average loss

        return total_loss, total_acc

    def information_gain_feature_selection(self, training_data, training_label, n_selected_feature = 20):
        '''
        information gain feature selection
        '''
        ig = mutual_info_regression(training_data, training_label)
        
        # Create a dictionary of feature importance scores
        feature_scores = {}
        for i in range(len(ig)):
            feature_scores[i] = ig[i]
        # Sort the features by importance score in descending order
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        selected_features_ids = [feat for feat, score in sorted_features[:n_selected_feature]]
        return selected_features_ids

    def setup_hf_feats(self, x_clips, y_clips=None):
        x_poses = x_clips[:, :-2]
        x_feats = x_clips[:, -2:-1]
        x_feats = x_feats.reshape(x_feats.shape[0], -1)
        # cut off where feats == -99
        x_feats = x_feats[:, x_feats[0] != -99]

        x_ratios = x_clips[:, -1:][:,0,0,:1]

        # Get subset of features
        if (y_clips is not None) and (self.n_HF_feats is not None):
            self.best_features_ids = self.information_gain_feature_selection(x_feats, y_clips[:, self.labeler_idx], self.n_HF_feats)
        x_feats = x_feats[:, self.best_features_ids] # select best features
        x_feats = torch.cat([torch.tensor(x_feats), torch.tensor(x_ratios)], dim=1).float()

        if (y_clips is None):
            x_feats = self.f_scaler.transform(x_feats)

        # reshape & pad to match model input
        # hand_dims = x_poses[0,0].flatten().shape[0]
        # pad = torch.ones(x_feats.shape[0], hand_dims - x_feats.shape[1])*-99
        # x_feats = torch.cat([torch.tensor(x_feats), pad], dim=1).reshape(-1, 1, x_poses.shape[2], x_poses.shape[3])
        # x_ratios = x_ratios[:,:, :x_poses.shape[2], :x_poses.shape[3]]
        # x_clips = torch.cat([torch.tensor(x_poses), x_feats, torch.tensor(x_ratios)], dim=1).float()

        if y_clips is not None:
            return x_clips, x_feats, y_clips
        else:
            return x_clips, x_feats


    def make_dists_from_kpts(self, x):
        '''
        Convert keypoints to distances
        '''
        fingertip_kpts = [8, 12, 16, 20]
        palm = x[:, :, :1, :]
        tips = x[:, :, fingertip_kpts, :]
        out_x = np.linalg.norm(tips - palm, axis=-1)
        # out_x = out_x[:, :, :2]
        return out_x        

    def trainer(self, x, y, 
              train_subj_ids=None,
              x_val=None, y_val=None):
        '''
        '''
        x, hf, y = self.setup_hf_feats(x, y)
        x = self.make_dists_from_kpts(x[:, :self.sequence_len])

        if x_val is None:
            trainset, valset, finetuneset, sampler = self.setup_dataset(x, y, hf, train_subj_ids)
        else:
            x_tensor = torch.from_numpy(x).float()
            x_val_tensor = torch.from_numpy(x_val).float()
            y_tensor = torch.from_numpy(y).long()
            y_val_tensor = torch.from_numpy(y_val).long() if self.task == 'multiclass' else torch.from_numpy(y_val).float()
            transforms = [loader.scale_rand, loader.noise_rand,] # loader.amp_decrement]
            transforms_p = [0.5, 0.5,] # 0.1]
            trainset = loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), 
                                                  transforms=transforms, 
                                                  transforms_p=transforms_p, )
            valset = loader.CustomTensorDataset(tensors=(x_val_tensor, y_val_tensor), 
                                                # transforms=transforms, 
                                                # transforms_p=transforms_p, 
                                                )
            sampler = None



        train_loader = torch.utils.data.DataLoader(trainset, batch_size=configs.batch_size, 
                                                  shuffle=True, sampler=sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=configs.batch_size, 
                                                shuffle=False, drop_last=False)

        svr_loader = torch.utils.data.DataLoader(trainset, batch_size=configs.batch_size, 
                                                  shuffle=False, sampler=sampler, drop_last=False)

        finetune_loader = torch.utils.data.DataLoader(finetuneset, batch_size=configs.batch_size, 
                                                  shuffle=True, sampler=sampler, drop_last=True)

    

        for param in self.classifier.logits.parameters():
            param.requires_grad = False
        for param in self.classifier.logits_simple.parameters():
            param.requires_grad = False
        for param in self.classifier.logits_feature.parameters():
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
            pretrain_model = copy.deepcopy(self.state_dict())

            if (epoch % 10) == 0:
                print(f'Pre-training Epoch: {epoch}\t Train Loss: {total_loss:.4f}\t CL Loss: {total_cl_loss:.4f}\t RB Loss: {total_rb_loss:.4f}\n')

            if (epoch % configs.ft_freq) == 0:
                # initialize new classifier, if desired
                if configs.reinit_classifier:
                    self.init_classifier()
                # Enable classifier training
                for param in self.classifier.logits.parameters():
                    param.requires_grad = True
                for param in self.classifier.logits_simple.parameters():
                    param.requires_grad = True
                for param in self.classifier.logits_feature.parameters():
                    param.requires_grad = True
                # Disable encoder training, if desired
                if configs.freeze_encoder:
                    for param in self.encoder.parameters():
                        param.requires_grad = False

                ft_model_optimizer = torch.optim.Adam(self.parameters(), lr=configs.finetune_lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
                ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.finetune_epoch)
                
                for ft_epoch in range(1, configs.finetune_epoch + 1):

                    train_loss, train_acc = self.model_finetune(train_loader, 
                                                                self.device,
                                                                ft_model_optimizer,
                                                                ft_scheduler)
                    
                    # if ft_epoch%10 == 0:
                    val_loss, valid_acc = self.model_test(val_loader,self.device)

                    print(f'Ep {epoch} ft_Ep {ft_epoch} ||  train_loss: {train_loss.item():.3f}, val_loss: {val_loss:.4f} || train_bal_acc: {train_acc.item():.3f}, valid_bal_acc: {valid_acc.item():.3f}')
                    if best_val_metric > val_loss:
                        print(f" -----> (Ep {epoch} ft_Ep {ft_epoch}) New best val_loss = {val_loss:.4f}")
                        best_val_metric = val_loss
                        best_epoch = (epoch, ft_epoch)
                        best_model = copy.deepcopy(self.state_dict())

                # # svr training
                # if best_model is not None:
                #     self.load_state_dict(best_model)
                # self.eval()
                # train_x = []
                # train_y = []
                # for data, labels in svr_loader:
                #     if not train_x:
                #         train_x = data
                #         train_y = labels
                #     else:
                #         train_x = torch.concatenate((train_x, data), dim=0)
                #         train_y = torch.concatenate((train_y, labels), dim=0)
                #     a=1

                for param in self.classifier.logits.parameters():
                    param.requires_grad = False
                for param in self.classifier.logits_simple.parameters():
                    param.requires_grad = False
                for param in self.classifier.logits_feature.parameters():
                    param.requires_grad = False

                params_group = [{'params': self.parameters()}]
                model_optimizer = torch.optim.Adam(params_group, lr=configs.pretrain_lr, betas=(configs.beta1, configs.beta2),
                                                weight_decay=0)
                model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.pretrain_epoch)
            
            # Reset encoder weights to end of last pretrain epoch
            if configs.reinit_encoder:
                self.load_state_dict(pretrain_model)
            # Reenable encoder training, if needed
            if configs.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = True

        # Load best model after warmup
        if best_model is not None:
            self.load_state_dict(best_model)
            print('\nBest epoch:', best_epoch)
            print(f'Best val metric: {best_val_metric:.5f}')



# self.conv_block1 = nn.Sequential(
#             nn.Conv1d(configs.input_channels, 2, kernel_size=configs.kernel_size,
#                       stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
#             nn.BatchNorm1d(2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(configs.dropout)
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(2, 4, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(4),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(4, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )


class cnn_block(nn.Module):
    def __init__(self, input_channels, filters, kernel, stride, padding, skips=False, bias=False):
        super(cnn_block, self).__init__()
        self.skips = skips
        self.conv = nn.Conv1d(input_channels, filters, kernel_size=kernel, stride=stride, bias=bias, padding=padding)
        self.bn = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        if self.skips:
            self.identity = nn.Conv1d(filters, filters, 1)
        
    def forward(self, x):
        if self.skips:
            identity = self.identity(x)
            x = self.conv(x)
            x = self.bn(x)
            x = x + identity
            x = self.relu(x)
            x = self.pool(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
        return x

        


class target_classifier(nn.Module):  # Classification head
    def __init__(self, configs, n_HF_feats):
        super(target_classifier, self).__init__()
        self.size_logits = 16
        self.size_feature = 16

        self.logits = nn.Linear(configs.final_out_channels * configs.CNNoutput_channel, self.size_logits)
        self.logits_feature = nn.Linear(n_HF_feats, self.size_feature)
        self.logits_simple = nn.Linear(self.size_logits + self.size_feature, configs.num_classes_target)
        
        self.bn_logits = nn.BatchNorm1d(8)
        self.bn_feature = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb, feature):
        """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = self.relu(self.logits(emb_flat))
        emb = self.dropout(emb)
        feature_emb = self.logits_feature(feature)
        emb = torch.concatenate((emb, feature_emb), dim=1)

        pred = self.logits_simple(emb)
        return pred, emb

