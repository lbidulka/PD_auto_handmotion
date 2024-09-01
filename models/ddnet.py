import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from .base_deepnet import Base_DeepNet
import utils.dataloader as loader
import utils.focal_loss
import utils.features


class OF_loss():
        def __init__(self, gamma, alpha, num_classes, beta=0.2):
            '''
            beta:    weighting factor for ordinal loss
            '''
            self.alpha = alpha
            self.beta = beta    # weighting factor for ordinal loss
            self.gamma = gamma
            self.num_classes = num_classes

            # self.focal = utils.focal_loss.FocalLoss(gamma=self.gamma, alpha=self.alpha, reduction='none')
            self.CE = nn.CrossEntropyLoss(reduction='none')
            self.nll_loss = nn.NLLLoss(weight=self.alpha, reduction='none')

        def focal_term(self, x, y):
            '''
            
            '''
            # compute weighted cross entropy term: -alpha * log(pt)
            # (alpha is already part of self.nll_loss)
            log_p = F.log_softmax(x, dim=-1)

            # get true class column from each row
            all_rows = torch.arange(len(x))
            log_pt = log_p[all_rows, y]

            # compute focal term: (1 - pt)^gamma
            pt = log_pt.exp()
            focal_term = (1 - pt)**self.gamma
            return focal_term

        def ordinal_term(self, logits, labels):
            '''
            Ordinal loss, to penalize predictions that are too far from the true label
            '''
            if self.num_classes == 4:
                ordinal_matrix = torch.tensor([[0, 4, 6, 8,], 
                                          [4, 0, 4, 6,],
                                          [6, 4, 0, 4,],
                                          [8, 6, 4, 0,],]).float()
            else:
                ordinal_matrix = torch.tensor([[0, 4, 6, 8, 10], 
                                          [4, 0, 4, 6, 8],
                                          [6, 4, 0, 4, 6],
                                          [8, 6, 4, 0, 4],
                                          [10, 8, 6, 4, 0],]).float()

            # pred_labels = torch.argmax(logits, dim=1)
            # ordinal_dist = torch.abs(pred_labels - labels)
            # ordinal_loss = ((1 + ordinal_dist) / self.num_classes) * self.focal(logits, labels) #self.CE(logits, labels)
            # ordinal_loss = ()
            # ordinal_loss = ordinal_loss.mean()

            ordinal_term = (torch.softmax(logits, 1) * ordinal_matrix.to(labels.device)[labels]).sum(1)


            return ordinal_term

        def __call__(self, logits, labels):
            # focal = self.focal(logits, labels).mean()
            # ordinal = self.ordinal_loss(logits, labels)
            # loss =  (1 - self.beta) * focal + self.beta * ordinal

            focal_term = self.focal_term(logits, labels)
            ordinal_term = self.ordinal_term(logits, labels)
            ce = self.CE(logits, labels)

            loss = ((1 - self.beta) * focal_term + self.beta * ordinal_term) * ce
            loss = loss.mean()
            return loss

class DDNet(Base_DeepNet):
    def __init__(self, task, datasets, UPDRS_task, device,
                 class_weights=None,):
        super().__init__()
        self.name = 'ddnet'
        self.datasets = datasets
        self.class_weights = class_weights
        self.UPDRS_task = UPDRS_task
        
        # Model params
        self.m_branch = True  # Use motion branch?
        # Feature branch options
        self.f_branch = True  # Use handcrafted feature branch?
        self.f_branch_n_HF_feats = 20  # number of handcrafted features to select with info gain
        self.f_branch_type = 'lin_embed'  # lin_embed, cascade_mlp
        self.f_branch_preconcat_layers = 0  # number of layers before concat with handcraft feats
        self.f_branch_warmup_eps = 0   # no. to train kpt branches before enabling handcraft branch

        self.skip_connections = True

        # self.best_features_ids = [0, 1, 2, 4, 6, 10, 15, 21, 24, 25, 26, 27, 34, 35, ]#38] # 38(root coord std())
        self.best_features_ids = [i for i in range(self.f_branch_n_HF_feats)] # 24

        # Task
        self.task = task
        self.labeler_idx = 1

        # Data params
        if self.f_branch:
            self.sample_format = 'scaled_kpt_hf'
        else:
            self.sample_format = 'scaled_kpt'   # input data format: 'scaled_kpt', 'unscaled_kpt', 'scaled', 'unscaled'
        self.combine_34 = True              # combine classes 3/4 to label 3?
        self.shuffle = True
        self.drop_last = False
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.print_loss = True
        self.print_epochs = 1

        self.flip_L_to_R = True  # swap all L hands to R hands if True
        self.equalize_class_samples = False

        self.full_seq = True # use full sequence?
        # self.input_kpts = [0, 8, 12, 16, 20] 5   # wrist and all finger tips
        self.input_kpts = [i for i in range(21)]    # all kpts
        # self.input_kpts = [i for i in range(21) if i not in [1,2,3,4]]    # all kpts excluding thumb

        if task == 'binclass':
            raise NotImplementedError
        elif task == 'multiclass':
            self.joint_n = len(self.input_kpts)   # the number of joints
            self.joint_d = 2    # the dimension of joints
            self.clc = 4 if self.combine_34 else 5         # num classes
            if self.joint_n == 21:
                self.feat_d = 210
            elif self.joint_n == 17:
                self.feat_d = 136
            else:
                self.feat_d = 10 #210   # flat JCD len
            
            self.filters = 16
            self.num_linear = 2 # number of linear layers after conv blocks
            self.linear_size = 32 # size of linear layers after conv blocks/fusion
            
            if self.sample_format in ['scaled_kpt', 'scaled_kpt_hf']: self.use_ratio = True

            if self.full_seq:
                self.frame_l = 256 #256 # max len for CAMERA = 1467
            else:
                self.frame_l = 80   # network input length     80

            # Training params
            self.selection_metric = 'loss'    # f1, loss
            self.val_frac = 0.25
            if self.datasets == 'PD4T':
                self.batch_size = 64
                self.num_epochs = 50
                self.lr = 1e-3
                self.focal_gamma = 1.5
            elif self.datasets == 'CAMERA':
                self.batch_size = 32
                self.num_epochs = 75 #75
                self.lr = 5e-4
                self.focal_gamma = 2
            elif self.datasets == 'CAMERA,PD4T' or self.datasets == 'PD4T,CAMERA':
                if self.UPDRS_task == 'hand_movement':

                    self.batch_size = 128
                    self.focal_gamma = 2    #1.5
                    self.num_epochs = 75 #100
                    
                    if (self.f_branch_type == 'lin_embed'):
                        if self.skip_connections:
                            self.lr = 1e-3
                        else:
                            self.lr = 5e-4
                    else:
                        self.lr = 1e-3

                    
                elif self.UPDRS_task == 'finger_tapping':
                    self.batch_size = 128
                    self.num_epochs = 50
                    self.lr = 1e-3
                    self.focal_gamma = 2
            else:
                if self.full_seq:
                    self.batch_size = 64
                    self.num_epochs = 50
                    self.lr = 5e-4
                else:
                    self.batch_size = 64
                    self.num_epochs = 1 #50
                    self.lr = 5e-4
                self.focal_gamma = 2
            # self.loss_type = 'CrossEntropy'
            self.loss_type = 'Focal' #'Focal', 'OF'

            self.scheduler_type = None #'cosine' #'cosine', None, 
            self.scheduler_lr_min = 1e-5
            self.scheduler_T_max = int(self.num_epochs*2)
            self.print_lr = False

            # Data augmentations
            self.transforms = [loader.noise_rand, loader.scale_rand]#loader.Hflip]# loader.scale_rand,]# loader.trim_rand]
            self.transforms_p = [0.9, 0.9]# 0.5]# 0.9,]# 0.0]
            if not self.flip_L_to_R:
                self.transforms.append(loader.Hflip)
                self.transforms_p.append(0.5)

            self.f_scaler = StandardScaler()    # handcrafted feature normalizer
        
        self._build_model()

    def _build_model(self, *args, **kwargs):
        self.model = DDNet_Original(self.frame_l, self.joint_n, self.joint_d, self.feat_d, self.filters, self.clc, 
                                    self.sample_format, self.input_kpts, self.m_branch, 
                                    self.f_branch, self.f_branch_type, self.f_branch_preconcat_layers, len(self.best_features_ids),
                                    self.num_linear, self.linear_size, self.skip_connections)
        self.model.to(self.device)
        self._build_criterion(*args, **kwargs)
        if self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_T_max, eta_min=self.scheduler_lr_min)
        
    def _build_criterion(self, *args, **kwargs):
        class_cnts = kwargs['class_cnts'] if len(kwargs) > 0 else None
        if self.task == 'multiclass': 
            if self.loss_type == 'CrossEntropy':
                self.criterion = torch.nn.CrossEntropyLoss()
            elif self.loss_type == 'Focal':
                alpha = None
                if class_cnts is not None:
                    alpha = (1/ class_cnts) / torch.linalg.norm((1 / class_cnts).float(), ord=1)
                    alpha = alpha.to(self.device)
                self.criterion = utils.focal_loss.FocalLoss(gamma=self.focal_gamma, alpha=alpha)
            elif self.loss_type == 'OF':
                alpha = None
                if class_cnts is not None:
                    alpha = (1/ class_cnts) / torch.linalg.norm((1 / class_cnts).float(), ord=1)
                    alpha = alpha.to(self.device)
                self.criterion = OF_loss(gamma=self.focal_gamma, alpha=alpha, num_classes=self.clc)
        elif self.task == 'binclass':
            self.criterion = torch.nn.BCELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def eval(self,):
        self.model.eval()

    def __call__(self, x, reduce='mean'):
        '''
        fwd pass
        
        reduce: method to combine clip predictions. 'mean', 'max', 'min' else return all
        '''
        if self.sample_format == 'unscaled_kpt':
            # split into as many clips as possible
            vid_clips = self.make_clips(x, odd_clip_min_ratio=0.2)
        else:
            vid_clips = torch.tensor(x, dtype=torch.float32)

        # Get handcraft feats and insert into input, if required
        if self.model.f_branch:
            x_clips = vid_clips
            x_poses = x_clips[:, :-2]
            x_feats = x_clips[:, -2:-1]
            x_feats = x_feats.reshape(x_feats.shape[0], -1)
            # cut off where feats == -99
            x_feats = x_feats[:, x_feats[0] != -99]

            x_ratios = x_clips[:, -1:].float()
            x_poses = self.norm_input(x_poses)
            x_feats = self.f_scaler.transform(x_feats)
            x_feats = x_feats[:, self.best_features_ids] # select best features

            # reshape & pad to match model input
            hand_dims = x_poses[0,0].flatten().shape[0]
            pad = torch.ones(x_feats.shape[0], hand_dims - x_feats.shape[1])*-99
            x_feats = torch.cat([torch.tensor(x_feats), pad], dim=1).reshape(-1, 1, x_poses.shape[2], x_poses.shape[3])
            x_ratios = x_ratios[:,:, :x_poses.shape[2], :x_poses.shape[3]]
            x_clips = torch.cat([x_poses, x_feats, x_ratios], dim=1).float()
        else:
            x_clips = self.norm_input(vid_clips)

        if self.full_seq:
            x_clips = x_clips.unsqueeze(1)

        # Get avg pred over all clips in each input sample
        self.model.eval()
        preds_all_clips = []
        for clips in x_clips:
            preds_clip = []
            for clip in clips:
                clip = clip.to(self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(clip)

                if self.task == 'multiclass':
                    preds = torch.argmax(logits, dim=1)
                elif self.task == 'binclass':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                preds_clip.append(preds)

            # TEMP: handle too-short sequence (no clips)
            if len(preds_clip) == 0:
                preds_clip = [torch.zeros(1, dtype=torch.long).to(self.device)]
            preds_all_clips.append(torch.stack(preds_clip).float())
        if reduce == 'mean':
            return torch.hstack([pred.mean().round() for pred in preds_all_clips])
        elif reduce == 'max':
            return torch.hstack([pred.max() for pred in preds_all_clips])
        elif reduce == 'min':
            return torch.hstack([pred.min() for pred in preds_all_clips])
        else:
            return preds_all_clips

    def norm_input(self, x):
        '''
        Normalize input clips: [B, # frames, J, D]
        '''
        # only keep desired dims
        x = x[..., :self.joint_d]
        for i, clip in enumerate(x):
            if self.use_ratio:
                clip = clip[:-1]

            # swap all L hands to R hands if needed
            if self.flip_L_to_R:
                # check cross prod of index-root and pinky-root
                # cross = torch.cross(clip[:,5] - clip[:,0], clip[:,17] - clip[:,0], dim=1).mean(0)
                # check thumb pos w.r.t. root
                thumb_x = ((clip[:,4,0] + clip[:,3,0] + clip[:,2,0] + clip[:,1,0]) / 4).mean(0)
                root_x = clip[:,0,0].mean(0)
                if thumb_x < root_x:
                    clip[:,:,0] *= -1
                    if self.use_ratio:
                        x[i,:-1,:,0] *= -1
                    else:
                        x[i,:,:,0] *= -1
           
            # normalize size & mean
            # palm size
            wrist = clip[:,0]
            palm_vector = ((clip[:,5] - wrist) + (clip[:,9] - wrist) + (clip[:,13] - wrist) + (clip[:,17] - wrist)) / 4
            palm_size = np.linalg.norm(palm_vector, axis=1).reshape(-1,1,1)
            # for all 0 frames, set palm size to 1
            palm_size[palm_size == 0] = 1
            # mean root joint
            mean_root = clip[:,:1].mean((0,1), keepdims=True)

            if self.use_ratio:
                x[i,:-1] -= mean_root
                x[i,:-1] /= palm_size
            else:
                x[i] -= mean_root
                x[i] /= palm_size

        return x

    def make_clips(self, x,
                   odd_clip_min_ratio=None):
        '''
        '''
        # split into as many clips as possible, trimming max index to that provided in the sample
        vid_raw_clips = [
            torch.split(torch.tensor(vid[:int(vid[-1,0,0])], dtype=torch.float32), self.frame_l, dim=0) for vid in x
        ]
        
        if odd_clip_min_ratio is None:
            if self.full_seq:
                odd_clip_min_ratio = 0.25
            else:
                odd_clip_min_ratio = 0.5
        
        vid_clips = []
        for clips in vid_raw_clips:
            _clips = []
            for clip in clips:
                if clip.shape[0] == self.frame_l:
                    _clip = clip
                # if last clip is shorter than the seq len, but not too short (or if only one clip), pad with 0s to equal length and include
                elif (clip.shape[0] < self.frame_l) and (odd_clip_min_ratio <= (clip.shape[0] / self.frame_l)) or (len(clips) == 1):
                    pad = torch.zeros((self.frame_l - clip.shape[0], self.joint_n, self.joint_d))
                    _clip = torch.cat([clip, pad], dim=0)
                _clips.append(_clip)
            vid_clips.append(_clips)
        
        return vid_clips

    def make_trainval_sets(self, x_train, y_train, x_val, y_val):
        '''
        Override to specify frame length for handcraft feature fusion
        '''
        trainset = loader.CustomTensorDataset(tensors=(x_train, y_train), 
                                                    transforms=self.transforms, 
                                                    transforms_p=self.transforms_p, 
                                                    use_ratio=self.use_ratio,
                                                    seq_len=self.frame_l)
        valset = loader.CustomTensorDataset(tensors=(x_val, y_val), 
                                                # transforms=self.transforms, 
                                                # transforms_p=self.transforms_p, 
                                                use_ratio=self.use_ratio,
                                                seq_len=self.frame_l)
        return trainset, valset

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

    def trainer(self, x, y, train_subj_ids=None,):
        self.train(x, y, train_subj_ids=train_subj_ids)

    def train(self, x, y, 
              train_subj_ids=None,
              x_val=None, y_val=None):
        '''
        '''
        # make clips
        if self.sample_format == 'unscaled_kpt':
            vid_clips = self.make_clips(x)
            vid_clips_labels = [
                [label for clip in clips] for clips, label in zip(vid_clips, y)
            ]
            vid_cliips_ids = [
                [subj_id for clip in clips] for clips, subj_id in zip(vid_clips, train_subj_ids)
            ]
            # remove empty clips (sequence too short)
            vid_clips_labels = [labels for labels, clips in zip(vid_clips_labels, vid_clips) if len(clips) > 0]
            vid_cliips_ids = [ids for ids, clips in zip(vid_cliips_ids, vid_clips) if len(clips) > 0]
            vid_clips = [clips for clips in vid_clips if len(clips) > 0]

            # combine into new x and y
            x_clips = torch.cat([torch.stack(clips) for clips in vid_clips], dim=0).numpy()
            y_clips = np.concatenate([np.stack(labels) for labels in vid_clips_labels], axis=0)
            ids_clips = np.concatenate([np.stack(ids) for ids in vid_cliips_ids], axis=0)
        else:
            x_clips = x
            y_clips = y
            ids_clips = train_subj_ids

        # Get handcrafted features
        if self.model.f_branch:
            x_poses = x_clips[:, :-2]
            x_feats = x_clips[:, -2:-1]
            x_feats = x_feats.reshape(x_feats.shape[0], -1)
            # cut off where feats == -99
            x_feats = x_feats[:, x_feats[0] != -99]

            x_ratios = torch.from_numpy(x_clips[:, -1:]).float()
            x_poses = self.norm_input(torch.from_numpy(x_poses).float())
            self.f_scaler.fit(x_feats)
            x_feats = self.f_scaler.transform(x_feats)

            # Get subset of features
            if self.f_branch_n_HF_feats is not None:
                self.best_features_ids = self.information_gain_feature_selection(x_feats, y_clips[:, self.labeler_idx], self.f_branch_n_HF_feats)
            # self.best_features_ids = [0, 1, 2, 4, 6, 10, 15, 21, 24, 25, 26, 27, 34, 35, 38] #(root coord std())
            x_feats = x_feats[:, self.best_features_ids]

            # reshape & pad to match model input
            hand_dims = x_poses[0,0].flatten().shape[0]
            pad = torch.ones(x_feats.shape[0], hand_dims - x_feats.shape[1])*-99
            x_feats = torch.cat([torch.tensor(x_feats), pad], dim=1).reshape(-1, 1, x_poses.shape[2], x_poses.shape[3])
            x_ratios = x_ratios[:,:, :x_poses.shape[2], :x_poses.shape[3]]
            x_clips = torch.cat([x_poses, x_feats, x_ratios], dim=1)
        else:
            x_clips = self.norm_input(torch.from_numpy(x_clips).float())

        super().train(x_clips, y_clips, train_subj_ids=ids_clips,)

def poses_diff(x):
    _, H, W, _ = x.shape

    # x.shape (batch,channel,joint_num,joint_dim)
    x = x[:, 1:, ...] - x[:, :-1, ...]

    # x.shape (batch,joint_dim,channel,joint_num,)
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(H, W),
                      align_corners=False, mode='bilinear')
    x = x.permute(0, 2, 3, 1)
    # x.shape (batch,channel,joint_num,joint_dim)
    return x

def poses_motion(P):
    # different from the original version
    # TODO: check the funtion, make sure it's right
    P_diff_slow = poses_diff(P)
    P_diff_slow = torch.flatten(P_diff_slow, start_dim=2)
    P_fast = P[:, ::2, :, :]
    P_diff_fast = poses_diff(P_fast)
    P_diff_fast = torch.flatten(P_diff_fast, start_dim=2)
    # return (B,target_l,joint_d * joint_n) , (B,target_l/2,joint_d * joint_n)
    return P_diff_slow, P_diff_fast

# Calculate JCD feature
def norm_scale(x):
    return (x-torch.mean(x)) / torch.mean(x)

def get_CG(p, joint_n, frame_l):
    M = []
    # upper triangle index with offset 1, which means upper triangle without diagonal
    iu = torch.triu_indices(joint_n, joint_n, 1)
    dm = torch.cdist(p, p, p=2)
    M = dm[:, iu[0], iu[1]]
    M = norm_scale(M)  # normalize
    return M

class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel, skips=False, linear=False):
        super(c1D, self).__init__()
        self.skips = skips
        self.linear = linear
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)
        self.relu = nn.ReLU() #nn.LeakyReLU(0.2)

    def forward(self, x_in):
        # x (B,D,C)
        x = x_in.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        # Residual
        if self.skips:
            # Downsample if needed
            if output.shape[-1] != x_in.shape[-1]:
                x_in = F.interpolate(x_in, size=(output.shape[-1],), mode='nearest').squeeze(-1)
            output = output + x_in
        if not self.linear:
            output = self.relu(output)
        return output


class cnn_block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel, skips=False):
        super(cnn_block, self).__init__()
        self.skips = skips
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel, linear=self.skips)  # have to apply activation after skip
        if self.skips:
            self.c_identity = nn.Conv1d(input_dims, filters, 1)
            self.relu = nn.ReLU() #nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        # Residual
        if self.skips:
            if output.shape[-1] != x.shape[-1]:
                skip = self.c_identity(x.permute(0,2,1)).permute(0,2,1)
            output = output + skip
            output = self.relu(output)
        return output


class d1D(nn.Module):
    def __init__(self, input_dims, filters, linear=False, skips=False):
        super(d1D, self).__init__()
        self.skips = skips
        self.non_linear = not linear
        self.linear = nn.Linear(input_dims, filters)
        if self.skips:
            self.l_identity = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)
        self.relu = nn.ReLU() #nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        # Residual
        if self.skips:
            # Downsample if needed
            if output.shape[1] != x.shape[1]:
                # x = F.interpolate(x.unsqueeze(1), size=(output.shape[1],), mode='nearest').squeeze(1)
                x = self.l_identity(x)
            output += x
        if self.non_linear:
            output = self.relu(output)
        return output


class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout1d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class DDNet_Original(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num, 
                 sample_format, input_kpts,
                 m_branch=True, 
                 f_branch=False, f_branch_type='lin_embed', f_branch_preconcat_layers=0, f_branch_num_feats=15,
                 num_linear=1, linear_size=128, skip_connections=False):
        super(DDNet_Original, self).__init__()
        self.frame_l = frame_l
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.feat_d = feat_d
        self.filters = filters
        self.class_num = class_num
        self.sample_format = sample_format
        self.input_kpts = input_kpts
        self.m_branch = m_branch
        self.f_branch = f_branch
        self.f_branch_type = f_branch_type
        self.f_branch_preconcat_layers = f_branch_preconcat_layers
        self.num_linear = num_linear
        self.lin_size = linear_size
        self.skip_connections = skip_connections

        # self.num_hand_features = 21 #15
        self.num_hand_features = f_branch_num_feats + 1 #15   add 1 for rescale ratio
        self.f_branch_warming_up = False

        conv_kernel_size = 3 #3

        # JCD part
        self.jcd_conv1 = nn.Sequential(
            c1D(frame_l, feat_d, 2 * filters, 1),
            spatialDropout1D(0.1)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(0.1), 
        )
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1),
        )

        if self.m_branch:
            # diff_slow part
            self.slow_conv1 = nn.Sequential(
                c1D(frame_l, joint_n * joint_d, 2 * filters, 1),
                spatialDropout1D(0.1)
            )
            self.slow_conv2 = nn.Sequential(
                c1D(frame_l, 2 * filters, filters, 3),
                spatialDropout1D(0.1)
            )
            self.slow_conv3 = c1D(frame_l, filters, filters, 1)
            self.slow_pool = nn.Sequential(
                nn.MaxPool1d(kernel_size=2),
                spatialDropout1D(0.1)
            )

            # fast_part
            self.fast_conv1 = nn.Sequential(
                c1D(frame_l//2, joint_n * joint_d, 2 * filters, 1), spatialDropout1D(0.1))
            self.fast_conv2 = nn.Sequential(
                c1D(frame_l//2, 2 * filters, filters, 3,), spatialDropout1D(0.1))
            self.fast_conv3 = nn.Sequential(
                c1D(frame_l//2, filters, filters, 1,), spatialDropout1D(0.1))

            self.block1_in_dims = 3 * filters
        else:
            self.block1_in_dims = 1 * filters

        # after cat sizes
        self.block1_in_ch = frame_l//2
        self.block1_filters = 2 * filters

        self.block2_in_ch = self.block1_in_ch//2
        self.block2_in_dims = self.block1_filters
        self.block2_filters = 2 * self.block1_filters

        self.block3_in_ch = self.block2_in_ch//2
        self.block3_in_dims = self.block2_filters
        self.block3_filters = 2 * self.block2_filters

        # 
        if self.f_branch:
            for i in range(self.f_branch_preconcat_layers):
                setattr(self, f'pre_concat_linear{i+1}', nn.Sequential(
                    d1D(self.lin_size, self.lin_size, skips=self.skip_connections),
                    nn.Dropout(0.25)
                ))

        # after cat
        self.block1 = cnn_block(self.block1_in_ch, self.block1_in_dims, self.block1_filters, conv_kernel_size, self.skip_connections)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block2 = cnn_block(self.block2_in_ch, self.block2_in_dims, self.block2_filters, conv_kernel_size, self.skip_connections)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(
            kernel_size=2), spatialDropout1D(0.1))

        self.block3 = nn.Sequential(
            cnn_block(self.block3_in_ch, self.block3_in_dims, self.block3_filters, conv_kernel_size, self.skip_connections), spatialDropout1D(0.1))

        if self.f_branch:
            if self.f_branch_type == 'lin_embed':
                self.f_embed_size = 64
                self.f_embed = d1D(self.num_hand_features, self.f_embed_size, linear=True)  # Linear embedding
            elif self.f_branch_type == 'cascade_mlp':
                self.f_embed_size = 64
                self.f_embed = nn.Sequential(
                    d1D(self.num_hand_features, 32),
                    nn.Dropout(0.2),
                    d1D(32, 64),
                    nn.Dropout(0.3),
                    d1D(64, self.f_embed_size),
                    nn.Dropout(0.3),                    
                )

            self.lin1_in = self.block3_filters + self.f_embed_size
        else:
            self.lin1_in = self.block3_filters

        self.linear1 = nn.Sequential(
            d1D(self.lin1_in, self.lin_size, skips=self.skip_connections),
            nn.Dropout(0.25)
        )
        for i in range(self.num_linear-1):
            setattr(self, f'linear{i+2}', nn.Sequential(
                d1D(self.lin_size, self.lin_size, skips=self.skip_connections),
                nn.Dropout(0.25)
            ))

        self.linear_out = nn.Linear(self.lin_size, class_num)

    def forward(self, P, M=None):
        if self.sample_format in ['scaled_kpt', 'scaled_kpt_hf'] :
            # remove rescale ratio from end of sample
            rescale_ratios = P[:,-1, 0, 0]
            if self.f_branch:
                hand_feats = P[:, -2].flatten(1)[:, :self.num_hand_features - 1]
                # hand_feats = self.f_embed(hand_feats)

            P = P[:, :self.frame_l]
        # P_og = P.detach().clone()
        
        # Only use desired kpts
        P = P[:, :, self.input_kpts]

        # Get JCD feature map
        M = []
        for i in range(P.shape[0]):
            M.append(get_CG(P[i], self.joint_n, self.frame_l))
        M = torch.stack(M).float()

        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0, 2, 1)
        # pool will downsample the D dim of (B,C,D)
        # but we want to downsample the C channels
        # 1x1 conv may be a better choice
        x = self.jcd_pool(x)
        x = x.permute(0, 2, 1)

        if self.m_branch:
            diff_slow, diff_fast = poses_motion(P)
            x_d_slow = self.slow_conv1(diff_slow)
            x_d_slow = self.slow_conv2(x_d_slow)
            x_d_slow = self.slow_conv3(x_d_slow)
            x_d_slow = x_d_slow.permute(0, 2, 1)
            x_d_slow = self.slow_pool(x_d_slow)
            x_d_slow = x_d_slow.permute(0, 2, 1)

            x_d_fast = self.fast_conv1(diff_fast)
            x_d_fast = self.fast_conv2(x_d_fast)
            x_d_fast = self.fast_conv3(x_d_fast)
            # x,x_d_fast,x_d_slow shape: (B,framel//2,filters)

            x = torch.cat((x, x_d_slow, x_d_fast), dim=2)

        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        # max pool over (B,C,D) C channels
        x = torch.max(x, dim=1).values

        if self.f_branch:
            # run through pre-concat layers if needed
            if self.f_branch_preconcat_layers > 0:
                for i in range(self.f_branch_preconcat_layers):
                    x = getattr(self, f'pre_concat_linear{i+1}')(x)

        #     # Get, embed, and fuse handcrafted features from pose series
            rescale_ratios = rescale_ratios.unsqueeze(1)
            
            # Disable handcraft feats until a few eps into training
            if self.f_branch_warming_up:
                hand_feats = torch.zeros((x.shape[0], self.f_embed_size)).to(x.device)
            else:
                hand_feats = torch.cat((hand_feats, rescale_ratios), dim=1)
                hand_feats = self.f_embed(hand_feats)
            x = torch.cat((x, hand_feats), dim=1)

        for i in range(self.num_linear):
            x = getattr(self, f'linear{i+1}')(x)

        x = self.linear_out(x)
        return x