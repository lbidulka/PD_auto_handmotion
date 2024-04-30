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

class DDNet(Base_DeepNet):
    def __init__(self, task, datasets, device,
                 class_weights=None,):
        super().__init__()
        self.name = 'ddnet'
        self.datasets = datasets
        self.class_weights = class_weights
        
        # Task
        self.task = task
        self.labeler_idx = 1

        # Data params
        self.sample_format = 'scaled_kpt'   # input data format: 'scaled_kpt', 'unscaled_kpt', 'scaled', 'unscaled'
        self.combine_34 = True              # combine classes 3/4 to label 3?
        self.shuffle = True
        self.drop_last = False
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.print_loss = True
        self.print_epochs = 1

        self.flip_L_to_R = True  # swap all L hands to R hands

        self.full_seq = True # use full sequence?
        # self.input_kpts = [0, 8, 12, 16, 20] 5   # wrist and all finger tips
        self.input_kpts = [i for i in range(21)]    # all kpts

        if task == 'binclass':
            raise NotImplementedError
        elif task == 'multiclass':
            self.joint_n = len(self.input_kpts)   # the number of joints
            self.joint_d = 3    # the dimension of joints
            self.clc = 4        # num classes
            if self.joint_n == 21:
                self.feat_d = 210
            else:
                self.feat_d = 10 #210   # flat JCD len
            
            self.filters = 16
            self.num_linear = 1 # number of linear layers after conv blocks
            self.m_branch = True  # Use motion branch?
            self.f_branch = True  # Use handcrafted feature branch?
            
            if self.sample_format == 'scaled_kpt': self.use_ratio = True

            if self.full_seq:
                self.frame_l = 256 # max len for CAMERA = 1467
            else:
                self.frame_l = 80   # network input length     80

            # Training params
            self.selection_metric = 'loss'    # f1, loss
            self.val_frac = 0.3
            if self.datasets == 'PD4T':
                self.batch_size = 64
                self.num_epochs = 50
                self.lr = 5e-4
            elif self.datasets == 'CAMERA,PD4T' or self.datasets == 'PD4T,CAMERA':
                self.batch_size = 128
                self.num_epochs = 50
                self.lr = 1e-3
            else:
                if self.full_seq:
                    self.batch_size = 64
                    self.num_epochs = 50
                    self.lr = 5e-4
                else:
                    self.batch_size = 64
                    self.num_epochs = 1 #50
                    self.lr = 5e-4
            # self.loss_type = 'CrossEntropy'
            self.loss_type = 'Focal'
            self.focal_gamma = 2

            self.scheduler_type = None #'cosine'
            self.scheduler_lr_min = 1e-5
            self.scheduler_T_max = int(self.num_epochs*2)
            self.print_lr = False

            self.transforms = [loader.noise_rand,] # loader.scale_rand,]# loader.trim_rand]
            self.transforms_p = [0.9,] # 0.9,]# 0.0]
            self.f_scaler = StandardScaler()    # handcrafted feature normalizer
        
        self._build_model()
    
    def _build_model(self):
        self.model = DDNet_Original(self.frame_l, self.joint_n, self.joint_d, self.feat_d, self.filters, self.clc, 
                                    self.sample_format, self.input_kpts, self.m_branch, self.f_branch, self.num_linear)
        self.model.to(self.device)
        self._build_criterion()
        if self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_T_max, eta_min=self.scheduler_lr_min)
        
    def _build_criterion(self, focal_alpha=None):
        if self.task == 'multiclass': 
            if self.loss_type == 'CrossEntropy':
                self.criterion = torch.nn.CrossEntropyLoss()
            elif self.loss_type == 'Focal':
                self.criterion = utils.focal_loss.FocalLoss(gamma=self.focal_gamma, alpha=focal_alpha)
        elif self.task == 'binclass':
            self.criterion = torch.nn.BCELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
        x_clips = self.norm_input(vid_clips)

        # Get handcraft feats and insert into input, if required
        if self.model.f_branch:
            rescale_ratios = x_clips[:, -1, 0, :1]
            hand_feats = self.get_handcraft_features(x_clips[:, :-1]) if self.use_ratio else self.model.get_handcraft_features(x_clips)
            hand_feats = np.concatenate((hand_feats, rescale_ratios), axis=1)

            # normalization
            hand_feats = self.f_scaler.transform(hand_feats)
            hand_feats = torch.tensor(hand_feats, device=x_clips.device).float()

            # pad and reshape to match model input
            hand_dims = x_clips[0,0].flatten().shape[0]
            pad = torch.ones(hand_feats.shape[0], hand_dims - hand_feats.shape[1])*-99
            hand_feats = torch.cat([hand_feats, pad], dim=1).reshape(-1, 1, x_clips.shape[2], x_clips.shape[3])
            # insert at 2nd last position
            x_clips = torch.cat([x_clips[:,:-1], hand_feats, x_clips[:,-1:]], dim=1)

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
        for i, clip in enumerate(x):
            if self.use_ratio:
                clip = clip[:-1]

            # swap all L hands to R hands if needed
            if self.flip_L_to_R:
                # check cross prod of index-root and pinky-root
                cross = torch.cross(clip[:,5] - clip[:,0], clip[:,17] - clip[:,0], dim=1).mean(0)
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
            mean_root = clip[:,0].mean(0)

            if self.use_ratio:
                x[i,:-1] /= palm_size
                x[i,:-1] -= mean_root
            else:
                x[i] /= palm_size
                x[i] -= mean_root

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
                                                transforms=self.transforms, 
                                                transforms_p=self.transforms_p, 
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

    def get_handcraft_features(self, P, labels=None, info_gain=True, num_info_feats=20):
        '''
        Compute some handcrafted features given the input batch of pose series P

        args:
            P: (B, frame_l, joint_n, joint_d) tensor
        '''
        # Get finger-palm distances & cycle peaks/valleys/idxs
        dists = utils.features.get_finger_palm_distance(P.cpu())
        dists = [dist.mean(axis=1) for dist in dists]
        peak_idxs, peak_vals = utils.features.get_cycle_peaks(dists, keep=10, savgol_win=5, prominence=0.10, min_peak_dist=10)
        valley_idxs, valley_vals = utils.features.get_cycle_valleys(dists, peak_idxs, savgol_win=5)
        # peak_idxs, peak_vals, valley_idxs, valley_vals, peak_width, valley_width = utils.features.adjust_peaks(dists, peak_idxs, peak_vals, valley_idxs, valley_vals)
        # peak_idxs = np.array(peak_idxs)
        # peak_vals = np.array(peak_vals)

        min_num_peaks = 7
        hesitation_resid_thresh = 0.2
        amp_dec_thresh = 0.9

        # UPDRS features
        num_hesitations = utils.features.get_UPDRS_num_hesitations(peak_idxs, peak_vals, 
                                                                min_num_peaks, hesitation_resid_thresh, score=False)
        amp_dec_idxs = utils.features.get_UPDRS_amplitude_decrement(peak_vals, min_num_peaks, 
                                                                    amp_dec_thresh, score=False)
        slowings = utils.features.get_UPDRS_slowing(peak_idxs, min_num_peaks, score=False)
        
        # Catch 22 features
        catch24_feats = utils.features.get_catch22_features(dists)
        
        # Cycle features
        cycle_feats = utils.features.get_cycle_features(dists, peak_idxs, peak_vals, valley_vals)
        effective_distance_completed_mean = [np.mean(eff_dist) for eff_dist in cycle_feats[0]]
        effective_distance_completed_std = [np.std(eff_dist) for eff_dist in cycle_feats[0]]
        total_distance_travelled_mean = [np.mean(dist) for dist in cycle_feats[1]]
        total_distance_travelled_std = [np.std(dist) for dist in cycle_feats[1]]
        cycle_times_mean = [np.mean(times) for times in cycle_feats[2]]
        cycle_times_std = [np.std(times) for times in cycle_feats[2]]
        total_average_speed_mean = [np.mean(spd) for spd in cycle_feats[3]]
        total_average_speed_std = [np.std(spd) for spd in cycle_feats[3]]
        smoothness_mean = [np.mean(s) for s in cycle_feats[5]]
        smoothness_std = [np.std(s) for s in cycle_feats[5]]

        # Other features
        amp_fft_var = utils.features.get_fft_var(dists)

        if info_gain:
            # combine all features into vector for each sample
            all_features = []
            for i in range(len(dists)):
                all_features.append([])
                if len(cycle_feats[0][i]) > 0:
                    for c24_feat in catch24_feats[i]:
                        all_features[i].append(c24_feat)
                    all_features[i].append(num_hesitations[i])
                    all_features[i].append(amp_dec_idxs[i])
                    all_features[i].append(slowings[i])
                    all_features[i].append(effective_distance_completed_mean[i])
                    all_features[i].append(effective_distance_completed_std[i])
                    all_features[i].append(total_distance_travelled_mean[i])
                    all_features[i].append(total_distance_travelled_std[i])
                    all_features[i].append(cycle_times_mean[i])
                    all_features[i].append(cycle_times_std[i])
                    all_features[i].append(total_average_speed_mean[i])
                    all_features[i].append(total_average_speed_std[i])
                    all_features[i].append(smoothness_mean[i])
                    all_features[i].append(smoothness_std[i])
                    all_features[i].append(amp_fft_var[i])
                else: 
                    all_features[i] = [0]*(14 + 24)
            all_features = np.array(all_features)
            if labels is not None:
                self.best_features_ids = self.information_gain_feature_selection(all_features, labels, num_info_feats)
            features = torch.tensor(all_features[:,self.best_features_ids], device=P.device).float()
        else:
            UPDRS_feats = torch.tensor([num_hesitations, amp_dec_idxs, slowings], device=P.device).permute(1,0).float()
            catch24_feats_subset = [0, 1, 2, 4, 6, 10, 15, 21]
            catch24_feats = [[feats[i] for i in catch24_feats_subset] for feats in catch24_feats]
            cycle_feats = np.array([effective_distance_completed_mean, total_average_speed_std, smoothness_mean]).swapaxes(0,1)
            # features = torch.cat([UPDRS_feats, torch.tensor(catch24_feats, device=P.device),], axis=1).float()
            features = torch.cat([UPDRS_feats, torch.tensor(catch24_feats), torch.tensor(cycle_feats)], axis=1).float()
        
        if torch.isnan(features).any():
            features[torch.isnan(features)] = 0
        return features

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

        x_clips = self.norm_input(torch.from_numpy(x_clips).float())

        # precompute handcrafted features
        if self.model.f_branch:
            rescale_ratios = x_clips[:, -1, 0, :1]
            hand_feats = self.get_handcraft_features(x_clips[:, :-1], y_clips[:,self.labeler_idx]) if self.use_ratio else self.get_handcraft_features(x_clips, y_clips[:,self.labeler_idx])
            hand_feats = np.concatenate((hand_feats, rescale_ratios), axis=1)

            # normalization
            self.f_scaler.fit(hand_feats)
            hand_feats = self.f_scaler.transform(hand_feats)
            hand_feats = torch.tensor(hand_feats, device=x_clips.device).float()

            # pad and reshape to match model input
            hand_dims = x_clips[0,0].flatten().shape[0]
            pad = torch.ones(hand_feats.shape[0], hand_dims - hand_feats.shape[1])*-99
            hand_feats = torch.cat([hand_feats, pad], dim=1).reshape(-1, 1, x_clips.shape[2], x_clips.shape[3])
            # insert at 2nd last position
            x_clips = torch.cat([x_clips[:,:-1], hand_feats, x_clips[:,-1:]], dim=1)

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
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output


class d1D(nn.Module):
    def __init__(self, input_dims, filters, linear=False):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        if not self.linear:
            output = F.leaky_relu(output, 0.2)
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
                 m_branch=True, f_branch=False,
                 num_linear=1):
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
        self.num_linear = num_linear

        self.num_hand_features = 21 #15

        # JCD part
        self.jcd_conv1 = nn.Sequential(
            c1D(frame_l, feat_d, 2 * filters, 1),
            spatialDropout1D(0.1)
        )
        self.jcd_conv2 = nn.Sequential(
            c1D(frame_l, 2 * filters, filters, 3),
            spatialDropout1D(0.1)
        )
        self.jcd_conv3 = c1D(frame_l, filters, filters, 1)
        self.jcd_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
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
                c1D(frame_l//2, 2 * filters, filters, 3), spatialDropout1D(0.1))
            self.fast_conv3 = nn.Sequential(
                c1D(frame_l//2, filters, filters, 1), spatialDropout1D(0.1))

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

        # after cat
        self.block1 = block(self.block1_in_ch, self.block1_in_dims, self.block1_filters, 3)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block2 = block(self.block2_in_ch, self.block2_in_dims, self.block2_filters, 3)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(
            kernel_size=2), spatialDropout1D(0.1))

        self.block3 = nn.Sequential(
            block(self.block3_in_ch, self.block3_in_dims, self.block3_filters, 3), spatialDropout1D(0.1))

        if self.f_branch:
            self.f_embed_size = 64
            self.f_embed = d1D(self.num_hand_features, self.f_embed_size, linear=True)  # Linear embedding

            self.lin1_in = self.block3_filters + self.f_embed_size
        else:
            self.lin1_in = self.block3_filters

        self.linear1 = nn.Sequential(
            d1D(self.lin1_in, 128),
            nn.Dropout(0.25)
        )
        for i in range(self.num_linear-1):
            setattr(self, f'linear{i+2}', nn.Sequential(
                d1D(128, 128),
                nn.Dropout(0.25)
            ))

        self.linear_out = nn.Linear(128, class_num)

    def forward(self, P, M=None):
        if self.sample_format == 'scaled_kpt':
            # remove rescale ratio from end of sample
            # rescale_ratios = P[:, -1, 0, 0]
            if self.f_branch:
                hand_feats = P[:, -2].reshape(P.shape[0], -1)[:, :self.num_hand_features]
                hand_feats = self.f_embed(hand_feats)

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
        #     # Get, embed, and fuse handcrafted features from pose series
        #     rescale_ratios = rescale_ratios.unsqueeze(1) #.repeat([1, 10])
        #     hand_feats = self.get_handcraft_features(P_og)
        #     hand_feats = torch.cat((hand_feats, rescale_ratios), dim=1)

        #     hand_feats = self.f_embed(hand_feats)
            x = torch.cat((x, hand_feats), dim=1)

        for i in range(self.num_linear):
            x = getattr(self, f'linear{i+1}')(x)

        x = self.linear_out(x)
        return x