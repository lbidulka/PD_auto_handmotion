import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import medfilt
import scipy.ndimage.interpolation as inter

from .base_deepnet import Base_DeepNet
import utils.dataloader as loader
import utils.loss
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
        self.drop_last = True
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.print_loss = True
        self.print_epochs = 1

        self.flip_L_to_R = False  # swap all L hands to R hands

        self.full_seq = True # use full sequence?
        # self.input_kpts = [0, 8, 12, 16, 20]    # wrist and all finger tips
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
            self.filters = 32
            self.m_branch = True  # Use motion branch?
            self.f_branch = False  # Use handcrafted feature branch?
            if self.f_branch: self.use_ratio = True

            if self.full_seq:
                self.frame_l = 256 # max len for CAMERA = 1467
            else:
                self.frame_l = 80   # network input length     80

            # Training params
            self.selection_metric = 'loss'    # f1, loss
            self.val_frac = 0.30
            if self.datasets == 'PD4T':
                self.batch_size = 64
                self.num_epochs = 50
                self.lr = 5e-4
            elif self.datasets == 'CAMERA,PD4T' or self.datasets == 'PD4T,CAMERA':
                self.batch_size = 64
                self.num_epochs = 50
                self.lr = 5e-4
            else:
                if self.full_seq:
                    self.batch_size = 32
                    self.num_epochs = 50
                    self.lr = 5e-4
                else:
                    self.batch_size = 64
                    self.num_epochs = 35
                    self.lr = 5e-4
            # self.criterion = 'CrossEntropy'
            self.loss_type = 'Focal'
            self.focal_gamma = 1

            self.scheduler_type = None #'cosine'
            self.scheduler_lr_min = 1e-6
            self.scheduler_T_max = self.num_epochs//2

            self.transforms = [loader.noise_rand, loader.scale_rand]
            self.transforms_p = [0.9, 0.9]
        
        self._build_model()
    
    def _build_model(self):
        self.model = DDNet_Original(self.frame_l, self.joint_n, self.joint_d, self.feat_d, self.filters, self.clc, 
                                    self.sample_format, self.input_kpts, self.m_branch, self.f_branch)
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
            vid_clips = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

        # Get avg pred over all clips in each input sample
        self.model.eval()
        preds_all_clips = []
        for clips in vid_clips:
            preds_clip = []
            for clip in clips:
                clip = self.norm_input(clip.unsqueeze(0))
                clip = clip.to(self.device)
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
        # if self.full_seq:
        #     # get final frame index for each sample, ie first frame where == (0,0,0)
        #     is_zero = (x == 0).all(dim=3).all(dim=2)
        #     x_seq_last_idxs = torch.max(is_zero, dim=1)[1]

        #     # DEBUG: trim actual sequence to max len if too long
        #     for i, clip in enumerate(x):
        #         if x_seq_last_idxs[i] > self.frame_l:
        #             # set early frames to 0
        #             x[i, x_seq_last_idxs[i] - self.frame_l:] = 0
        #             # shift first non-zero frame to start
        #             # trim total length to frame_l

        # swap all L hands to R hands
        if self.flip_L_to_R:
            for i, clip in enumerate(x):
                # if self.full_seq:
                #     thumb_x = ((clip[:x_seq_last_idxs[i] + 1,4,0] + clip[:x_seq_last_idxs[i] + 1,3,0] + clip[:x_seq_last_idxs[i] + 1,2,0] + clip[:x_seq_last_idxs[i] + 1,1,0]) / 4).mean(0)
                #     root_x = clip[:x_seq_last_idxs[i] + 1,0,0].mean(0)
                # else: 
                thumb_x = ((clip[:,4,0] + clip[:,3,0] + clip[:,2,0] + clip[:,1,0]) / 4).mean(0)
                root_x = clip[:,0,0].mean(0)
                if thumb_x < root_x:
                    x[i,:,:,0] *= -1

        # normalize size & mean
        for i, clip in enumerate(x):
            # if self.full_seq:
            #     # palm size
            #     wrist = clip[:x_seq_last_idxs[i],0]
            #     palm_vector = ((clip[:x_seq_last_idxs[i],5] - wrist) + (clip[:x_seq_last_idxs[i],9] - wrist) + (clip[:x_seq_last_idxs[i],13] - wrist) + (clip[:x_seq_last_idxs[i],17] - wrist)) / 4
            #     palm_size = np.linalg.norm(palm_vector, axis=1).reshape(-1,1,1)
            #     # mean root joint 
            #     mean_root = clip[:x_seq_last_idxs[i],0].mean(0)
                
            #     x[i,:x_seq_last_idxs[i]] /= palm_size
            #     x[i,:x_seq_last_idxs[i]] -= mean_root
            # else:
            # palm size
            wrist = clip[:,0]
            palm_vector = ((clip[:,5] - wrist) + (clip[:,9] - wrist) + (clip[:,13] - wrist) + (clip[:,17] - wrist)) / 4
            palm_size = np.linalg.norm(palm_vector, axis=1).reshape(-1,1,1)
            # for all 0 frames, set palm size to 1
            palm_size[palm_size == 0] = 1
            # mean root joint
            mean_root = clip[:,0].mean(0)
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

    def train(self, x, y, 
              train_subj_ids=None,
              x_val=None, y_val=None):
        '''
        '''
        # if self.full_seq:
        #     # change max index padding to 0 padding
        #     x_clips = x
        #     # set all frames after last frame to 0
        #     idx_mask = (x == x[:,-1:]).all(axis=(2,3))
        #     x_clips[idx_mask] = 0            
        #     y_clips = y

        #     ids_clips = train_subj_ids
        # else:
            
        # make clips
        if self.sample_format == 'unscaled_kpt':
            vid_clips = self.make_clips(x)
            
            # vid_clips = [
            #     [clip for clip in clips if clip.shape[0] == self.frame_l] for clips in vid_clips
            # ]
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
    for f in range(frame_l):
        # iterate all frames, calc all frame's JCD Matrix
        # p[f].shape (15,2)
        d_m = torch.cdist(p[f], p[f], p=2)
        d_m = d_m[(iu[0], iu[1])]
        # the upper triangle of Matrix and then flattned to a vector. Shape(105)
        M.append(d_m)
    M = torch.stack(M)
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
                 m_branch=True, f_branch=False):
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

        self.num_hand_features = 4

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
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(128, class_num)

    def forward(self, P, M=None):
        if self.sample_format == 'scaled_kpt':
            # remove rescale ratio from end of sample
            rescale_ratios = P[:, -1, 0, 0]
            P = P[:, :-1,]
        P_og = P.detach().clone()
        
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
            # Get, embed, and fuse handcrafted features from pose series
            rescale_ratios = rescale_ratios.unsqueeze(1) #.repeat([1, 10])
            hand_feats = self.get_handcraft_features(P_og).to(rescale_ratios.device)
            hand_feats = torch.cat((hand_feats, rescale_ratios), dim=1)

            hand_feats = self.f_embed(hand_feats)
            x = torch.cat((x, hand_feats), dim=1)

        x = self.linear1(x)
        # x = self.linear2(x)
        x = self.linear3(x)
        return x
    
    def get_handcraft_features(self, P):
        '''
        Compute some handcrafted features given the input batch of pose series P

        args:
            P: (B, frame_l, joint_n, joint_d) tensor
        '''
        # UDPRS features from fingertip-palm distances
        dists = utils.features.get_finger_palm_distance(P.cpu())
        dists = [dist.mean(axis=1) for dist in dists]
        peak_idxs, peak_vals = utils.features.get_cycle_peaks(dists, keep=10, savgol_win=5, prominence=0.10, min_peak_dist=10)
        # peak_idxs = np.array(peak_idxs)
        # peak_vals = np.array(peak_vals)

        min_num_peaks = 7
        hesitation_resid_thresh = 0.2
        amp_dec_thresh = 0.9

        num_hesitations = utils.features.get_UPDRS_num_hesitations(peak_idxs, peak_vals, 
                                                                min_num_peaks, hesitation_resid_thresh, score=False)
        amp_dec_idxs = utils.features.get_UPDRS_amplitude_decrement(peak_vals, min_num_peaks, 
                                                                    amp_dec_thresh, score=False)
        slowings = utils.features.get_UPDRS_slowing(peak_idxs, min_num_peaks, score=False)

        features = torch.tensor([num_hesitations, amp_dec_idxs, slowings]).permute(1,0).float()
        return features