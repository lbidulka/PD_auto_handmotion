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

class DDNet(Base_DeepNet):
    def __init__(self, task, datasets, device):
        super().__init__()
        self.name = 'ddnet'
        self.datasets = datasets
        
        # Task
        self.task = task
        self.labeler_idx = 1

        # Data params
        self.shuffle = True
        self.drop_last = False
        self.device = torch.device(device) #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.print_loss = True
        self.print_epochs = 1

        self.flip_L_to_R = False  # swap all L hands to R hands
        self.zero_palm = False   # zero poses to palm (kpt 0)

        if task == 'binclass':
            raise NotImplementedError
        elif task == 'multiclass':
            self.frame_l = 80   # the length of frames
            self.joint_n = 21   # the number of joints
            self.joint_d = 3    # the dimension of joints
            self.clc = 4        # num classes
            self.feat_d = 210   # flat JCD len
            self.filters = 32
            self.m_branch = True  # Use motion branch?

            # Training params
            self.val_frac = 0.3
            if self.datasets == 'PD4T':
                self.batch_size = 128
                self.num_epochs = 50
            elif self.datasets == 'CAMERA,PD4T' or self.datasets == 'PD4T,CAMERA':
                self.batch_size = 128
                self.num_epochs = 30
            else:
                self.batch_size = 64
                self.num_epochs = 50
            self.lr = 1e-3
            # self.criterion = 'CrossEntropy'
            self.loss_type = 'Focal'
            self.focal_gamma = 0.75

            self.scheduler_type = None #'cosine'
            self.scheduler_lr_min = 1e-6
            self.scheduler_T_max = self.num_epochs//2

            self.transforms = [loader.noise_rand,]
            self.transforms_p = [0.8,]
        
        self._build_model()
    
    def _build_model(self):
        self.model = DDNet_Original(self.frame_l, self.joint_n, self.joint_d, self.feat_d, self.filters, self.clc, self.m_branch)
        self.model.to(self.device)
        if self.task == 'multiclass': 
            if self.loss_type == 'CrossEntropy':
                self.criterion = torch.nn.CrossEntropyLoss()
            elif self.loss_type == 'Focal':
                self.criterion = utils.focal_loss.FocalLoss(gamma=self.focal_gamma)
        elif self.task == 'binclass':
            self.criterion = torch.nn.BCELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_T_max, eta_min=self.scheduler_lr_min)
        

    def __call__(self, x):
        '''
        fwd pass
        '''
        # x = x[:, :self.frame_l,]
        
        # DEBUG: split into as many clips as possible
        # clips = torch.split(torch.tensor(x, dtype=torch.float32), self.frame_l, dim=1)
        vid_clips = [
            torch.split(torch.tensor(vid[:int(vid[-1,0,0])], dtype=torch.float32), self.frame_l, dim=0) for vid in x
        ]

        # Get avg pred over all clips in each input sample
        self.model.eval()
        preds_all_clips = []
        for clips in vid_clips:
            # trim last clip if too short
            if clips[-1].shape[1] < self.frame_l:
                clips = clips[:-1]
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
            preds_all_clips.append(torch.stack(preds_clip).float().mean())
        avg_pred = torch.stack(preds_all_clips).round()
        return avg_pred

    def norm_input(self, x):
        '''
        Normalize input clips: [B, # frames, J, D]
        '''
        # zero poses to palm (kpt 0)
        if self.zero_palm:
            x -= x[:, :, :1, :]
        # swap all L hands to R hands
        if self.flip_L_to_R:
            for i, clip in enumerate(x):
                thumb_x = ((clip[:,4,0] + clip[:,3,0] + clip[:,2,0] + clip[:,1,0]) / 4).mean(0)
                root_x = clip[:,0,0].mean(0)
                if thumb_x < root_x:
                    x[i,:,:,0] *= -1

        # normalize size & mean
        for i, clip in enumerate(x):
            # palm size
            wrist = clip[:,0]
            palm_vector = ((clip[:,5] - wrist) + (clip[:,9] - wrist) + (clip[:,13] - wrist) + (clip[:,17] - wrist)) / 4
            palm_size = np.linalg.norm(palm_vector, axis=1).reshape(-1,1,1)
            x[i] /= palm_size
            # mean root joint 
            mean_root = clip[:,0].mean(0)
            x[i] -= mean_root

        return x

    def train(self, x, y, 
              train_subj_ids=None,
              x_val=None, y_val=None):
        '''
        '''
        # split into as many clips as possible, trimming max index to that provided in the sample
        vid_clips = [
            torch.split(torch.tensor(vid[:int(vid[-1,0,0])], dtype=torch.float32), self.frame_l, dim=0) for vid in x
        ]
        vid_clips = [
            [clip for clip in clips if clip.shape[0] == self.frame_l] for clips in vid_clips
        ]
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

        x_clips = self.norm_input(x_clips)

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
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
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
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num, m_branch=True):
        super(DDNet_Original, self).__init__()
        self.frame_l = frame_l
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.feat_d = feat_d
        self.filters = filters
        self.class_num = class_num
        self.m_branch = m_branch

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

        # after cat
        self.block1 = block(frame_l//2, 3 * filters, 2 * filters, 3)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(
            kernel_size=2), spatialDropout1D(0.1))

        self.block3 = nn.Sequential(
            block(frame_l//8, 4 * filters, 8 * filters, 3), spatialDropout1D(0.1))

        self.linear1 = nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(128, class_num)

    def forward(self, P, M=None):
        # Get JCD feature
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

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x