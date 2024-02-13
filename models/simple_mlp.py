import torch
import numpy as np

import utils.dataloader as loader
from .base_deepnet import Base_DeepNet

# Simple MLP
class SimpleMLP(Base_DeepNet):
    def __init__(self, sample_len, in_channels, 
                 task, ) -> None:
        super().__init__()
        # Task
        self.task = task #'multiclass'
        self.labeler_idx = 0

        # Data params
        self.shuffle = True
        self.drop_last = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 0
        self.print_loss = True
        self.print_epochs = 10        

        if task == 'binclass':
            # Model arch
            self.data_channels = in_channels
            self.input_dim = sample_len
            self.hidden_dim = 256
            self.output_dim = 5 if task == 'multiclass' else 1
            self.dropout = 0.3

            # Training params
            self.val_frac = 0.25
            self.batch_size = 64
            self.num_epochs = 35
            self.lr = 5e-4

        elif task == 'multiclass':
            # Model arch
            self.data_channels = in_channels
            self.input_dim = sample_len
            self.hidden_dim = 128
            self.output_dim = 5 if task == 'multiclass' else 1
            self.dropout = 0.3

            # Training params
            self.val_frac = 0.25
            self.batch_size = 64
            self.num_epochs = 35
            self.lr = 3e-4
        else:
            raise NotImplementedError

        # Init
        self._build_model()

    def _build_model(self):
        self.model = MLP(data_channels=self.data_channels, 
                         input_dim=self.input_dim, 
                         hidden_dim=self.hidden_dim, 
                         output_dim=self.output_dim, 
                         dropout=self.dropout)
        self.model.to(self.device)
        if self.task == 'multiclass':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.task == 'binclass':
            self.criterion = torch.nn.BCELoss(reduction='none')
        # self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


class MLP(torch.nn.Module):
    def __init__(self, data_channels, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.input_dim = input_dim * data_channels
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, self.output_dim)
        return x


