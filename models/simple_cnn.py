import torch

from .base_deepnet import Base_DeepNet

class SimpleCNN(Base_DeepNet):
    '''
    Simple CNN model
    '''
    def __init__(self, sample_len, in_channels,
                 task,) -> None:
        super().__init__()
        self.name = 'simple_cnn'
        # Task
        self.task = task #'multiclass'
        self.labeler_idx = 1
        self.use_ratio = False

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
            self.num_init_filters = 16
            self.hidden_dim = 32
            self.output_dim = 5 if task == 'multiclass' else 1
            self.dropout = 0.3

            # Training params
            self.val_frac = 0.25
            self.batch_size = 64
            self.num_epochs = 20
            self.lr = 5e-4

        elif task == 'multiclass':
            # Model arch
            self.data_channels = in_channels
            self.input_dim = sample_len
            self.num_init_filters = 8
            self.hidden_dim = 64
            self.output_dim = 5 if task == 'multiclass' else 1
            self.dropout = 0.3

            # Training params
            self.val_frac = 0.25
            self.batch_size = 64
            self.num_epochs = 35
            self.lr = 5e-4
        else:
            raise NotImplementedError

        # Init
        self._build_model()

    def _build_model(self):
        self.model = CNN(data_channels=self.data_channels, 
                         input_dim=self.input_dim, 
                         num_init_filters=self.num_init_filters,
                         hidden_dim=self.hidden_dim,
                         output_dim=self.output_dim, 
                         dropout=self.dropout)
        self.model.to(self.device)
        if self.task == 'multiclass':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.task == 'binclass':
            self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
class CNN(torch.nn.Module):
    def __init__(self, data_channels, input_dim, 
                 num_init_filters, hidden_dim, 
                 output_dim, dropout):
        super(CNN, self).__init__()
        kernel_size = 5
        stride = 1
        padding = 1
        
        self.conv1 = torch.nn.Conv1d(data_channels, num_init_filters, 
                                     kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(num_init_filters, num_init_filters*2, 
                                     kernel_size=3, stride=1, padding=1)

        # compute activation map size
        map_size = int(((input_dim - kernel_size + 2*padding) / stride) + 1)
        lin_size = (map_size+1) * (num_init_filters//2) # 3456

        self.fc1 = torch.nn.Linear(lin_size, output_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        self.maxpool = torch.nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, input):
        # reorder input
        x = input.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x
