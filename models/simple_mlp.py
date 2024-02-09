import torch
import numpy as np

import utils.dataloader as loader

# Simple MLP
class SimpleMLP():
    def __init__(self, sample_len, in_channels) -> None:
        # Task
        self.task = 'multiclass'

        # Model arch
        self.data_channels = in_channels
        self.input_dim = sample_len #1202
        self.hidden_dim = 128
        self.output_dim = 5
        self.dropout = 0.3

        # Training params
        self.batch_size = 64
        self.num_epochs = 20
        self.lr = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data params
        self.shuffle = True
        self.drop_last = False

        # Init
        self._build_model()
    
    def __call__(self, x):
        '''
        fwd pass
        '''
        # convert to tensor
        self.model.eval()
        logits = self.model(torch.tensor(x, dtype=torch.float32))
        # make preds into class labels
        if self.task == 'multiclass':
            preds = torch.argmax(logits, dim=1)
        else:
            raise NotImplementedError
        return preds

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
            self.criterion = torch.nn.BCELoss()
        # self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, x, y):
        '''
        '''
        # Create dataset
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).long()
        transforms = [loader.scale_rand, loader.noise_rand]
        trainset = loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), transforms=transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

        # Train
        self.model.train()
        # print('Training SimpleMLP...')
        for epoch in range(self.num_epochs):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # if (i % 100 == 0) and (epoch % 5 == 0):
                #     print(f'|Epoch {epoch} Iter {i}| Loss: {loss.item()}')


class MLP(torch.nn.Module):
    def __init__(self, data_channels, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.input_dim = input_dim * data_channels
        self.fc1 = torch.nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


