import torch

import utils.dataloader as loader

class Base_DeepNet():
    '''
    Base class for deep learning models
    '''
    def __init__(self) -> None:
        pass

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
        elif self.task == 'binclass':
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).int()
        else:
            raise NotImplementedError
        return preds
    
    def init_model(self):
        self._build_model()

    def loss(self, outputs, labels):
        if self.task == 'binclass':
            # TEMP: to handle multiple labelers, use all one labeler, except for samples with a -1
            eval_labels = torch.zeros_like(labels[:,0])
            for i in range(len(labels)):
                if (labels[i] == -1).any():
                    eval_labels[i] = labels[i,labels[i] != -1][0]
                else:
                    eval_labels[i] = labels[i,self.labeler_idx]
            # apply sigmoid
            outputs = torch.sigmoid(outputs)
            loss = self.criterion(outputs, eval_labels)

            # apply class weights and reduce
            loss[eval_labels == 0] = loss[eval_labels == 0] * self.class0_reweight
            loss = loss.mean()

        else:
            # TEMP: to handle multiple labelers, use all one labeler, except for samples with a -1
            eval_labels = torch.zeros_like(labels[:,0])
            for i in range(len(labels)):
                if (labels[i] == -1).any():
                    eval_labels[i] = labels[i,labels[i] != -1][0]
                else:
                    eval_labels[i] = labels[i,self.labeler_idx]

            loss = self.criterion(outputs, eval_labels)
        return loss

    def setup_dataset(self, x, y):
        '''
        '''
        # Create dataset
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).long() if self.task == 'multiclass' else torch.from_numpy(y).float()
        transforms = [loader.scale_rand, loader.noise_rand]

        # Split the tensors into Train/val
        val_size = int(x_tensor.shape[0] * self.val_frac)
        train_size = x_tensor.shape[0] - val_size
        trainset, valset = torch.utils.data.random_split(loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), transforms=transforms, use_ratio=self.use_ratio), 
                                                         [train_size, val_size])

        # Setup weighted random sample for trainset
        if self.task == 'binclass':
            class_sample_count = torch.tensor(
                [(y_tensor == 0).sum(), (y_tensor == 1).sum()])
            ratio = class_sample_count[1] / class_sample_count[0]
            self.class0_reweight = ratio
        #     samples_weight = weight[y_tensor.long()]
        #     sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        #     self.shuffle = False
        # else:
        #     sampler = None
        sampler = None
        
        return trainset, valset, sampler


    def train(self, x, y):
        '''
        '''
        # Setup data
        trainset, valset, sampler = self.setup_dataset(x, y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, 
                                                  shuffle=self.shuffle, drop_last=self.drop_last, 
                                                  sampler=sampler, num_workers=self.num_workers)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, 
                                                shuffle=self.shuffle, drop_last=self.drop_last,
                                                num_workers=self.num_workers)

        # Train
        best_model = None
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self.print_loss and (i % 100 == 0) and (epoch % self.print_epochs == 0):
                    print(f'|| Epoch {epoch} Iter {i} || Loss: {loss.item():.3}')
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, labels)
                    val_loss += loss.item()
                    # total += labels.size(0)
                    # if self.task == 'binclass':
                    #     predicted = (outputs > 0.5).int()
                    # else:
                    #     _, predicted = outputs.max(1)
                    # correct += predicted.eq(labels).sum().item()
            val_loss /= len(valloader)
            if self.print_loss and (epoch % self.print_epochs == 0):
                print(f'                     Val Loss: {val_loss:.3f}') #, Val Acc: {(100 * correct / total):.3f}')
            
            # Save best model
            if epoch == 0:
                best_loss = val_loss
            elif val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()
                print(f'                    ---> New best model saved at epoch {epoch}')
        # Load best model after training
        if best_model is not None:
            self.model.load_state_dict(best_model)
            