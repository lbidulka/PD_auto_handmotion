import torch
import wandb
import sklearn.metrics

import utils.dataloader as loader
import utils.data as data_utils

class Base_DeepNet():
    '''
    Base class for deep learning models
    '''
    def __init__(self) -> None:
        self.name = 'default: base_deepnet'
        self.use_ratio = False
        
        self.scheduler = None
        self.transforms = []
        self.transforms_p = []

        self.equalize_class_samples = False
        self.selection_metric = 'loss'  # for choosing best valset model: f1, loss
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

    def get_model_dict(self):
        return self.model.state_dict()

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

    def make_trainval_sets(self, x_train, y_train, x_val, y_val):
        trainset = loader.CustomTensorDataset(tensors=(x_train, y_train), 
                                                    transforms=self.transforms, 
                                                    transforms_p=self.transforms_p, 
                                                    use_ratio=self.use_ratio)
        valset = loader.CustomTensorDataset(tensors=(x_val, y_val), 
                                                transforms=self.transforms, 
                                                transforms_p=self.transforms_p, 
                                                use_ratio=self.use_ratio)
        return trainset, valset

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
            if self.equalize_class_samples:
                x_train, y_train = data_utils.equalize_class_samples(x_train, y_train)
            
            x_train = torch.tensor(x_train).float()
            x_val = torch.tensor(x_val).float()
            y_train = torch.tensor(y_train).long() if self.task == 'multiclass' else torch.tensor(y_train).float()
            y_val = torch.tensor(y_val).long() if self.task == 'multiclass' else torch.tensor(y_val).float()
            train_ids = torch.tensor(train_ids).unique()
            val_ids = torch.tensor(val_ids).unique()

            # setup loss w/ class weights if needed
            if self.loss_type == 'Focal':
                class_weights_idx = 1   # TEMP: have to choose label to use for counting
                class_weights = torch.bincount(y_train[:, class_weights_idx].flatten())
                self.class_weights = 1 / class_weights
                focal_alpha = self.class_weights * (self.clc / torch.linalg.norm(self.class_weights, ord=1))   # norm 
                self._build_criterion(focal_alpha.to(self.device))

            trainset, valset = self.make_trainval_sets(x_train, y_train, x_val, y_val)
        else:
            trainset, valset = torch.utils.data.random_split(loader.CustomTensorDataset(tensors=(x_tensor, y_tensor), 
                                                                                        transforms=self.transforms,
                                                                                        transforms_p=self.transforms_p, 
                                                                                        use_ratio=self.use_ratio), 
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


    def train(self, x, y,
              train_subj_ids=None,
              x_val=None, y_val=None):
        '''
        '''
        # Setup data
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
                                                  use_ratio=self.use_ratio,
                                                  seq_len=self.seq_len)
            valset = loader.CustomTensorDataset(tensors=(x_val_tensor, y_val_tensor), 
                                                # transforms=transforms, 
                                                # transforms_p=transforms_p, 
                                                use_ratio=self.use_ratio,
                                                seq_len=self.seq_len)
            sampler = None

        if 'cuda' in self.device.type:
            pin_memory = True
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, 
                                                  shuffle=self.shuffle, drop_last=self.drop_last, 
                                                  sampler=sampler, num_workers=self.num_workers,
                                                  pin_memory=pin_memory)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, 
                                                shuffle=self.shuffle, drop_last=False, #self.drop_last,
                                                num_workers=self.num_workers, pin_memory=pin_memory)

        # Train
        best_model = None
        for epoch in range(self.num_epochs):
            train_loss, train_f1, train_acc = 0, 0, 0
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                if (inputs.shape[0] == 1):
                    inputs = torch.cat([inputs, inputs], dim=0)
                    labels = torch.cat([labels, labels], dim=0)
                outputs = self.model(inputs)
                # catch Nan
                if torch.isnan(outputs).any():
                    continue
                loss = self.loss(outputs, labels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                # log metrics
                if self.task == 'multiclass':
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    train_f1 += sklearn.metrics.f1_score(labels[:,1].cpu().numpy(), preds, average='weighted')
                    train_acc += sklearn.metrics.accuracy_score(labels[:,1].cpu().numpy(), preds)
            # Validate
            self.model.eval()
            val_loss, val_f1, val_acc = 0, 0, 0
            with torch.no_grad():
                for data in valloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.loss(outputs, labels)
                    val_loss += loss.item()
                    # log metrics
                    if self.task == 'multiclass':
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        # make sample weights based on sample weights
                        # if self.class_weights is not None:
                        #     f1_sample_weights = self.class_weights[labels[:,1].cpu().numpy()]
                        # else:
                        #     f1_sample_weights = None
                        val_f1 += sklearn.metrics.f1_score(labels[:,1].cpu().numpy(), preds, average='weighted',)# sample_weight=f1_sample_weights)
                        val_acc += sklearn.metrics.accuracy_score(labels[:,1].cpu().numpy(), preds)
            # log
            metrics = {
                'train': {
                    'loss': train_loss / len(trainloader),
                    'f1': train_f1 / len(trainloader),
                    'acc': train_acc / len(trainloader),
                },
                'val': {
                    'loss': val_loss / len(valloader),
                    'f1': val_f1 / len(valloader),
                    'acc': val_acc / len(valloader),
                }
            }
            wandb_log = {
                    'train_loss': metrics['train']['loss'], 'val_loss': metrics['val']['loss'],
                    'train_f1': metrics['train']['f1'], 'val_f1': metrics['val']['f1'],
                }
            metrics_str = 'loss (train,val): {:.3f}, {:.3f} | f1 (train,val): {:.3f}, {:.3f} | acc (train,val): {:.3f}, {:.3f}'.format(metrics['train']['loss'], 
                                                                                                           metrics['val']['loss'], 
                                                                                                           metrics['train']['f1'], 
                                                                                                           metrics['val']['f1'],
                                                                                                           metrics['train']['acc'], 
                                                                                                           metrics['val']['acc'])
            if self.print_loss and (epoch % self.print_epochs == 0):
                print(f'|| Epoch {epoch} ||  ' + metrics_str)
                if self.scheduler is not None and self.print_lr:
                    print(f'|| LR: {self.scheduler.get_last_lr()[0]:.6f}')

            # Save best model
            save_model = False
            if epoch == 0:
                save_model = True
            elif (self.selection_metric == 'f1') and (metrics['val'][self.selection_metric] > best_val_metric):
                save_model = True
            elif (self.selection_metric == 'loss') and (metrics['val'][self.selection_metric] < best_val_metric):
                save_model = True
            if save_model:
                best_val_metric = metrics['val'][self.selection_metric]
                best_metrics = metrics
                best_model = self.model.state_dict()
                print(f'                          ---> New best saved @ Ep {epoch}: ' + metrics_str)
            wandb_log.update({
                'best_train_f1': best_metrics['train']['f1'], 'best_val_f1': best_metrics['val']['f1'],
            })
            # Dump to wandb if available
            if wandb.run is not None:
                wandb.log(wandb_log)
            
            if self.scheduler is not None:
                self.scheduler.step()     

        # Load best model after training
        if best_model is not None:
            self.model.load_state_dict(best_model)
            