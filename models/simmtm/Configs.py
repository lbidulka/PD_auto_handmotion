
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.increased_dim = 1
        self.final_out_channels = 8 # encoder out channels
        self.num_classes = 4
        self.num_classes_target = 4
        self.dropout = 0.2
        self.masking_ratio = 0.5
        self.lm = 3 # average length of masking subsequences

        self.kernel_size = 8
        self.stride = 3
        self.features_len = 127
        self.features_len_f = self.features_len

        self.TSlength_aligned = 178

        self.CNNoutput_channel = 10 # 90 # 10 for Epilepsy model

        self.CNN_skip_connections = False    # use skip connections? (NOT YET WORKING)

        # training configs
        self.num_epoch = 40
        self.reinit_classifier = True # reinitialize the classifier before each new fine-tuning?
        self.reinit_encoder = False     # reset encoder to pre- fine-tuning weights afterwards?
        self.freeze_encoder = True  # Freeze the encoder during the training of the classifier? 
        self.finetune_frac = 0.25   # fraction of train data to use for fine-tuning the classifier

        self.debug_recon_eps_printout = 50  # print out reconstruction results at what ep. freq. during training

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        # self.lr = 3e-8 # 3e-4
        # self.lr_f = self.lr
        self.pretrain_lr = 5e-5    # 1e-4
        self.pretrain_epoch = 50
        self.finetune_lr = 5e-4    # 1e-4
        self.finetune_epoch = 50
        self.ft_freq = 5          # fine-tune a classifier every this many epochs

        # masking
        self.masking_ratio = 0.5
        self.lm = 3 # average masked length

        # data parameters
        self.drop_last = True
        self.batch_size = 32

        self.temperature = 0.2
        self.positive_nums = 3

        """For Epilepsy, the target batchsize is 60"""
        self.target_batch_size = 32   # the size of target dataset (the # of samples used to fine-tune).

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50
