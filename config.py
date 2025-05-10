from easydict import EasyDict

cfg = EasyDict(__name__='Config: AIS')

# DDP
cfg.port = '12355'
cfg.num_gpus = 1

# Super parameters
cfg.num_hiding = 3
cfg.clamp = 1.0
cfg.init_scale = 0.01

# Optim
cfg.lr_hiding = 1e-4
cfg.lr_authen = 2 * 1e-4
cfg.lr_gen = 2 * 1e-4
cfg.betas = (0.9, 0.999)
cfg.weight_decay = 1e-5

# Train
cfg.epochs = 1000
cfg.batch_size = 24
cfg.cropsize_train = 144

# Val
cfg.cropsize_val_div2k = 1024
cfg.cropsize_val_paris = 128
cfg.cropsize_val_imagenet = 256
cfg.batchsize_val = cfg.num_hiding + 1
cfg.shuffle_val = False
cfg.val_freq = 1

# Test
cfg.cropsize_test_div2k = 1024
cfg.cropsize_test_paris = 128
cfg.cropsize_test_imagenet = 256
cfg.batchsize_test = cfg.num_hiding + 1
cfg.shuffle_test = False

# Dataset
cfg.dataset_train_mode = 'DIV2K'  # ImageNet / DIV2K
cfg.dataset_val_mode = 'DIV2K'  # ImageNet / DIV2K
cfg.dataset_test_mode = 'DIV2K'  # ImageNet / DIV2K

cfg.train_path_div2k = '/home/boot/STU/workspaces/data/DIV2K/train/'
cfg.val_path_div2k = '/home/boot/STU/workspaces/data/DIV2K/val/'
cfg.test_path_div2k = '/home/boot/STU/workspaces/data/DIV2K/test/'

cfg.train_path_imagenet = '/home/boot/STU/workspaces/data/Imagenet/train/'
cfg.val_path_imagenet = '/home/boot/STU/workspaces/data/Imagenet/val/'
cfg.test_path_imagenet = '/home/boot/STU/workspaces/data/Imagenet/test/'

cfg.train_imagenet_size = 20000
cfg.val_imagenet_size = 5000
cfg.test_imagenet_size = 5000

# Checkpoints
cfg.model_path = 'checkpoints/'
cfg.save_freq = 5

# Load
## Train
cfg.suffix_load = 'checkpoint_DIV2K_00000'
cfg.train_next = False
cfg.trained_epoch = 0
cfg.pretrain = False
cfg.pretrain_path = 'checkpoints/'
cfg.suffix_pretrain = 'checkpoint_00000'
## Test
cfg.test_path = 'checkpoints/'
cfg.suffix_test = 'model'
cfg.image_save_path = 'images/'
