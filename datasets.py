import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import cfg
from natsort import natsorted


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        self.files = []
        if self.mode == "train":
            if cfg.dataset_train_mode == 'DIV2K':
                self.TRAIN_PATH = cfg.train_path_div2k
                self.format_train = 'png'
                self.files = natsorted(sorted(glob.glob(self.TRAIN_PATH + "/*." + self.format_train)))

            if cfg.dataset_train_mode == 'ImageNet':
                for d in os.listdir(cfg.train_path_imagenet):
                    dict_path = cfg.train_path_imagenet + d
                    if os.path.isdir(dict_path):
                        files = [os.path.join(dict_path, f) for f in os.listdir(dict_path)]
                        self.files.extend(files)

        if self.mode == "val":
            if cfg.dataset_val_mode == 'DIV2K':
                self.VAL_PATH = cfg.val_path_div2k
                self.format_val = 'png'
                self.files = natsorted(sorted(glob.glob(self.VAL_PATH + "/*." + self.format_val)))

            if cfg.dataset_val_mode == 'ImageNet':
                for d in os.listdir(cfg.val_path_imagenet):
                    dict_path = cfg.val_path_imagenet + d
                    if os.path.isdir(dict_path):
                        files = [os.path.join(dict_path, f) for f in os.listdir(dict_path)]
                        self.files.extend(files)

        if self.mode == "test":
            if cfg.dataset_test_mode == 'DIV2K':
                self.TEST_PATH = cfg.test_path_div2k
                self.format_test = 'png'
                self.files = natsorted(sorted(glob.glob(self.TEST_PATH + "/*." + self.format_test)))

            if cfg.dataset_test_mode == 'ImageNet':
                for d in os.listdir(cfg.test_path_imagenet):
                    dict_path = cfg.test_path_imagenet + d
                    if os.path.isdir(dict_path):
                        files = [os.path.join(dict_path, f) for f in os.listdir(dict_path)]
                        self.files.extend(files)

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index]).convert('RGB')
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == "train" and cfg.dataset_train_mode == 'ImageNet':
            return cfg.train_imagenet_size
        if self.mode == "val" and cfg.dataset_val_mode == 'ImageNet':
            return cfg.val_imagenet_size
        if self.mode == "test" and cfg.dataset_test_mode == 'ImageNet':
            return cfg.test_imagenet_size
        return len(self.files)

if cfg.dataset_val_mode == 'DIV2K':
    cropsize_val = cfg.cropsize_val_div2k
if cfg.dataset_val_mode == 'ImageNet':
    cropsize_val = cfg.cropsize_val_imagenet

if cfg.dataset_test_mode == 'DIV2K':
    cropsize_test = cfg.cropsize_test_div2k
if cfg.dataset_test_mode == 'ImageNet':
    cropsize_test = cfg.cropsize_test_imagenet

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(cfg.cropsize_train),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(cropsize_val),
    T.ToTensor(),
])

transform_test = T.Compose([
    T.CenterCrop(cropsize_test),
    T.ToTensor(),
])


train_dataset = Hinet_Dataset(transforms_=transform, mode="train")
val_dataset = Hinet_Dataset(transforms_=transform_val, mode="val")
test_dataset = Hinet_Dataset(transforms_=transform_test, mode="test")
