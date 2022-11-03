from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

def transfrom(reshape=None, transpose=None):
    return reshape,transpose
    
def read_data(data_dir, label_dir, transfrom): 
        np_data = np.load(data_dir)
        np_label = np.load(label_dir).reshape(-1)
        reshape, transpose = transfrom
        if transpose:
            np_data = np_data.transpose(*transpose)
        if reshape:
            num = np_data.shape[0]
            np_data = np_data.reshape((num,*reshape))
        data = torch.from_numpy(np_data).float()
        label = torch.from_numpy(np_label).int()
        return data,label 
#dataset
class npdata(Dataset):
    def __init__(self, img, label, transform):
        super(Dataset)
        self.transform = transform
        self.img = img
        self.label = label
        assert len(self.img) == len(self.label)
    def __getitem__(self,idx):
        img = self.img[idx]
        label = self.label[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.img)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
