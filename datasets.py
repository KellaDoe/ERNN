import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm


class DimTransform:
    def __init__(self, target_dim, class_split):
        self.target_dim = target_dim
        self.class_split = class_split

    def __call__(self, x):
        if np.argmax(x) in self.class_split[0]:
            label = torch.zeros(self.target_dim)
            label[self.class_split[0].index(np.argmax(x))] = 1
        else:
            label = torch.ones(self.target_dim)
            label = -1. / self.target_dim * label
        return label


class Subset(Dataset):
    def __init__(self, data, label, data_split, transform=None, label_transform=None):
        self.data = data
        self.label = label
        self.data_split = data_split
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return data, label

    def __len__(self):
        return len(self.data)


class Base(Dataset):
    def __init__(self, img_dir, label_dir, transform, aug_transform, label_transform):
        self.data_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.aug_transform = aug_transform
        self.label_transform = label_transform

        self.data = []
        self.label = []
        self.load_data()

        self.idx_by_class = []
        self.train_idx = []
        self.valid_idx_id = []
        self.valid_idx_ood = []

    def load_data(self):
        pass

    def __getitem__(self, index: int) -> (np.ndarray, np.ndarray):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return data, label

    def __len__(self) -> int:
        return len(self.data)

    def split(self, fold: int, class_split: tuple,
              data_transform: torchvision.transforms = None,
              aug_transform: torchvision.transforms = None,
              split_label_transform: torchvision.transforms = None) -> (Subset, Subset, Subset):
        assert len(class_split) == 2, f'Wrong split setting is given! expect 2, given {len(class_split)}.'

        if data_transform is None:
            data_transform = self.transform
        if aug_transform is None:
            aug_transform = self.aug_transform
        if split_label_transform is None:
            split_label_transform = DimTransform(len(class_split[0]), class_split=class_split)

        # split data according to categories
        for i in range(self.label.shape[-1]):
            idx = np.where(np.argmax(self.label, axis=1) == i)[0]
            self.idx_by_class.append(idx)

        for class_id in class_split[1]:
            self.valid_idx_ood += self.idx_by_class[class_id].tolist()

        idx_id = np.setdiff1d(np.arange(len(self.data)), self.valid_idx_ood)
        # valid_idx = np.linspace(fold, len(idx_id), len(idx_id) // 5, endpoint=False, dtype=np.int)
        self.valid_idx_id = idx_id[np.linspace(fold, len(idx_id), len(idx_id) // 5, endpoint=False, dtype=np.int)]
        self.train_idx = np.setdiff1d(idx_id, self.valid_idx_id)
        # self.train_idx = self.train_idx[np.linspace(0, len(self.train_idx), len(self.train_idx) // 5, endpoint=False, dtype=np.int)]

        train_set = Subset([self.data[idx] for idx in self.train_idx], [self.label[idx] for idx in self.train_idx],
                           data_split=class_split,
                           transform=aug_transform,
                           label_transform=split_label_transform)
        valid_set_id = Subset([self.data[idx] for idx in self.valid_idx_id],
                              [self.label[idx] for idx in self.valid_idx_id],
                              data_split=class_split,
                              transform=data_transform,
                              label_transform=split_label_transform)
        valid_set_ood = Subset([self.data[idx] for idx in self.valid_idx_ood],
                               [self.label[idx] for idx in self.valid_idx_ood],
                               data_split=class_split,
                               transform=data_transform,
                               label_transform=split_label_transform)

        return train_set, valid_set_id, valid_set_ood


class ISIC(Base):
    def __init__(self, img_dir, label_dir, transform=None, aug_transform=None, label_transform=None):
        super(ISIC, self).__init__(img_dir, label_dir, transform, aug_transform, label_transform)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        if aug_transform is None:
            self.aug_transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.RandomCrop((224, 224), padding=4),
                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    def load_data(self):
        if not os.path.exists(self.data_dir):
            raise RuntimeError('data path does not exist!')
        if not os.path.exists(self.label_dir):
            raise RuntimeError('label path does not exist!')

        if os.path.isfile(self.label_dir):  # 读取csv标签
            csv_reader = pd.read_csv(self.label_dir, header=0, index_col='image')
        else:
            raise RuntimeError('wrong label path is given!')
        print(f'Start loading ISIC from {self.data_dir}')
        with tqdm(total=len(os.listdir(self.data_dir)), ncols=100) as _tqdm:
            for step, img in enumerate(os.listdir(self.data_dir)):
                p_img = os.path.join(self.data_dir, img)
                if p_img.endswith('jpg'):
                    data = io.imread(p_img)
                    id = img.split('.')[0]
                    label = np.array(csv_reader.loc[id])

                    self.data.append(data)
                    self.label.append(label)
                _tqdm.update(1)
        self.label = np.array(self.label)
        print('Finish loading data!')
