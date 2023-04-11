from tqdm.auto import tqdm
from torch.utils.data import Dataset
from os.path import exists
import pandas as pd
import numpy as np
from torchvision.datasets.folder import default_loader
import torch

class Dataset(Dataset):
    def __init__(self, label_path, transform=None, transform_flag=True):
        self._image_paths = []
        self._labels = []
        self.indices = []
        self._label_names = {}
        self.transform = transform
        self.transform_flag = transform_flag

        # Reading the dataframe
        df = pd.DataFrame()
        for path in label_path:
            df_ = pd.read_csv(path)
            df = df.append(df_)

        #df['labels'] = df['labels'].apply(self.group_labels)
        df = df.fillna(' ')
        df = df[df['labels']!=' ']
        self._image_paths = df['paths'].to_list()
        self._labels = df['labels'].to_list()
        for i,labels in enumerate(list(set(self._labels))):
            self._label_names['label'+str(i)]=labels

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        img = default_loader(self._image_paths[idx])
        if self.transform is not None and self.transform_flag:
            img = self.transform(img)
        
        labels = np.array(self._labels[idx]).astype(np.float32)
        self.indices.append(idx)
        
        return img, labels
    
    def get_mean_std(self):
        images = []
        for img, label in self:
            img = np.array(img)
            images.append(img)
        images = np.array(images)
        print(images.shape)
        mean = images.mean(axis=(0,2,3))
        std = images.std(axis=(0,2,3))

        return mean, std

    def get_class_weights(self):
        class_count = self.get_class_distribution()
        class_freq = class_count / np.sum(class_count)

        class_weights = 1.0 / class_freq
        norm_weights = class_weights / np.sum(class_weights)

        return norm_weights

    def get_class_distribution(self):
        return np.bincount(self._labels)

    def get_labels(self):
        labels = np.array(self._labels).astype(np.float32)
        
        return labels
    
    def group_labels(self, n):
        try:
            n = float(n)
            if n<1:
                return 0
            else:
                return 1
        except:
            return ' '