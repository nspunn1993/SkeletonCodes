from data.transformer import DataTransformer
from data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

class BuildData:
    def __init__(self, trans_params, batch_size=32, shuffle=True, path=[]):
        # Data Loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.path = path
        self.trans_params = trans_params

    def build_data(self, transform_type='train'):        
        
        mean = None
        std = None

        data_trans_param = dict(height=self.trans_params['height'], width=self.trans_params['width'], set_type=transform_type, mean=mean, std=std)
        transform=DataTransformer(**data_trans_param).transformer
        dataset = Dataset(self.path,transform=transform)

        if self.trans_params['normalize']:
            mean, std = dataset.get_mean_std()
            print(mean)
            print(std)
            dataset.transform.transforms.append(transforms.Normalize(mean, std))

        self.dataset = dataset

        return dataset
    
    def build_loader(self):
        batch_size = self.batch_size
        sample = None
        shuffle = self.shuffle
        workers = 0

        loader =  DataLoader(self.dataset, sampler=sample, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
        
        # Print the data distribution
        print('Class distribution: {}, Class weights: {}'.format(self.dataset.get_class_distribution(), self.dataset.get_class_weights()))

        return loader
