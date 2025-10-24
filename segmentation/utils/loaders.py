from torch.utils.data import Dataset, DataLoader
from masks import _transform_masks
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
import logging

class SemanticDataset(Dataset):
    def __init__(self, X, y, in_transforms=['outer-distance'], out_transforms=None, transforms_kwargs={}):
        self.X = X
        self.y = y
        self.in_transforms = in_transforms
        self.out_transforms=out_transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis = -1

    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):
    
            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]
    
            data_format='channels_last'
            for transform in self.in_transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        y_semantic_list = [ys[0] for ys in y_semantic_list]
        return y_semantic_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y_semantic_list = self._transform_labels(self.y[idx:idx+1])      
        
        if self.out_transforms:
            x, y_semantic_list = self.out_transforms(x, y_semantic_list)
            
        return (x, y_semantic_list)


class CroppingDatasetTorch(Dataset):
    def __init__(self, X, y, in_transforms=['outer-distance'], transforms_kwargs={}, out_transforms=None):
        self.X = X
        self.y = y
        self.in_transforms = in_transforms
        self.out_transforms = out_transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis=-1

        
    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):
    
            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            data_format='channels_last'
            for transform in self.in_transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        y_semantic_list = [ys[0] for ys in y_semantic_list]
        return y_semantic_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y_semantic_list = self._transform_labels(self.y[idx:idx+1]) 

        if self.out_transforms:
            x, y_semantic_list = self.out_transforms(x, y_semantic_list)


        return (x, y_semantic_list)