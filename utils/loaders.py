from torch.utils.data import Dataset, DataLoader
from masks import _transform_masks
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
import logging

class SemanticDataset(Dataset):
    def __init__(self, X, y, transforms=['outer-distance'], min_objects=3, transforms_kwargs={}):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        # self.channel_axis=1
        self.channel_axis = -1
        self.min_objects = min_objects

        # # Remove images with small numbers of cells
        # invalid_batches = []
        # for b in range(self.X.shape[0]):
        #     if len(np.unique(self.y[b])) - 1 < self.min_objects:
        #         invalid_batches.append(b)

        # invalid_batches = np.array(invalid_batches, dtype='int')

        # if invalid_batches.size > 0:
        #     logging.warning('Removing %s of %s images with fewer than %s '
        #                     'objects.', invalid_batches.size, self.X.shape[0],
        #                     self.min_objects)

        # self.X = np.delete(self.X, invalid_batches, axis=0)
        # self.y = np.delete(self.y, invalid_batches, axis=0)

    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):
    
            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]
    
            # data_format='channels_first'
            data_format='channels_last'
            for transform in self.transforms:
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
        
        transform = transforms.Compose([
            transforms.ToImage(),          # Convert the image to a PyTorch tensor
            # transforms.ToDtype(torch.float32, scale=True),
        ])
        
        x, y_semantic_list = transform(x, y_semantic_list)
        return (x, y_semantic_list)


class CroppingDatasetTorch(Dataset):
    def __init__(self, X, y, rotation_range, shear_range, zoom_range, horizontal_flip, vertical_flip, crop_size, batch_size=8, transforms=['outer-distance'], transforms_kwargs={}, seed=0, min_objects=3):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis=-1
        self.seed = seed
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.crop_size = (crop_size, crop_size)
        self.batch_size = batch_size
        self.min_objects = min_objects

        # # Remove images with small numbers of cells
        # invalid_batches = []
        # for b in range(self.X.shape[0]):
        #     if len(np.unique(self.y[b])) - 1 < self.min_objects:
        #         invalid_batches.append(b)

        # invalid_batches = np.array(invalid_batches, dtype='int')

        # if invalid_batches.size > 0:
        #     logging.warning('Removing %s of %s images with fewer than %s '
        #                     'objects.', invalid_batches.size, self.X.shape[0],
        #                     self.min_objects)

        # self.X = np.delete(self.X, invalid_batches, axis=0)
        # self.y = np.delete(self.y, invalid_batches, axis=0)

        
    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):
    
            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            data_format='channels_last'
            for transform in self.transforms:
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
        if self.seed is not None and idx%self.batch_size==0:
            np.random.seed(self.seed + idx//self.batch_size)
            torch.manual_seed(self.seed+idx//self.batch_size)

        # Create a Compose object with a list of transformations
        # This also converts from NHWC to NCHW
        transform = transforms.Compose([
            transforms.ToImage(),          # Convert the image to a PyTorch tensor
            # transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomRotation(degrees=self.rotation_range),
            transforms.RandomResizedCrop(size=256, scale=self.zoom_range),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

        x = self.X[idx]
        y_semantic_list = self._transform_labels(self.y[idx:idx+1]) 

        x, y_semantic_list = transform(x, y_semantic_list)

        return (x, y_semantic_list)