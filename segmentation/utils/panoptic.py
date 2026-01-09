import math
import numpy as np
import re

import torch
import torch.nn as nn
from torch.nn import LazyConv2d

from utils.fpn import __create_pyramid_features
from utils.fpn import __create_semantic_head
from utils.layers import Location2D
from utils.backbone import get_backbone


class PanopticModel(nn.Module):
    
    def __init__(self, preprocessing, semantic_heads):
        super().__init__()
        self.preprocessing = preprocessing
        self.semantic_heads = nn.ModuleList(semantic_heads)
    
    def forward(self, x):
        # Apply preprocessing ONCE
        x_preprocessed = self.preprocessing(x)
        
        # Each semantic head processes the preprocessed input
        # They share the same backbone/pyramid computation internally
        outputs = []
        for head in self.semantic_heads:
            outputs.append(head(x_preprocessed))
        
        return outputs

class concat_components(nn.Module):
    def __init__(self, path1, path2):
        super(concat_components, self).__init__()
        self.path1=path1
        self.path2=path2
    
    def forward(self, input):
        
        return torch.cat([self.path1(input), self.path2(input)], dim=1)

def PanopticNet(backbone,
                input_shape,
                inputs=None,
                backbone_levels=['C1','C2','C3', 'C4', 'C5'],
                pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                create_pyramid_features=__create_pyramid_features,
                create_semantic_head=__create_semantic_head,
                frames_per_batch=1,
                temporal_mode=None,
                num_semantic_classes=[3],
                required_channels=3,
                norm_method=None,
                pooling=None,
                location=True,
                use_imagenet=True,
                lite=False,
                upsample_type='upsampling2d',
                interpolation='bilinear',
                name='panopticnet',
                z_axis_convolutions=False,
                **kwargs):
    """Constructs a Mask-RCNN model using a backbone from
    ``torchvision`` with optional semantic segmentation transforms.

    Args:
        backbone (str): Name of backbone to use.
        input_shape (tuple): The shape of the input data.
        backbone_levels (list): The backbone levels to be used.
            to create the feature pyramid.
        pyramid_levels (list): Pyramid levels to use.
        create_pyramid_features (function): Function to get the pyramid
            features from the backbone.
        create_semantic_head (function): Function to build a semantic head
            submodel.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        temporal_mode: Mode of temporal convolution. Choose from
            ``{'conv','lstm', None}``.
        num_semantic_classes (list or dict): Number of semantic classes
            for each semantic head. If a ``dict``, keys will be used as
            head names and values will be the number of classes.
        norm_method (str): Normalization method to use with the
            :mod:`torch-mesmer.layers.normalization.ImageNormalization2D` layer.
        location (bool): Whether to include a
            :mod:`torch-mesmer.layers.location.Location2D` layer.
        use_imagenet (bool): Whether to load imagenet-based pretrained weights.
        lite (bool): Whether to use a depthwise conv in the feature pyramid
            rather than regular conv.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.
        pooling (str): optional pooling mode for feature extraction
            when ``include_top`` is ``False``.

            - None means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
            - 'avg' means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
            - 'max' means that global max pooling will
              be applied.

        z_axis_convolutions (bool): Whether or not to do convolutions on
            3D data across the z axis.
        required_channels (int): The required number of channels of the
            backbone.  3 is the default for all current backbones.
        kwargs (dict): Other standard inputs for ``retinanet_mask``.

    Raises:
        ValueError: ``temporal_mode`` not 'conv', 'lstm'  or ``None``

    Returns:
        pytorch.nn.Module: Panoptic model with a backbone.
    """
    
    # Using channels first as that's the default for torch
    input_shape = (1, 256, 256)

    conv = LazyConv2d
    conv_kernel = (1, 1)

    if inputs is None:
        temp_input_shape = [1] + list(input_shape)
        inputs = torch.randn(size=temp_input_shape)
            
    norm = nn.Identity()
    loc = nn.Sequential(norm, Location2D())
        
    concat = concat_components(norm, loc)

    # Force the channel size for backbone input to be `required_channels`
    preprocessing = nn.Sequential(concat, conv(required_channels, conv_kernel, stride=1, padding='same'))

    # Set axis as 0 for pytorch's channels_first approach    

    backbone_dict = get_backbone(backbone, use_imagenet=use_imagenet)
                                    

    backbone_dict_reduced = {k: backbone_dict[k] for k in backbone_dict
                             if k in backbone_levels}
    
    ndim = 2 if frames_per_batch == 1 else 3

    pyramid_dict = create_pyramid_features(backbone_dict_reduced,
                                           ndim=ndim,
                                           lite=lite,
                                           interpolation=interpolation,
                                           upsample_type=upsample_type,
                                           z_axis_convolutions=z_axis_convolutions)

    semantic_levels = [int(re.findall(r'\d+', k)[0]) for k in pyramid_dict]
    
    target_level = min(semantic_levels)

    semantic_head_list = []
    if not isinstance(num_semantic_classes, dict):
        num_semantic_classes = {
            k: v for k, v in enumerate(num_semantic_classes)
        }

    for k, v in num_semantic_classes.items():
        semantic_head_list.append(create_semantic_head(
            pyramid_dict, n_classes=v,
            input_target=inputs, target_level=target_level,
            semantic_id=k, ndim=ndim, upsample_type=upsample_type,
            interpolation=interpolation, **kwargs))
        
    model = PanopticModel(preprocessing, semantic_head_list)

    return model