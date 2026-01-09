import re
import numpy as np
from functools import reduce
from operator import __add__

import torch
import torch.nn as nn
from torch.nn import Conv3d, LazyConv2d, Upsample, BatchNorm2d

class Conv2dSamePadding(nn.LazyConv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        tmp = self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
        return tmp

class add_components(nn.Module):
    def __init__(self, path1, path2):
        super(add_components, self).__init__()
        self.path_li = nn.ModuleList([path1, path2])
    
    def forward(self, input):
        return torch.add(self.path_li[0](input), self.path_li[1](input))

def create_pyramid_level(backbone_input,
                         upsamplelike_input=None,
                         addition_input=None,
                         upsample_type='upsamplelike',
                         level=5,
                         ndim=2,
                         lite=False,
                         interpolation='bilinear',
                         feature_size=256,
                         z_axis_convolutions=False):
    """Create a pyramid layer from a particular backbone input layer.

    Args:
        backbone_input (torch.nn.Module): Backbone layer to use to
            create they pyramid layer.
        upsamplelike_input (tensor): Optional input to use
            as a template for shape to upsample to.
        addition_input (torch.nn.Module): Optional layer to add to
            pyramid layer after convolution and upsampling.
        upsample_type (str): Choice of upsampling methods
            from ``['upsamplelike','upsampling2d','upsampling3d']``.
        level (int): Level to use in layer names.
        feature_size (int): Number of filters for the convolutional layer.
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        lite (bool): Whether to use depthwise conv instead of regular conv for
            feature pyramid construction
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Returns:
        tuple: Pyramid layer after processing, upsampled pyramid layer

    Raises:
        ValueError: ``ndim`` is not 2 or 3
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``
    """

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError(f'Interpolation mode "{interpolation}" not supported. '
                         f'Choose from {list(acceptable_interpolation)}.')

    backbone_input = [backbone_input]
    
    # Apply 1x1 conv to backbone layer
    backbone_input.append(LazyConv2d(feature_size, (1, 1), stride=(1, 1),
                        padding='same'))
    pyramid = backbone_input
    pyramid = nn.Sequential(*pyramid)

    
    # Add and then 3x3 conv
    if addition_input is not None:
        temp_pyramid = pyramid
        addition_input = nn.Sequential(*addition_input)
        pyramid = [add_components(temp_pyramid, addition_input)]
        pyramid = nn.Sequential(*pyramid)


    upsampling = Upsample
    size = (2, 2)
    upsampling_kwargs = {
        'scale_factor': size,
        'mode': interpolation
    }


    temp_li = [pyramid]
    temp_li.append(upsampling(**upsampling_kwargs))
    pyramid_upsample = nn.Sequential(*temp_li)
    
    temp_li = [pyramid]
    temp_li.append(LazyConv2d(feature_size, (3, 3), stride=(1, 1),
                            padding='same'))
    pyramid_final = nn.Sequential(*temp_li)
    
    return pyramid_final, pyramid_upsample

def __create_pyramid_features(backbone_dict,
                              ndim=2,
                              feature_size=256,
                              include_final_layers=True,
                              lite=False,
                              upsample_type='upsamplelike',
                              interpolation='bilinear',
                              z_axis_convolutions=False):
    """Creates the FPN layers on top of the backbone features.

    Args:
        backbone_dict (dictionary): A dictionary of the backbone layers, with
            the names as keys, e.g. ``{'C0': C0, 'C1': C1, 'C2': C2, ...}``
        feature_size (int): The feature size to use for
            the resulting feature levels.
        include_final_layers (bool): Add two coarser pyramid levels
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        lite (bool): Whether to use depthwise conv instead of regular conv for
            feature pyramid construction
        upsample_type (str): Choice of upsampling methods
            from ``['upsamplelike','upsamling2d','upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Returns:
        dict: The feature pyramid names and levels,
        e.g. ``{'P3': P3, 'P4': P4, ...}``
        Each backbone layer gets a pyramid level, and two additional levels
        are added, e.g. ``[C3, C4, C5]`` --> ``[P3, P4, P5, P6, P7]``

    Raises:
        ValueError: ``ndim`` is not 2 or 3
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``
    """

    # Get names of the backbone levels and place in ascending order
    backbone_names = get_sorted_keys(backbone_dict)
    backbone_features = [backbone_dict[name] for name in backbone_names]

    pyramid_names = []
    pyramid_finals = []
    pyramid_upsamples = []

    # Reverse lists to put them in descending order
    backbone_names.reverse()
    backbone_features.reverse()

    for i, N in enumerate(backbone_names):
        level = int(re.findall(r'\d+', N)[0])
        pyramid_names.append(f'P{level}')
        backbone_input = backbone_features[i]

        # Don't add for the bottom of the pyramid
        if i == 0:
            if len(backbone_features) > 1:
                upsamplelike_input = backbone_features[i + 1]
            else:
                upsamplelike_input = None
            addition_input = None

        # Don't upsample for the top of the pyramid
        elif i == len(backbone_names) - 1:
            upsamplelike_input = None
            addition_input = pyramid_upsamples[-1]

        # Otherwise, add and upsample
        else:
            upsamplelike_input = backbone_features[i + 1]
            addition_input = pyramid_upsamples[-1]

        pf, pu = create_pyramid_level(backbone_input,
                                      upsamplelike_input=upsamplelike_input,
                                      addition_input=addition_input,
                                      upsample_type=upsample_type,
                                      level=level,
                                      ndim=ndim,
                                      lite=lite,
                                      interpolation=interpolation,
                                      z_axis_convolutions=z_axis_convolutions)
        pyramid_finals.append(pf)
        pyramid_upsamples.append(pu)

    # Add the final two pyramid layers
    if include_final_layers:
        # "Second to last pyramid layer is obtained via a
        # 3x3 stride-2 conv on the coarsest backbone"
        N = backbone_names[0]
        F = backbone_features[0]
        level = int(re.findall(r'\d+', N)[0]) + 1
        P_minus_2_name = f'P{level}'

        temp_F = [F]
        temp_F.append(Conv2dSamePadding(feature_size, kernel_size=(3, 3),
                            stride=(2, 2)))
        P_minus_2 = nn.Sequential(*temp_F)


        pyramid_names.insert(0, P_minus_2_name)
        pyramid_finals.insert(0, P_minus_2)

        # "Last pyramid layer is computed by applying ReLU
        # followed by a 3x3 stride-2 conv on second to last layer"
        level = int(re.findall(r'\d+', N)[0]) + 2
        P_minus_1_name = f'P{level}'

        P_minus_1 = [P_minus_2]
        P_minus_1.append(torch.nn.ReLU())
        
        P_minus_1.append(Conv2dSamePadding(feature_size, kernel_size=(3, 3),
                            stride=(2, 2)))
        P_minus_1 = nn.Sequential(*P_minus_1)


        pyramid_names.insert(0, P_minus_1_name)
        pyramid_finals.insert(0, P_minus_1)

    pyramid_dict = dict(zip(pyramid_names, pyramid_finals))

    return pyramid_dict

def semantic_upsample(x,
                      n_upsample,
                      target=None,
                      n_filters=64,
                      ndim=2,
                      semantic_id=0,
                      upsample_type='upsamplelike',
                      interpolation='bilinear'):
    """Performs iterative rounds of 2x upsampling and
    convolutions with a 3x3 filter to remove aliasing effects.

    Args:
        x (tensor): The input tensor to be upsampled.
        n_upsample (int): The number of 2x upsamplings.
        target (tensor): An optional tensor with the target shape.
        n_filters (int): The number of filters for
            the 3x3 convolution.
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        semantic_id (int): ID of the semantic head.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Raises:
        ValueError: ``ndim`` is not 2 or 3.
        ValueError: ``interpolation`` not in ``['bilinear', 'nearest']``.
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``.
        ValueError: ``target`` is ``None`` and
            ``upsample_type`` is ``'upsamplelike'``

    Returns:
        tensor: The upsampled tensor.
    """

    # Check input to upsample_type

    conv = LazyConv2d
    conv_kernel = (3, 3)

    upsampling = Upsample
    size = (2, 2)

    temp = [x]
    if n_upsample > 0:
        for i in range(n_upsample):
            temp.append(conv(n_filters, conv_kernel, stride=1, padding='same'))

            # Define kwargs for upsampling layer
            upsample_name = f'upsampling_{i}_semantic_upsample_{semantic_id}'

            if upsample_type == 'upsamplelike':
                assert(False)
                if i == n_upsample - 1 and target is not None:
                    x = UpsampleLike(name=upsample_name)([x, target])
            else:
                upsampling_kwargs = {
                    'scale_factor': size,
                    'mode': interpolation
                }

                temp.append(upsampling(**upsampling_kwargs))
        x = nn.Sequential(*temp)


    return x

def __create_semantic_head(pyramid_dict,
                           input_target=None,
                           n_classes=3,
                           n_filters=128,
                           n_dense=128,
                           semantic_id=0,
                           ndim=2,
                           include_top=True,
                           target_level=2,
                           upsample_type='upsamplelike',
                           interpolation='bilinear',
                           **kwargs):
    """Creates a semantic head from a feature pyramid network.

    Args:
        pyramid_dict (dict): Dictionary of pyramid names and features.
        input_target (tensor): Optional tensor with the input image.
        n_classes (int): The number of classes to be predicted.
        n_filters (int): The number of convolutional filters.
        n_dense (int): Number of dense filters.
        semantic_id (int): ID of the semantic head.
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        include_top (bool): Whether to include the final layer of the model
        target_level (int): The level we need to reach. Performs
            2x upsampling until we're at the target level.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Raises:
        ValueError: ``ndim`` must be 2 or 3
        ValueError: ``interpolation`` not in ``['bilinear', 'nearest']``
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``

    Returns:
        torch.nn.Sequential: The semantic segmentation head
    """

    conv = LazyConv2d
    conv_kernel = (1,1)
    channel_axis = 1

    if n_classes == 1:
        include_top = False

    # Get pyramid names and features into list form
    pyramid_names = get_sorted_keys(pyramid_dict)
    pyramid_features = [pyramid_dict[name] for name in pyramid_names]

    # Reverse pyramid names and features
    pyramid_names.reverse()
    pyramid_features.reverse()

    semantic_sum = pyramid_features[-1]

    n_upsample = target_level
    x = semantic_upsample(semantic_sum, n_upsample,
                          # n_filters=n_filters,  # TODO: uncomment and retrain
                          target=input_target, ndim=ndim,
                          upsample_type=upsample_type, semantic_id=semantic_id,
                          interpolation=interpolation)
    
    temp = [x]
    
    # Apply conv in place of previous tensor product
    temp.append(conv(n_dense, conv_kernel, stride=1, padding='same'))

    
    temp.append(BatchNorm2d(n_dense))

    temp.append(torch.nn.ReLU())

    # Apply conv and softmax layer
    temp.append(conv(n_classes, conv_kernel, stride=1, padding='same'))
    
    if include_top:
        temp.append(torch.nn.Softmax(dim=channel_axis))
    else:
        temp.append(torch.nn.ReLU())
        
    x = nn.Sequential(*temp)
    return x

def get_sorted_keys(dict_to_sort):
    """Gets the keys from a dict and sorts them in ascending order.
    Assumes keys are of the form ``Ni``, where ``N`` is a letter and ``i``
    is an integer.

    Args:
        dict_to_sort (dict): dict whose keys need sorting

    Returns:
        list: list of sorted keys from ``dict_to_sort``
    """
    sorted_keys = list(dict_to_sort.keys())
    sorted_keys.sort(key=lambda x: int(x[1:]))
    return sorted_keys