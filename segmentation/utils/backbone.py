import torch.nn as nn
import torchvision
from torchvision.models import resnet50, efficientnet_v2_l
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights

def get_all_children(l, layer_li):
    start = None
    out = []
    for layer_name, layer in l.named_children():
        if start is None:
            start = layer
        else:
            start = nn.Sequential(start, layer)
        if layer_name in layer_li:
            out.append(start)
    return out

def get_backbone(backbone, use_imagenet=False):
    """Retrieve backbones for the construction of feature pyramid networks.

    Args:
        backbone (str): Name of the backbone to be retrieved.
        input_tensor (tensor): The input tensor for the backbone.
            Should have channel dimension of size 3
        use_imagenet (bool): Load pre-trained weights for the backbone
        return_dict (bool): Whether to return a dictionary of backbone layers,
            e.g. ``{'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}``.
            If false, the whole model is returned instead
        kwargs (dict): Keyword dictionary for backbone constructions.
            Relevant keys include ``'include_top'``,
            ``'weights'`` (should be ``None``),
            ``'input_shape'``, and ``'pooling'``.

    Returns:
        (torch.nn.Module, dict)
            if return_dict: 
                Tuple of instantiated backbone and dictionary of 
                torch.nn.Module backbone sub-layers
        torch.nn.Module 
            otherwise: 
                An instantiated backbone


    Raises:
        ValueError: bad backbone name
        ValueError: featurenet backbone with pre-trained imagenet
    """
    _backbone = str(backbone).lower()

    resnet_backbones = {
        'resnet50': resnet50
    }

    efficientnet_v2_backbones = {
        'efficientnetv2bl': torchvision.models.efficientnet_v2_l
    }

    if _backbone in resnet_backbones:
        model_cls = resnet_backbones[_backbone]
        if use_imagenet:
            print("Using ImageNet")
            
            model_cls=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model_cls = resnet50()

        all_layers = get_all_children(model_cls, ["relu", "layer1", "layer2", "layer3", "layer4"])

    elif _backbone in efficientnet_v2_backbones:

        model_cls = efficientnet_v2_backbones[_backbone]
        if use_imagenet:
            print("Using ImageNet")
            model_cls=efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        else:
            model_cls = efficientnet_v2_l()

        all_layers = get_all_children(model_cls.features, ['0','2','3','4','6'])

    
    output_dict = {f'C{i + 1}': j for i, j in enumerate(all_layers)}
    
    return output_dict

