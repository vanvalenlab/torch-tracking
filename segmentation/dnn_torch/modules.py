import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights

from collections import OrderedDict

class BackboneNetwork(nn.Module):

    def __init__(self, backbone='efficientnetv2bl', use_imagenet=True):
        super().__init__()
        
        self.use_imagenet = use_imagenet
        self.backbone=backbone
        self._extract_backbone()

    def _extract_backbone(self):

        _backbone = str(self.backbone).lower()
            
        if _backbone == 'efficientnetv2bl':
            if self.use_imagenet:
                model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            else:
                model = efficientnet_v2_l()
            
            # EfficientNetV2-L features
            
            self.backbone = nn.ModuleDict({
                'C1': model.features[0:1],      # Stage 0
                'C2': model.features[1:3],      # Stages 1-2
                'C3': model.features[3:4],      # Stage 3
                'C4': model.features[4:5],      # Stage 4
                'C5': model.features[5:7],      # Stages 5-6
            })
        
        else:
            raise ValueError("Unrecognized backbone.")
        
        
    def forward(self, x):
        """Extract backbone features sequentially through stages."""
        backbone_features = {}
        
        current = x
        for level_name in self.backbone.keys():
            current = self.backbone[level_name](current)
            backbone_features[level_name] = current
        
        return backbone_features

class FeaturePyramidNetwork(nn.Module):

    def __init__(self, levels = ['P3','P4','P5'], feature_size=256, interpolation='bilinear'):

        # n_levels should be the length of the pyramid levels list
        # or better yet, just put in the levels parameter

        super().__init__()
        self.levels = nn.ModuleDict()
        self.level_list = levels

        for i, curr_level in enumerate(levels):
            has_addition = (i > 0) # All except deepest levels are added

            level = PyramidLevel(
                feature_size=feature_size, 
                has_addition=has_addition, 
                interpolation=interpolation
                )
            
            self.levels[curr_level] = level

    def forward(self, backbone_features):

        # backbone features have five elements, but we only care about
        # the top 3 (C5, C4, and C3 in that order)

        pyramid_outputs = {}
        from_above = None

        for pyr_level in reversed(self.level_list):
            backbone_level = pyr_level.replace('P','C')
            output, upsampled = self.levels[pyr_level](backbone_features[backbone_level], from_above)

            pyramid_outputs[pyr_level] = output
            from_above = upsampled
        
        return pyramid_outputs

class PyramidLevel(nn.Module):

    def __init__(self, feature_size=256, has_addition=False, interpolation='bilinear'):
        super().__init__()

        self.lateral_conv = nn.LazyConv2d(feature_size, kernel_size=1, stride=1)
        self.has_addition = has_addition
        self.upsample = nn.Upsample(scale_factor=2, mode=interpolation)
        self.smooth_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

    def forward(self, x, from_above = None):

        """
        Args:
            x: backbone feature at this level
            from_above: optional feature from pyramid level above
        """

        # Lateral connection from backbone feature to pyramid
        lateral = self.lateral_conv(x)

        # Add if we have a top-down connection
        if self.has_addition and from_above is not None:
            lateral = lateral + from_above

        # Smooth with a 3x3 conv
        output = self.smooth_conv(lateral)

        # Upsample for next level down
        upsampled = self.upsample(output)

        return output, upsampled
    

class SemanticHead(nn.Module):
    """Semantic segmentation head that upsamples pyramid features."""
    
    def __init__(self, in_channels=256, n_classes=2, 
                 n_upsample=2, n_dense=128, interpolation='bilinear'):
        super().__init__()
        
        # Upsampling path with conv blocks
        upsample_blocks = []
        for i in range(n_upsample):
            upsample_blocks.extend([
                nn.Conv2d(in_channels, in_channels, 
                         kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode=interpolation)
            ])
        self.upsample_path = nn.Sequential(*upsample_blocks)
        
        # Dense processing
        self.dense_conv = nn.Conv2d(in_channels, n_dense, kernel_size=1)
        self.bn = nn.BatchNorm2d(n_dense)
        self.relu = nn.ReLU()
        
        # Output head
        self.output_conv = nn.Conv2d(n_dense, n_classes, kernel_size=1)
        
        # Final activation depends on n_classes
        if n_classes > 1:
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = nn.ReLU()
    
    def forward(self, pyramid_feature):
        """
        Args:
            pyramid_feature: feature from FPN (typically P3)
        
        Returns:
            segmentation output at input resolution
        """
        # Upsample to input resolution
        x = self.upsample_path(pyramid_feature)
        
        # Dense processing
        x = self.dense_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Output
        x = self.output_conv(x)
        x = self.final_activation(x)
        
        return x

class Location2D(torch.nn.Module):
    """Location Layer for 2D cartesian coordinate locations.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        input_shape = inputs.size()
        input_device = inputs.device
        input_dtype = inputs.dtype
        
        # shapes of (, height) and (, width)
        x = torch.arange(0, input_shape[2], dtype=input_dtype, device=input_device)
        y = torch.arange(0, input_shape[3], dtype=input_dtype, device=input_device)

        # Detach after normalization to prevent gradient flow
        x = (x / torch.max(x)).detach()
        y = (y / torch.max(y)).detach()

        # makes the mesh
        loc_x, loc_y = torch.meshgrid(x, y, indexing='ij')

        # (2, H, W)
        loc = torch.stack([loc_x, loc_y], dim=0)

        # unsqueeze to add batch dimension and permute
        location = torch.unsqueeze(loc, dim=0)
        location = torch.permute(location, dims=[0, 2, 3, 1])

        # Detatch and tile to add back batch dimension (they are all the same)
        location = location.detach()
        location = torch.tile(location, [input_shape[0], 1, 1, 1])

        location = torch.permute(location, dims=[0, 3, 1, 2])

        return location
    
if __name__ == '__main__':
    print()

    config = {
        'model_path': "data/segmentation/model/",
        'data_path': 'data/DynamicNuclearNet-segmentation-v1_0',
        'run_info': 'data/segmentation/logs/',
        'epochs': 16,
        'seed': 0,
        'min_objects': 1,
        'zoom_min': 0.75,
        'batch_size': 12,
        'backbone': 'efficientnetv2bl',
        'crop_size': 256,
        'lr': 1e-4,
        'outer_erosion_width': 1,
        'inner_distance_alpha': 'auto',
        'inner_distance_beta': 1,
        'inner_erosion_width': 0,
        'pyramid_levels': ['P3','P4','P5'],
        'backbone_levels': ['C1','C2','C3', 'C4','C5'],
        'num_workers': 16
    }

    backbone_levels=config['backbone_levels']
    pyr_levels = config['pyramid_levels']

    ## Test Location 2D 
    # expecting from an input of shape (B, 1, H, W)
    # output of (B, 2, H, W)

    location = Location2D()
    test = torch.rand(8, 1, 256, 256)
    loc = location(test)

    ## Test Semantic Head
    # expecting from an input of shape (B, 1, H, W)
    # output of (B, 2, H, W)

    ## Test BackboneNetwork
    bbnet = BackboneNetwork()
    test = torch.rand(8, 3, 256, 256)
    bb_out = bbnet(test)

    # From outputs of BackboneNetwork, test FPN

    fpn = FeaturePyramidNetwork(levels=pyr_levels)
    fpn_out = fpn(bb_out)

    head = SemanticHead()
    test = torch.rand(8, 256, 64, 64)
    output = head(fpn_out['P3'])
    print(output.shape)
    