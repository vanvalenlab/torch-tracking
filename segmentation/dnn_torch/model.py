import torch
from torch import nn
from math import log2

from modules import SemanticHead, FeaturePyramidNetwork, Location2D, BackboneNetwork

class PanopticNet(nn.Module):

    '''
    Complete implementation of DynamicNuclearNet, rewritten in PyTorch without Keras syntax.

    This model uses convolution paired to a feature pyramid network to extract features
    from the image. These features are then fed into the semantic heads of the model,
    which infers inner distance transforms, outer distance transforms, and foreground/background
    of each image.

    Args:
        input shape: shape expected from the model -- wil never change and is always 256
    '''

    def __init__(
            self,
            backbone='efficientnetv2bl',
            lr=1e-4,
            feature_size=256,
            crop_size=256,
            use_imagenet=True,
            backbone_levels=['C1','C2','C3', 'C4', 'C5'],
            pyramid_levels = ['P3', 'P4', 'P5'],
            pyramid_shape = [32, 64, 128],
            interpolation='bilinear',
            target_level='P3',
            n_semantic_classes = [1,1,2]
    ):
        
        super().__init__()

        # Store configuration
        
        self.backbone = backbone
        self.lr = lr
        self.use_imagenet = use_imagenet
        self.backbone_levels = backbone_levels
        self.pyramid_levels = pyramid_levels
        self.feature_size = feature_size
        self.interpolation=interpolation
        self.pyramid_shape = pyramid_shape
        self.target_level = target_level
        self.crop_size = crop_size
        self.upsample = int(log2(self.crop_size / self.pyramid_shape[self.pyramid_levels.index(self.target_level)]))
        self.n_semantic_classes = n_semantic_classes

        # Build model

        self.location = Location2D()

        self.bbnetwork = BackboneNetwork(
            backbone=self.backbone, 
            use_imagenet=self.use_imagenet
            )

        self.fpn = FeaturePyramidNetwork(
            levels=self.pyramid_levels,
            feature_size=self.feature_size,
            interpolation=self.interpolation
            )
        
        semantic_heads = [SemanticHead(n_classes=i, n_upsample=self.upsample) for i in self.n_semantic_classes]

        self.semantic_heads = nn.ModuleList(semantic_heads)

    def forward(self, x, format = 'channel_first'):
        # Inputs into training forward should be of shape (B, C, H, W) but may be (B, H, W, C)

        if format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        
        location = self.location(x)

        # Already in (B, 3, 256, 256) shape needed -- no need to conv
        x = torch.cat([x, location], dim=1)

        backbone_features = self.bbnetwork(x)

        pyramid_features = self.fpn(backbone_features)

        predictions = []

        for head in self.semantic_heads:
            predictions.append(
                head(pyramid_features[self.target_level])
            )

        return torch.cat(predictions, dim=1)

if __name__ == '__main__':

    device = 'cuda:0'

    model = PanopticNet(n_semantic_classes=[1,1,1]).to(device)

    test_tensor = torch.rand(8, 256, 256, 1).to(device)

    output = model(test_tensor, format='channels_last')

    for tensor in output:
        print(tensor.shape)