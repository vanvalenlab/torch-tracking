from panoptic import PanopticNet
import torch
import numpy as np
from training_utils import semantic_loss

def create_model(
        input_shape,
        backbone="resnet50",
        lr=1e-4,
        location=True,
        device=None,
        pyramid_levels=("P3", "P4", "P5", "P6", "P7"),):
    
    num_semantic_classes = [1, 1, 2]  # inner distance, pixelwise, inner distance, pixelwise
    model = PanopticNet(backbone=backbone,
        input_shape=input_shape,
        num_semantic_classes=num_semantic_classes,
        backbone_levels=['C1','C2','C3', 'C4', 'C5'],
        pyramid_levels=pyramid_levels,
        location=location,  # should always be true
        include_top=True)
    
    
    print("Model is using", device)

    loss = []
    model = model.to(device)
    for n_classes in num_semantic_classes:
        loss.append(semantic_loss(n_classes))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss, optimizer

def create_prediction_model(
        input_shape,
        backbone="resnet50",
        model_dir=None,
        location=True,
        device=None,
        pyramid_levels=("P3", "P4", "P5", "P6", "P7")
):
    
    num_semantic_classes = [1, 1, 2]  # inner distance, pixelwise, inner distance, pixelwise
    model = PanopticNet(
        weights=model_dir,
        backbone=backbone,
        input_shape=input_shape,
        norm_method=None,
        num_semantic_heads=4,
        num_semantic_classes=num_semantic_classes,
        backbone_levels=['C1','C2','C3', 'C4', 'C5'],
        pyramid_levels=pyramid_levels,
        location=location,  # should always be true
        include_top=True)
    
    print("Model is using", device)

    model = model.to(device)

    return model