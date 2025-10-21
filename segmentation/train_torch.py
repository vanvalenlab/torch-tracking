import sys
sys.path.append('utils/')

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import create_model
from toolbox import histogram_normalization
from loaders import SemanticDataset, CroppingDatasetTorch

import typer

from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_show_locals=False)

def load_npz(data_dir, splits):

    data = {}

    for split in splits:
        data[split] = np.load(f"{data_dir}/{split}.npz")

    return data

def train_one_epoch(model, dataloader, optimizer, losses, device):
    running_loss_avg = 0.
    count = 0

    for (li_inputs, li_labels) in tqdm(dataloader):

        count += 1
        inputs = li_inputs.to(device)
        labels = [l.to(device) for l in li_labels]

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = sum([losses[j](outputs[j], labels[j]) for j in range(len(losses))])            
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001, error_if_nonfinite=True)
    
        optimizer.step()

        running_loss_avg += loss.item()
    return running_loss_avg/count

def create_data_loaders(
    train,
    val,
    crop_size=256,
    min_objects=1,
    seed=0,
    zoom_min=0.75,
    batch_size=16,
    outer_erosion_width=1,
    inner_distance_alpha="auto",
    inner_distance_beta=1,
    inner_erosion_width=0,
):
    
    rotation_range = 180
    shear_range = 0
    zoom_range = (zoom_min, 1/zoom_min)
    horizontal_flip = True
    vertical_flip = True


    transforms = ["inner-distance", "outer-distance", "fgbg"]

    transforms_kwargs = {
        "outer-distance": {"erosion_width": outer_erosion_width},
        "inner-distance": {
            "alpha": inner_distance_alpha,
            "beta": inner_distance_beta,
            "erosion_width": inner_erosion_width,
        },
    }

    print('STARTING PREPROCESS')
    x_train = histogram_normalization(train["X"])
    x_val = histogram_normalization(val["X"])
    print('FINISH PREPROCESS')

    cdt = CroppingDatasetTorch(
        x_train, 
        train["y"], 
        rotation_range, 
        shear_range, 
        zoom_range, 
        horizontal_flip, 
        vertical_flip, 
        crop_size, 
        batch_size=batch_size, 
        transforms=transforms, 
        transforms_kwargs=transforms_kwargs, 
        seed=seed, 
        min_objects=min_objects)
    
    sd = SemanticDataset(
        x_val, 
        val["y"], 
        transforms=transforms, 
        min_objects=min_objects, 
        transforms_kwargs=transforms_kwargs)  
      
    dataloader = DataLoader(cdt, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valloader = DataLoader(sd, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataloader, valloader


def train_torch(dataloader,
    valloader,
    crop_size=256,
    backbone="resnet50",
    lr=1e-4,
    epochs=8,
    pyramid_levels=("P1", "P2", "P3", "P4", "P5", "P6", "P7"),
    save_path_prefix = "data/saved_model"
):

    # torch.cuda.empty_cache()
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print(device)

    model, losses, optimizer = create_model(
        input_shape=(crop_size, crop_size, 1),
        backbone=backbone,
        lr=lr,
        device=device,
        pyramid_levels=pyramid_levels
    )

    loss_tracking = []
    vloss_tracking = []
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=5,)

    epoch_number = 0
    start_epoch = 0

    best_vloss = 1_000_000.
    patience_count = 0

    model = model.to(device)

    save_path_prefix = "data/saved_model"

    for epoch in range(start_epoch, epochs):

        print('EPOCH {}:'.format(epoch_number + 1))
        print("TRAIN")

        model.train()

        avg_loss = train_one_epoch(model, dataloader, optimizer, losses, device)

        count = 0
        running_vloss_avg = 0.
        
        print("VAL")

        model.eval()
        with torch.no_grad():
            for (li_inputs, li_labels) in tqdm(valloader):
                count += 1
                
                vinputs = li_inputs.to(device)
                vlabels = [l.to(device) for l in li_labels]
                voutputs = model(vinputs)
                vloss = sum([losses[j](voutputs[j], vlabels[j]) for j in range(len(losses))])
                    
                running_vloss_avg += vloss
                    
        avg_vloss = running_vloss_avg/count

        decay_scheduler.step()
        plateau_scheduler.step(avg_vloss)
        print(decay_scheduler.get_last_lr())
        
        loss_tracking.append(avg_loss)
        vloss_tracking.append(avg_vloss)
        
        # Save model periodically
        if (epoch+1)%10==0:
            epoch_path_prefix = save_path_prefix + "_epoch" + str(epoch+1)
            dict_save_path = epoch_path_prefix + "_dict.pth"
            
            torch.save(model.state_dict(), dict_save_path)
        
        if avg_vloss<best_vloss:
            best_vloss = avg_vloss
            dict_save_path = save_path_prefix + "_best_dict.pth"
            
            torch.save(model.state_dict(), dict_save_path)
            patience_count = 0
        else:
            patience_count += 1
            
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        epoch_number += 1

        if patience_count >= 10:
            break

    return model

@app.command()
def main_torch(
    model_path: Annotated[
        str, typer.Option(help="Path to save segmentation model")
    ] = "data/models/",
    data_path: Annotated[
        str, typer.Option(help="Directory where training data is located")
    ] = None,
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 16,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    min_objects: Annotated[
        int, typer.Option(help="Minimum number of objects in each training image")
    ] = 1,
    zoom_min: Annotated[
        float, typer.Option(help="Smallest zoom value. Zoom max is inverse of zoom min")
    ] = 0.75,
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 16,
    backbone: Annotated[
        str, typer.Option(help="Backbone of the model")
    ] = "efficientnetv2bl",
    crop_size: Annotated[
        int, typer.Option(help="Size of square patches to train on")
    ] = 256,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-4,
    outer_erosion_width: Annotated[
        int,
        typer.Option(help="Erosion_width paramter for the outer-distance transform"),
    ] = 1,
    inner_distance_alpha: Annotated[
        str, typer.Option(help="Alpha parameter for the inner-distance transform")
    ] = "auto",
    inner_distance_beta: Annotated[
        float, typer.Option(help="Beta parameter for inner distance transform")
    ] = 1,
    inner_erosion_width: Annotated[
        int, typer.Option(help="erosion width for inner distance transform")
    ] = 0,
    pyramid_levels: Annotated[
        str, typer.Option(help="String of pyramid levels")
    ] = "P1-P2-P3-P4-P5-P6-P7",
):

    data = load_npz(data_path, ['train','val'])

    # Set up data generators with updated data
    train_data, val_data = create_data_loaders(
        data["train"],
        data["val"],
        crop_size=crop_size,
        min_objects=min_objects,
        zoom_min=zoom_min,
        seed=seed,
        batch_size=batch_size,
        outer_erosion_width=outer_erosion_width,
        inner_distance_alpha=inner_distance_alpha,
        inner_distance_beta=inner_distance_beta,
        inner_erosion_width=inner_erosion_width,
    )

    # train the model
    model = train_torch(
        train_data,
        val_data,
        crop_size=crop_size,
        backbone=backbone,
        lr=lr,
        epochs=epochs,
        pyramid_levels=pyramid_levels.split("-"),
    )
    
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    app()

