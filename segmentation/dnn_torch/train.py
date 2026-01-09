import torch
import os
torch.set_num_threads(24)
import datetime

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from model import PanopticNet
from loss import SemanticLoss
from loaders import create_data_loaders

import zarr

def train_torch(
        dataloader,
        valloader,
        crop_size=256,
        backbone="resnet50",
        lr=1e-4,
        epochs=8,
        pyramid_levels=['P3','P4','P5'],
        backbone_levels=['C1','C2','C3', 'C4','C5'],
        save_path_prefix = "data/saved_model",
        writer=None,
        n_semantic_classes = [1,1,2]
    ):

    model = PanopticNet(
        crop_size=crop_size,
        backbone=backbone,
        pyramid_levels=pyramid_levels,
        backbone_levels=backbone_levels,
        n_semantic_classes=n_semantic_classes,
    )

    # TODO: make embedded in dataloader instead of hard-coding it in here
    semantic_type = ['cont','cont','disc','disc']

    loss = SemanticLoss(n_semantic_classes=n_semantic_classes, semantic_type=semantic_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=5,)

    epoch_number = 0

    best_vloss = 1000000
    patience_count = 0

    model = model.to(device)

    for epoch in range(epochs):

        print('EPOCH {}:'.format(epoch_number + 1))
        print("TRAIN")

        model.train()

        running_loss_avg = 0.
        count = 0

        for batch_idx, batch in enumerate(tqdm(dataloader)):

            image, labels = batch
            count += 1
            
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            curr_loss = loss(outputs, labels)   

            count += 1

            curr_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001, error_if_nonfinite=True)
        
            optimizer.step()

            running_loss_avg += curr_loss.item()


        avg_loss = running_loss_avg/count
        writer.add_scalar('avg_loss/train', avg_loss, epoch)

        print("VAL")

        vcount = 0
        running_vloss_avg = 0.

        model.eval()
        
        with torch.no_grad():
            
            for batch_idx, batch in enumerate(tqdm(valloader)):

                image, labels = batch
                vcount += 1
                
                image = image.to(device)
                labels = labels.to(device)

                voutputs = model(image)
                curr_vloss = loss(voutputs, labels)
                    
                running_vloss_avg += curr_vloss

        avg_vloss = running_vloss_avg/vcount
        writer.add_scalar('avg_loss/val', avg_vloss, epoch)

        decay_scheduler.step()
        plateau_scheduler.step(avg_vloss)
        
        
        # Save model periodically
        if (epoch+1)%10==0:
            epoch_path_prefix = save_path_prefix + "/model_epoch" + str(epoch+1) + "_dict.pth"
            
            torch.save(model.state_dict(), epoch_path_prefix)
        
        if avg_vloss<best_vloss:
            best_vloss = avg_vloss
            dict_save_path = save_path_prefix + "/saved_model_best_dict.pth"
            
            torch.save(model.state_dict(), dict_save_path)
            patience_count = 0
            print("Saved new best model.")
        else:
            patience_count += 1
            
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        epoch_number += 1

        if patience_count >= 10:
            break

    return model

def main():

    config = {
        'model_path': "data/segmentation/model/",
        'data_path': 'data/DynamicNuclearNet-segmentation-v1_0',
        'run_info': 'data/segmentation/logs/',
        'epochs': 16,
        'seed': 0,
        'min_objects': 1,
        'zoom_min': 0.75,
        'batch_size': 24,
        'backbone': 'efficientnetv2bl',
        'crop_size': 256,
        'lr': 1e-4,
        'outer_erosion_width': 1,
        'inner_distance_alpha': 'auto',
        'inner_distance_beta': 1,
        'inner_erosion_width': 0,
        'pyramid_levels': ['P3','P4','P5'],
        'backbone_levels': ['C1','C2','C3', 'C4','C5'],
        'num_workers': 24
    }

    curr_time = f"{datetime.datetime.now():%Y%m%d%H%M%S}"

    z_train = zarr.open(f"{config['data_path']}/train.zarr")
    z_val = zarr.open(f"{config['data_path']}/val.zarr")

    run_info = config['run_info'] + '/' + curr_time
    model_path = config['model_path'] + '/' + curr_time
    
    if not os.path.isdir(run_info):
        os.makedirs(run_info, exist_ok=True)
    if not os.path.isdir(model_path):
        os.makedirs(model_path, exist_ok=True)

    writer = SummaryWriter(run_info)

    # Set up data generators with updated data
    train_data, val_data = create_data_loaders(
        z_train,
        z_val,
        crop_size=config['crop_size'],
        zoom_min=config['zoom_min'],
        batch_size=config['batch_size'],
        data_format='channels_last',
        outer_erosion_width=config['outer_erosion_width'],
        inner_distance_alpha=config['inner_distance_alpha'],
        inner_distance_beta=config['inner_distance_beta'],
        inner_erosion_width=config['inner_erosion_width'],
        preprocess=False,
        num_workers=config['num_workers']
    )

    # train the model
    model = train_torch(
        train_data,
        val_data,
        crop_size=config['crop_size'],
        backbone=config['backbone'],
        lr=config['lr'],
        epochs=config['epochs'],
        pyramid_levels=config['pyramid_levels'],
        save_path_prefix=model_path,
        writer=writer
    )

    writer.close()
    torch.save(model.state_dict(), config['model_path']+'last_model_dict.pth')

if __name__ == "__main__":
    main()
