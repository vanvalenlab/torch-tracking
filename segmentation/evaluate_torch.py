import sys
sys.path.append('utils/')

import torch
from collections import OrderedDict

from model import create_prediction_model
from evaluate_utils import evaluate
from dnn import DNN

import matplotlib.pyplot as plt
import numpy as np

import typer
import yaml
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from typing_extensions import Annotated
app = typer.Typer(pretty_exceptions_show_locals=False)

def load_npz(data_dir, splits):

    data = {}

    for split in splits:
        data[split] = np.load(f"{data_dir}/{split}.npz")

    return data

def create_overlays(x, gt, pred):
    x = np.squeeze(x)
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

    # Rescale raw data
    percentiles = np.percentile(x[np.nonzero(x)], [5, 95])
    raw = rescale_intensity(
        x, in_range=(percentiles[0], percentiles[1]), out_range="float32"
    )

    # Overlay gt on raw
    gt_overlay = label2rgb(gt, image=raw, bg_label=0)

    # Overlay pred on raw
    pred_overlay = label2rgb(pred, image=raw, bg_label=0)

    return gt_overlay, pred_overlay

@app.command()
def main(
    model_path: Annotated[
        str, typer.Option(help="Path to the trained models")
    ] = "data/saved_model_best_dict.pth",
    metrics_path: Annotated[
        str, typer.Option(help="Destination of evaluation metrics")
    ] = "evaluate-metrics.yaml",
    data_path: Annotated[
        str, typer.Option(help="Path to the training data")
    ] = None,
    backbone: Annotated[
        str, typer.Option(help="Path to the training data")
    ] = 'resnet50'
):
    data = load_npz(data_path, ['test'])
    X_test = data['test']["X"]
    y_test = data['test']["y"]

    # Load model and application
    model = create_prediction_model(
        input_shape=(1,256,256), 
        backbone=backbone, 
        pyramid_levels=("P1","P2", "P3", "P4", "P5", "P6", "P7"))
    
    device = torch.device('cuda:6')

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    app = DNN(model=model, device=device)

    # evaluate the model
    # TODO: evaluate based on experiment data type
    preds = app.predict(X_test)
    metrics = evaluate(preds, y_test)

    all_metrics = {
        "inference": OrderedDict(sorted(metrics.items())),
    }

    # save a metadata.yaml file in the saved model directory
    with open(metrics_path+'evaluate-metrics.yaml', "w") as f:
        yaml.dump(all_metrics, f)

    # Plot sample predictions
    n = 10
    # Configure plot
    _, ax = plt.subplots(n, 2, figsize=(20, 10 * n))
    ax[0, 0].set_title("Ground Truth")
    ax[0, 1].set_title("Prediction")
    plt.tight_layout()

    for j, i in enumerate(np.random.randint(X_test.shape[0], size=(n,))):
        gt, pred = create_overlays(
            X_test[i : i + 1], y_test[i : i + 1], app.predict(X_test[i : i + 1])
        )
        ax[j, 0].imshow(gt)
        ax[j, 0].axis("off")
        ax[j, 1].imshow(pred)
        ax[j, 1].axis("off")

    plt.savefig(metrics_path + 'sample_predictions.png')

if __name__ == "__main__":
    app()
