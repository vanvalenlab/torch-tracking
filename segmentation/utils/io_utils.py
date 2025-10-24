import os
import numpy as np

def load_data(filepath):
    """Load train, val, and test data"""
    X_train, y_train = _load_npz(os.path.join(filepath, "train.npz"))

    X_val, y_val = _load_npz(os.path.join(filepath, "val_256x256.npz"))

    return (X_train, y_train), (X_val, y_val)

def _load_npz(filepath):
    """Load a npz file"""
    data = np.load(filepath)
    X = data["X"]
    y = data["y"]

    print(
        "Loaded {}: X.shape: {}, y.shape {}".format(
            os.path.basename(filepath), X.shape, y.shape
        )
    )

    return X, y


def update_data_split(source, data_dir):
    """Match the nuclear segmentation training data split to the tracking split"""
    # Load all data splits
    X, y, meta, original_split = [], [], [], []
    for split in {"train", "test", "val"}:
        with np.load(os.path.join(data_dir, f"{split}.npz"), allow_pickle=True) as data:
            X.append(data["X"])
            y.append(data["y"])
            meta.append(data["meta"][1:])
            original_split.append([split] * data["X"].shape[0])

    X = np.concatenate(X)
    y = np.concatenate(y)
    meta = np.concatenate(meta)
    original_split = np.concatenate(original_split)

    # Check for missed data
    all_source = np.concatenate(list(source.values()))

    missing = []
    for f in np.unique(meta[:, 0]):
        if f not in all_source[:, 0]:
            missing.append(f)

    data = {}
    for split in {"train", "test", "val"}:
        data[split] = {"X": [], "y": []}
        for src in source[split][:, 0]:
            data[split]["X"].append(X[meta[:, 0] == src])
            data[split]["y"].append(y[meta[:, 0] == src])

    for f in missing:
        # Look up original split
        split = original_split[meta[:, 0] == f][0]
        data[split]["X"].append(X[meta[:, 0] == f])
        data[split]["y"].append(y[meta[:, 0] == f])

    for d in data.values():
        d["X"] = np.concatenate(d["X"])
        d["y"] = np.concatenate(d["y"])

    return data