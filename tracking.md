# Torch-ifying CellTracking model
## TensorFlow pipeline
1. Create agumentation functions:
    1. Sample gets a temporal slice from the data.
    2. Rotate applies a random rotation to the feature dictionary (?)
    3. Translate applies a random translation to the feature dictionary (?)
2. Load data:
    1. Uses `get_training_dataset` function from `deepcell.utils.tfrecords_utils` to load a TensorFlow records CSV with no additional arguments.
        1. `get_training_dataset` loads a CSV file using `tf.data.TFRecordDataset` with no additional keyword arguments.
        2. I assume the output of this is in the same shape as the `test.trks` dataset stored in the tracking data folder.
        3 **to do** configure the loading function to bypass the TFRecord and load directly from either JSON or numpy.
        4. The TFRecordDataset object has built in functions like `.shuffle` for shuffling the data, `.map` for mapping the augmentations, and `.batch` for preparing batches. These have similarities to the `DataLoader` functions in PyTorch.
        5. Loading the validation dataset in the similar manner.
3. Create the model:
    1. `create_model` takes in several arguments:
        - max_cells (int): maximum number of cells per movie in dataset
        - strategy: tensorflow training strategy for synchronous training
        - n_layers: number of graph convolution layers
        - graph_layer: either GCS, GCN or GAT. We are using GCS for training.
        - track_length: length of tracks, set to 8
        - lr: learning rate, set to 1e-4
        - n_filters: 64 (unsure what this means)
        - embedding dim: 64
        - encoder_dim: 64
        -norm_layer: 'batch'

