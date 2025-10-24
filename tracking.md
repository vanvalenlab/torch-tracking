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
    2. The model is composed by the GNNTrackingModel module, which does the following:
        1. During the `__init__` stage, it uses the inputs to make the expected shapes
        2. Model architecture for training branch is as follows:
            1. Inputs:
                - appearances
                - morphologies
                - centroids
                - adjacency matrices
            2. Reshape inputs using `tf.reshape` and store this in a `Lambda` layer beneath the inputs.
            3. These inputs are passed into the `get_neighborhood_encoder` function.
                1. The `appearance_encoder` is composed of:
                    1. Inputs ->
                    2. TimeDistributed layer (keras) -> ImageNormalization2D layer (deepcell) ->
                    3. Nested 3D convolution layers spanning log base 2 indices of appearance shape. ->
                        4. Normalization layer ->
                        5. Activation layer (ReLU) ->
                        6. MaxPool3D layer ->
                    4. A lambda layer (squeeze across axes 2 and 3) ->
                    5. A dense layer of shape `encoder_dim` ->
                    6. A normalization layer ->
                    7. Activation layer (ReLU) -> returned as an output `Model(inputs=inputs, outputs=x)` object.
                2. The `morphology_encoder` is composed of:
                    1. Inputs ->
                    2. Dense layer of shape `encoder_dim` ->
                    3. Normalization layer ->
                    4. Activation (ReLU) -> returned as an output `Model(inputs=inputs, outputs=x)` object.
                3. The `centroid_encoder` is composed of:
                    1. Inputs ->
                    2. Dense layer of shape `encoder_dim` ->
                    3. Normalization layer ->
                    4. Activation (ReLU) -> returned as an output `Model(inputs=inputs, outputs=x)` object.
                4. Adjacency matrix is kept as-is.
                5. Merging features into `Concatenate` object ->
                6. Dense object the size of `n_filters` ->
                7. Normalization layer ->
                8. Activation layer (ReLU)
                9. Construction of GNN convolution using the Spektral GCNConv GCSConv GATConv, depending on the input information. Graph layer depth is 3.
                10. For each layer depth, pipes in `node_features` and adjacency amtrix into the graph layer ->
                11. Normalization layer ->
                12. Activation layer (ReLU) ->
                13. Concatenating appearance features, morphology features, and node features ->
                14. Dense layer ->
                15. Normalization layer ->
                16. Activation layer -> Returns Model of inputs (app, mo, ce, and adj encoders) and outputs (node_features).
            4. Embeddings are "unmerged".
                1. Inputs ->
                2. Unmerge (deepcell layers) -> returns model
            5. Centroids are "unmerged".
                1. Inputs ->
                2. Unmerge (deepcell layers) -> returns model
            6. Lambda extracting current and future embedding
            7. Merge the current embeddings with TemporalMerge layer from deepcell
                1. Inputs ->
                2. TemporalMerge ->
            8. Compare the current and future embeddings using the Comparison layer from deepcell ->
            9. Convert raw position information to deltas between positions in the current frame using Lambda ->
            10. Pad the deltas with a constant matrix ->
            11. Find deltas across frames between current and future using Lambda ->
            12. Subtract the centroid deltas using Subtract() ->
            13. Activation function for both current and future deltas ->
            14. Encode deltas:
                1. Inputs for current frame ->
                2. Inputs across frames ->
                3. Dense layer + normalization + activation ->
                4. X0 (same frame) and X1 (across frames) are fed into these separately
                5. Delta encoder and delta across frames encoder are created from X0 and X1, respectively.
            15.

