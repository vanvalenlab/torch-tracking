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
        4. The TFRecordDataset object has built in functions like `.shuffle` for shuffling the data, `.map` for mapping the augmentations, and `.batch` for preparing batches. These have similarities to the `DataLoader` functions in PyTorch.
        5. Loading the validation dataset in the similar manner.
3. Model architecture
    1. Step 1: Neighborhood embedding ${\vec{N}}=[\vec{N_1}, \vec{N_2},...\vec{N_i}]$:
        1. Appearance encoder ${\vec{A_i}}$:
            1. Standard crop of each cell in the frame
            2. Dense -> normalization -> activation
            3. Returns appearance embedding
        2. Centroid encoder ${\vec{C_i}}$:
            1. X and Y coordinates of the centroid of each cell in the frame
            2. Dense -> normalization -> activation
            3. Returns centroid embedding
        3. Morphology encoder ${\vec{M_i}}$:
            1. area, perimeter, and eccentricity of each cell in the frame
            2. Dense -> normalization -> activation
            3. Returns morphology embedding
        4. Neighborhood encoder ${\vec{N_i}}$:
            1. Inputs: appearance, centroid, and morphology embeddings, adjacency matrix
            2. Inputs concatenated except adjacency matrix
            3. Dense -> normalization -> activation
            4. Creates a graph layer with adjacency matrix and node features from embeddings above.
            5. Normalization -> activation
            6. Repeat 4-5 twice more.
            7. Concat appearance, morphology, and node features (for some reason? why do we need to include the appearance and morphology if they're already embedded)
            8. Dense -> normalization -> activation
        9. Return node features and centroids
    2. Step 2: Tracking inference ${P}$ (adjacency/probability matrix):
        1. Reshape the embeddings to add back the time dimension:
            1. Unmerge embeddings by reshaping the embeddings and adding time axis with `tf.reshape()`. Output shape is ${(8, 32, 64)}$
        2. Reshape the centroids to add back the time dimension:
            1. Unmerge centroids by reshaping the centroids and adding time axis with `tf.reshape()`. Output shape is ${(8, N_{cells}, X_i, Y_i)}$.
        3. Merge current (-7:0) embeddings in a long short-term memory layer.
            1. Reshape the temporal axis away from the inputs (-1, cells, 64)
            2. Pass through an LSTM layer
            3. Add back the time axis and return (-1, cells, time, 64)
        4. Compare current LSTM embeddings and future embeddings.
            1. Add a dimension at dimension 3 (to 5 dimensions) to both `x_current` and `x_future`.
            2. Tile both to match across dimension 3
            3. Return concatenated `x_current` and `x_future`
        5. Encoding deltas (both local and across frames):
            1. Convert centroids to deltas of cells relative to every other cell
            2. Convert centroids to deltas of cells relative to every other cell across time
            3. Activation (absolute value) -> Dense -> Activation (ReLU)
            4. Merge deltas across frames to remove time axis
            5. Concatenate current and future delta embeddings
            6. Return ${[[X_{current}, X_{future}], [C_{current}, C_{future}]]}$
        6. Decode tracking:
            1. Concatenate embeddings and deltas
            2. Dense -> normalization -> activation
            3. Dense embedding with 3 classes (different, same, parent-child)
    3. Softmax of three-class embedding to return adjacency matrix of shape ${(N_{cells}, N_{cells})}$, where 0 is different cell, 1 is same cell, 2 is parent-child.
    4. Post-processing (reconstruction of lineages, swapping of track IDs in labeled image to match tracking labels)

## PyTorch Pipeline

### Model



### Training

Initial port from the TensorFlow to PyTorch was done with Claude, which got much of the pipeline wrong. After working through it, the initial training loop works, and has a pretty good training and validation metrics. The model architecture is the same, but the preprocessing and postprocessing steps can be improved.

Converted `.trk` files to `.zarr` files for faster loading and better readability. 

- DataLoader:
    1. `__init__`:
        1. Reads in `.zarr` files for raw nuclear images (`X`), labels (`y`), and lineage information
        2. Corrects lineage information
            - Relabels data to ensure labels are sequential
            - Adjusts labels in lineage information to make sure they match.
        3. Extracts appearance information, morphology, centroids, and adjacency matrices.
        4. Loader builds sample indices for the `__getitem__` function. Since the training data is longer than 8 frames, the sample indices create samples for `batch`, `start_frame` and `end_frame`, which generates ~4000 samples for training.
        5. If the augmentation option is selected, it generates the augmentation pipeline using the built-in `Transforms` function in PyTorch.
    1. `__getitem__`:
        1. Using an `idx`, get the sample indices from the generated samples previously.
        2. Selects item from complied dataset
        3. Transforms it if augmentation pipeline is selected, then converts it to a PyTorch tensor.
