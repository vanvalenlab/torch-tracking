# Torch-ifying DyanmicNuclearNet
## TensorFlow pipeline
1. Load data:
    1. Loading tracking data from `.npz` file.
    2. Update data using `update_data_split` function. `source` is tracking (a dictionary), `data_path` is the path to the segmentation data.
        1. Opens the segmentation data and separates out X, Y, meta, and original_split for each of the train/test/val splits.
        2. Concatenates them together
        3. If the metadata is not in the `source` set, then it's missing and we need to find it in the original split
        4. Create the `data` variable which is returned and populate train/test/val splits for each of the data sources.
        5. Search for missing values that are in the tracking but not segmentation and place it. in the new data file formed.
        6. Concatenate it all together.
        7. Return the data structure.
    3. Set up TensorFlow data generator objects (similar to DataLoader object in torch) using `create_data_generators` function:
        1. Set data augmentation parameters (`zoom_max` and `zoom_min`).
        2. Histogram normalize training and validation data using the built-in `histogram_noramlization` function in `deepcell`.
        3. Construct a training data generator using DeepCell's `CroppingDataGenerator` object.
        4. Construct a validation data generator using DeepCell's `CroppingDataGenerator` object without augmentations (i.e. only `crop_size` is specified).
        5. Generate bespoke transforms using names and keyword arguments as follows:
            1. `inner-distance`: alpha, beta, and inner erosion width
            2. `outer_distance`: outer erosion width
            3. `fgbg`: no additional keword arguments.
        6. Instantiate both the training and validation data generators using the `.flow()` method.
        7. Return `training_data` and `val_data` objects
2. Train the model:
    1. Clear clutter from previous runs of TensorFlow.
    2. Instantiate model using the `create_model` function.
        1. Create model using `PanopticNet` model from the DeepCell model zoo.
            1. Instantiated using all of the original properties except `backbone_levels` are C1-C5 instead of C3-C5.
    3. Create a logger file uisng the TensorFlow callbacks `CSVLogger` method pointing at `train_log`.
    4. Create callbacks for:
        1. `ModelCheckpoint` to save the entire model.
        2. `LearningRateScheduler` as indicated by the `rate_scheduler` function defined in `deepcell-tf`.
        3. `ReduceLROnPlateau` to slow down once accuracy begins to plateau.
        4. `EarlyStopping` with `patience` set to 10 and saved with previous best weights.
    5. Fit the model and return the `history` variable for monitoring the progress.
3. Save the new model using the `model.save()` function, not including the optimizer and overwriting previous files.
4. Save a `metadata.yaml` file in the model directory with the history of model training.

## PyTorch pipeline
1. Load data.
    1. Loading tracking data from `.npz` file.
    2. Update data using `update_data_split` function. `source` is tracking (a dictionary), `data_path` is the path to the segmentation data.
        1. Opens the segmentation data and separates out X, Y, meta, and original_split for each of the train/test/val splits.
        2. Concatenates them together
        3. If the metadata is not in the `source` set, then it's missing and we need to find it in the original split
        4. Create the `data` variable which is returned and populate train/test/val splits for each of the data sources.
        5. Search for missing values that are in the tracking but not segmentation and place it. in the new data file formed.
        6. Concatenate it all together.
        7. Return the data structure.
    3. Set up PyTorch DataLoader object using previously written in `utils.loaders` as `CroppingDatasetTorch` object.
        1. The only thing that the original pipeline does to preprocess the data is histogram normalization, but Mesmer had much more preprocessing. I'll stick with histnorm for now and include that before creating the DataLoader.
        2. Construct a DataLoader object using the `CroppingDatasetTorch` class.

3. Save the model.
4. Save the training metadata.



