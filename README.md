# PyTorch implementation of Caliban's cell tracker model

## Description

Caliban originally consisted of two parts: the first is the nuclear segmentation pipeline, which is based on a Panoptic Network and segmented nuclei in a live-cell time lapse. The second is a graph neural network used to infer linkages between two frames of the movie, based on the appearance, morphology, and location of each cell. This repo is the PyTorch port of the cell tracker model in Caliban.

## The Model
### The Encoders: Cell Identity

Cells in the image are identified using the nuclear mask provided. Each cell is cropped to a 32x32 image around the centroid of the nucleus (called the `appearance` image). Further, the cell's nuclear morphology (represented by area, perimeter, eccentricity) is recorded. Finally, the cell's location in the image is recorded (represented by the centroid). Given N cells in the image of time T:

- Appearances: `T, N, 1, 32, 32`, where `1` is the number of channels in the image.
- Morphology: `T, N, 3`.
- Centroid: `T, N, 2`, Y and X values for location in the image.

After the appearance, morphology, and centroid of each cell is encoded, they are each of shape `T, 64, N`, where `64` is the number of channels returned from each encoder.

### The Encoders: Cell Relationship

Cell relationships are based on the pairwise distances between objects. The centroids are used to determine pairwise distances, repsented in a distance matrix. A threshold is used to get an adjacency matrix, where a 1 value indicates an edge in the network between those two nodes, and a 0 means there is no connection. We set the threshold as 64 pixels.

The adjacency matrix is then normalized using symmetric normalization:

$$ \bold{\hat{A}} =  \bold{D}^{-1/2}\bold{A}\bold{D}^{-1/2}$$

A graph attention network is built using two graph convolution layers (a divergence from Caliban's original one graph convolution). Each node of the network is given the embeddings previously calculated with the identity encoder. Messages are then passed between neighboring cells. Centroids are passed through the `NeighborhoodEncoder` module unchanged.

## The Model Branches

At this point, the training and inference branches diverge significantly. I will discuss the training branch first, then discuss the inference branch.

### The Training Branch

Once the model has calculated the embeddings for each cell, the data is split into two arrays: the current embeddings (from indices 0 to -2), and future embeddings (from indices 2 to -1). This yields one frame of current embeddings (the cell tracker's state at the current moment), and the future embeddings (the ones we are trying to assign to tracks).

The current embeddings are passed into an LSTM module. This module acts as a short-term memory bank. It updates the embeddings based on the embeddings' "history", in this case on frames of the time lapse leading up to the frames we are testing. You may notice that we now have two arrays: one with a "history" applied, and the other without.

The current and future embeddings are merged together. At this point, the previous split embeddings are of shape `(T-1, Nx, F)` and `(T-1, Ny, F)`. These two arrays are concatenated together to give an array of shape `(T-1, Nx, Ny, 2F)`. We leave the morphology, centroid, and appearance embeddings alone or a moment, and turn our attention to the movement of cells within frames and between frames.

The difference between centroids within the same frame and between two frames is calculated using simple subtraction. These values are then passed into the `DeltaEncoder`, which encodes the changes in centroid between frames (i.e. movement), and within the same frame (i.e. pairwise distances between cells). The second (`deltas_current`) is then passed into a different LSTM to incorporate changes in relative distances between cells. The embedding comparisons and the deltas are returned to the decoder, which returns a temporal adjacency matrix (TAM), predicting the mapping of cells in the "current" frame onto the cells in the "future" frame.

### The Inference Branch

After being trained to infer deltas and embeddings, the model is able to accept "current" embeddings (embeddings from frames T-8 to T-1), and "future" embeddings (embeddings of the cells you want to track) as well as current and future centroids. These embeddings are calculated using the `NeighborhoodEncoder` module that was trained on images, centroids, and morphologies before being fed into the inference branch.

The model calculates the deltas and encodes them, alongside encoding the deltas and appearance embeddings using their respective LSTM layers. Finally, these delta and appearance embeddings are taken together and passed on to the decoder, which returns the TAM linking the current frame `T-1` to the future frame `T`.