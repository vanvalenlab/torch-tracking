# Tracking Model

## Model Architecture: an intuitive explanation

The cell tracking model is a multimodal deep learning model that generates feature embeddings from nuclei in an image and uses these to solve a linear assignment problem: given a list of tracks and a list of cells, what is the optimal pairing between cells and tracks? The model itself is composed of four modules: preprocessing, the neighborhood encoder, the delta encoder, and the tracking decoder.

### Preprocessing

The preprocessing step established a handful of important parameters and features that inform the downstream model. A subtle parameter that's generated during preprocessing is the `max_cell` parameter. Over the time-lapse, cells will proliferate, enter, and exit the frame, leading to new tracks being generated and old tracks being terminated. This parameter, along with the number of frames processed at once, defines the shape of the feature arrays. For brevity, we will refer to `max_cell` as `N` and number of time-lapse frames as `T`.

To extract parameters about each nucleus in the time lapse, we must first generate features that describe each nucleus. To do this, we generate four attributes:

- The **appearance** `(T, N, H, W, 1)`: a 32x32 crop centered on the nucleus
- The **morphology** `(T, N, 3)`: the area, circumference and eccentricity of a nucleus
- The **centroid** `(T, N, 2)`: the coordinates of the nucleus in the image in pixels
- The **adjacency matrix** `(T, N, N)`: describes graph edges between cells less than 64 pixels apart

```{admonition} Intuition
Instead of nuclei, imagine that your objects are extras milling about in a movie you're watching. We want to be able to track the instances of those extras as they pass by each other, hug, shake hands, or (if it's a sci-fi movie) split into two people. To do this, we want to make some sort of representation of each person by extracting their features, their behavior, and their rough location in the image. This is what preprocessing does: it extracts information about all of the nuclei in a time-lapse image and stores them for more downstream processing.
```

### The neighborhood encoder

The appearances, morphology, and centroid information are then passed through their own encoders. These encoders transforms the raw numerical values into embeddings: high-dimensional representations of each cell's appearance, morphology, and centroid.

These representations are then flattened and stacked, so that we are left with a large 3D array with shape `(T, N, 3 * encoder_dimension)` which is just to say we have an array over time of every cell from each of the encoders indicated above. Here's where we use the adjacency matrix we made in the preprocessing step to provide neighborhood context to these embeddings.

We then set up a graph attention network (GAT). The GAT acts on a graph, where the nodes represent cells and the edges represent adjacency (the attribute of being close to another cell or set of cells). The nodes attributes are the embeddings we generated earlier. Since cells that are close together in the frame are often sources of error (like track swapping), we want to learn the subtle appearance, morphology and location hints that can give us clues as to the best assignment to make later on. We use two graph convolutions to pass information between the network, updating the embeddings at each node to take into account neighborhood context.

The node features are extracted and passed on to the next step of the process.

```{admonition} Intuition
To continue our analogy, we have a record of the attributes of each person on screen: their shape, their appearance, and where they are in the shot. This is almost always enough information: the richness of individual appearances is enough to tell people apart (that's what our mind does automatically, albeit in a much more complicated way). 

But what if our movie set is full of John Malkoviches, as was the case during a particular scene in [*Being John Malkovich*][malkovich]? How could we tell them apart? This is where graph neural networks (GNNs) come in handy (a very good primer on GNNs can be found [here][gnnpub]). Instead of thinking of each John Malkovich as John Malkovich, we look at the context by assigning pairs or triplets of Malkoviches into neighborhoods: in this neighborhood, which Malkovich is wearing the dress? Which is on the piano? Which is the original Malkovich wearing a baseball cap? The context is learned during training, so we only keep our eyes on the details that matter and ignore unimportant information.

After updating our understanding of all of the John Malkovich clones, the features returned are still very similar, but have just enough updated information that can provide context for keeping track of them all.
```

### The delta encoder

At this point, we have extracted information about individual nuclei, updated with the context that each nucleus is in. We supply the model with 8 frames at a time, where the final frame is the one we want to predict, and the previous 7 are the tracking "history" that provides additional time context to our prediction. We do this by generating deltas, or changes, between the neighborhood embeddings or the centroids.

#### Embedding deltas

This compares each track along the `N` axis to its associated track in the next frame. When we calculate the deltas between frames, we have to collapse that history into an embedding that is compatible with our inference frame. One way we could have done this is use a multilayer perceptron (MLP) across the time dimension. Another more sophisticated way is to use a long short-term memory ([LSTM][lstm]) module. This is the route we chose. Using this, we can "squash" our track history into a representation that we can compare to our inference frame. We concatenate the history and inference frame together and send it to the next module.

#### Centroid deltas  

In our data, we have two different types of changes to account for: the changes between the same object between frames (between-frame deltas) and the difference between each object in the historical frames and all objects in the inference frame (within-frame deltas). To calculate the between-frame deltas, we calculate the change in centroid between adjacent pairs of frames, encode this using an MLP, and pass it through another LSTM to squash the centroid history into an embedding. To calculate within-frame deltas, we simply subtract the object centroids from the previous frames from the object centroids of the current frame. After embedding, we concatenate the between-frame and within-frame deltas and send it to the next module.

```{admonition} Intuition (Embeddings)
Now that we have the Malkovich embeddings and locations in the frame, it's time to incorporate that information together to predict the next frame's Malkoviches. To do this, we have to generate comparisons of two different types: location-based and appearance-based.

Appearance based changes are easy: between these two frames, although two Malkoviches look alike, there are subtle clues as to which one is which. We have to compare each Malkovich to every other and make a mental map of how different the Malkoviches are from each other. This is what the embedding delta module does. But we're not given access to just two frames to compare: we're given the whole scene! How do we make this comparison? We use our memory! 

We remember aspects of what each Malkovich is doing in the scene over time: one is playing piano, one is singing while laying on the piano, one is serving food. For example, you only know that the Malkovich on the piano is singing because they're holding the microphone close to their mouth, and the only way you know that is by remembering that motion through time. This is what the LSTM does: it integrates historical information across the time dimension and updates the embeddings to make comparisons with current embeddings more meaningful.
```

```{admonition} Intuition (Centroids)
The embeddings only tell half the story: they give context for what the Malkoviches look like, but not where they are relative to each other. We turn to the centroids (the location of each Malkovich). 

As with the embeddings, we want to remember each Malkovich's movement. This gives an idea of "movement" attributes of each Malkovich. For example, some may be walking quickly, like the waitresses, while others are sitting. It would be foolish to assign a moving Malkovich to one that was sitting for the whole scene! In this way, we use the LSTM module to generate embeddings that describe each Malkovich's movement habits. This is our "between-frames" embedding.

In the same way, we want a way to describe how different the locations of the Malkoviches change between frames. This is relatively simple since there's no "history" to remember: it's simply looking at the previous frame and the next to compare the movements of each Malkovich instance. This gets us closer to accomplishing our original task: linking together the Malkoviches frame by frame. This is our "within-frames" embedding.
```

### The decoder

Our embedding and centroid deltas are combined together and sent along to the tracking decoder. This module uses an MLP to transform the embedding and centroid deltas into a prediction of shape `(N, N, 3)`. Each row of the matrix corresponds to a historical object in the frame (all nuclei at time `t-1`) and each column of the matrix corresponds to the objects in the inference frame (all nuclei at time `t`). The third dimension stores probabilities `p_different`, `p_same`, `p_div`, which describe each cell pair's relationship to each other. Cell pairs with high `p_different` are not likely to be the same cell. Pairs with high `p_same` are likely to be the same cell. Pairs with high `p_div` are likely to have a parent-child relationship, such that the cell in frame `t-1` divides and becomes two or more cells in frame `t`. This is returned to the inference harness for post-processing and cell lineage reconstruction.


### The linear assignment problem

Now that we have the assignment probabilities of objects in two adjacent frames, it's time to find the most optimal assignment of current objects to previous tracks. To do this, we can convert the matrix `p_same` into a cost matrix by subtracting it from 1. We then use the Hungarian algorithm to find the optimal assignment of objects to tracks. In a perfect world, there is always a 1:1 ratio of objects to tracks, so each track is assigned an object.

But what is the case where the number of tracks exceeds the number of objects? Or if the number of objects exceeds the number of tracks? This is very common! During live-cell imaging, cells will die, move out of or into of the field of view, or divide, leading to a mismatch between objects and tracks to assign them to. This is called an imbalanced assignment problem. To circumvent this, we give each object the opportunity to be assigned to a "shadow object" which either represents a birth (a new cell, no track to assign it to) or a death (an old track with no object). In these cases, we either create or terminate a track if the object or track does not have a low-cost assignment.

Mitosis is a special case. If a cell divides, it gives birth to two new objects. These can both be categorized as two new tracks, or one new track and one incorrect assignment to the parent track. This is where `p_div` comes in. During any new track creation, we have the option to override the assignment if `p_div` is sufficiently high, that is, if the parent track is predited to be dividing at that frame. In this case, the parent track is terminated, two new tracks are created, and the division link is indicated in the lineage data structure.

```{admonition} Intuition
Now that we have a clear mental map of the Malkoviches in the scene, it's time to convert the learned aspects of each instance into an actionable Malkovich tracking scheme.

Let's say for each pair of frames of the scene, we keep track of all pairs of Malkoviches. The information that we've generated about each instance is such that, for the most part, it's clear when one Malkovich is distinct from all of the others. For example, we can say that, in our mind, the probability of a fast Malkovich in a dress being a waitress is very high. We can then assign the fast Malkovich in a dress to a waitress we saw in the previous frame. On the other hand, we would not assign this Malkovich to one that was sitting and wearing a suit since the probability of that assignment is very low. Lastly, if in a hypothetical schlock horror remake *Being John Malkoviches*, one Malkovich were to reproduce by fission into two new Malkoviches, the probability of a Malkovich-Malkoviches relationship with those two new instances would be high.

To reconstruct the Malkovich tracks, it's simply a matter of finding the assignments with the highest probability across frames. Easy! Congratulations, you just took the long way to understanding that scene of from *Being John Malkovich.*
```

[malkovich]: https://en.wikipedia.org/wiki/Being_John_Malkovich

[gnnpub]: https://distill.pub/2021/gnn-intro/

[lstm]: https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf