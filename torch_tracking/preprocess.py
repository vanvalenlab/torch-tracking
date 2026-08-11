import os
import io
import json
import tarfile
import zarr

from torch_tracking.utils import relabel_sequential_lineage, get_image_features, histogram_normalization, resize
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path

import numpy as np

def get_temporal_adjacency(lineage, max_frames, max_cells):
    
    adjacency = np.full((max_frames, max_cells, max_cells), 0)

    frame_list = []
    div_list = []
    id_list = []
    curr_m_frames = 0

    # In the Cell ID domain (along axes 2 and 3)
    for _, attr in lineage.items():
        frame_list.append(attr['frames'])
        id_list.append(int(attr['label'])-1)
        
        div_list.append((attr['daughters'], attr['frames'][-1]))

        length = len(attr['frames'])
        if curr_m_frames < length:
            curr_m_frames = length

    frame_mat = np.full((max_frames, max_cells), 0)

    # In the frame domain (along axis 1)
    # Find self/not self
    for i, frames in enumerate(frame_list):
        end_frame = frames[-1]
        frame_mat[frames, id_list[i]] = 1
        frame_mat[end_frame:, id_list[i]] = 0

    # Place self/not self on diagonal of frame/cell/cell axis
    idx = np.arange(max_cells)
    adjacency[:, idx, idx] = 0
    adjacency[:, idx, idx] = frame_mat

    # Place divisions and daughter padding
    for i, (daughters, frame) in enumerate(div_list):
        if frame is not None:

            # adjust label to index
            daughters = [daughter-1 for daughter in daughters]

            # Place daughters
            adjacency[frame, id_list[i], daughters] = 2

    # Place padding
    for frame in range(max_frames-1):

        # find where padding is 
        padding_where = np.argwhere(adjacency[frame+1] > 0).squeeze()

        # If the row and column need padding (not empty)
        if padding_where.any():
            padding = padding_where[-1,1] + 1
            adjacency[frame, padding:, :] = -1
            adjacency[frame, :, padding:] = -1

    # # Place temporal padding
    adjacency = adjacency[:-1]
    adjacency[curr_m_frames:,] = -1   

    return adjacency


def get_features(X, y, lineages, max_cells, appearance_shape=(32, 32, 1), 
                 crop_mode='fixed', clahe=False, distance_threshold=64, mpps=None):
    
    B, T, H, W, C = X.shape

    max_tracks = max_cells
    
    batch_shape = (B, T, max_tracks)
    
    # Initialize feature arrays
    appearances = np.zeros(batch_shape + appearance_shape, dtype='float32')
    morphologies = np.zeros(batch_shape + (3,), dtype='float32')
    centroids = np.zeros(batch_shape + (2,), dtype='float32')
    adj_matrix = np.zeros(batch_shape + (max_tracks,), dtype='float32')
    temporal_adj_matrix = np.zeros(
        (B, T-1, max_tracks, max_tracks), 
        dtype='float32'
    )
    mask = np.zeros(batch_shape + (max_tracks,), dtype='bool')
    track_length = np.zeros((B, max_tracks, 2), dtype='int32')

    # Histogram normalization on image
    if clahe:
        print("Preprocessing whole image with CLAHE...")
        X = histogram_normalization(X, data_format='channels_last')

    for batch in tqdm(range(B), desc="Processing movie batches"):
        
        new_size = (int(0.55/mpps[batch]*H), int(0.55/mpps[batch]*W)) # resizing image so that crops are standardized to MPP of 0.55

        X_batch = resize(X[batch], shape=new_size, data_format='channels_last')
        y_batch = resize(y[batch], shape=new_size, data_format='channels_last', labeled_image=True)

        for frame in range(T):
            frame_features = get_image_features(
                X_batch[frame], 
                y_batch[frame],
                appearance_dim=appearance_shape[0],
                crop_mode=crop_mode
            )
            
            track_ids = frame_features['labels'] - 1  # Convert to 0-indexed
            
            # Store features
            centroids[batch, frame, track_ids] = frame_features['centroids']
            morphologies[batch, frame, track_ids] = frame_features['morphologies']
            appearances[batch, frame, track_ids] = frame_features['appearances']
            
            # Update mask for valid cells
            if track_ids.size > 0:
                mask[batch, frame, :track_ids[-1] + 1, :track_ids[-1] + 1] = True
            
            # Compute spatial adjacency matrix (distance-based)
            cent = centroids[batch, frame]
            distances = cdist(cent, cent, metric='euclidean')
            within_threshold = distances < distance_threshold
            
            # Exclude padding (cells with zero morphology)
            morph = morphologies[batch, frame]
            is_padding = np.matmul(morph, morph.T) == 0
            
            adj = within_threshold * (~is_padding)
            adj_matrix[batch, frame] = adj.astype(np.float32)
        
        temporal_adj_matrix[batch] = get_temporal_adjacency(lineages[batch], max_frames=T, max_cells=max_cells)

    features = {
        'appearances': appearances,
        'morphologies': morphologies,
        'centroids': centroids,
        'labels': temporal_adj_matrix,
        'track_length': track_length
    }
    
    return features 

def load_trks(filename):
    """Load a trk/trks file.

    Args:
        filename (str or BytesIO): full path to the file including .trk/.trks
            or BytesIO object with trk file data

    Returns:
        dict: A dictionary with raw, tracked, and lineage data.
    """

    if isinstance(filename, io.BytesIO):
        kwargs = {'fileobj': filename}
    else:
        kwargs = {'name': filename}

    with tarfile.open(mode='r', **kwargs) as trks:

        # numpy can't read these from disk...
        with io.BytesIO() as array_file:
            array_file.write(trks.extractfile('raw.npy').read())
            array_file.seek(0)
            raw = np.load(array_file)

        with io.BytesIO() as array_file:
            array_file.write(trks.extractfile('tracked.npy').read())
            array_file.seek(0)
            tracked = np.load(array_file)

        # trks.extractfile opens a file in bytes mode, json can't use bytes.
        try:
            trk_data = trks.getmember('lineages.json')
        except KeyError:
            try:
                trk_data = trks.getmember('lineage.json')
            except KeyError:
                raise ValueError('Invalid .trk file, no lineage data found.')

        lineages = json.loads(trks.extractfile(trk_data).read().decode())
        lineages = lineages if isinstance(lineages, list) else [lineages]

    return {'lineages': lineages, 'X': raw, 'y': tracked}

def correct_lineages(X, y, lineages):
    """Ensure valid lineages and sequential labels for all batches"""
    new_X = []
    new_y = []
    new_lineages = []
    for batch in tqdm(range(y.shape[0]), desc="Validating lineages"):

        y_relabel, new_lineage = relabel_sequential_lineage(
            y[batch], lineages[batch])

        new_X.append(X[batch])
        new_y.append(y_relabel)
        new_lineages.append(new_lineage)

    X = np.stack(new_X, axis=0)
    y = np.stack(new_y, axis=0)
    lineages = new_lineages

    return X, y, lineages

def convert_trk_to_zarr(filename, out_dir=None):

    data = load_trks(filename)
    dir = os.path.dirname(filename)

    metadata = np.load(os.path.join(dir, 'data-source.npz'),allow_pickle=True)

    if out_dir is None:
        file_dir = os.path.dirname(filename)
        split = os.path.splitext(os.path.basename(filename))[0]
        processed_file = os.path.join(file_dir, split) + '_proc.zarr'

    X = data['X']
    y = data['y']
    lineages = data['lineages']
    mpps = metadata[split][:,2]


    # Step 1: correct lineages function
    X, y, lineages = correct_lineages(X, y, lineages)

    # Step 2: get maximum cells from lineages and save
    max_cells = 0

    for i in lineages:
        curr_m = len(i.keys())
        if curr_m > max_cells:
            max_cells = curr_m

    with open(os.path.join(file_dir, split) + '.json', 'w') as file:
        json.dump(lineages, file)

    # Write regular unprocessed file
    output_file = os.path.join(file_dir, split) + '.zarr'
    z = zarr.open(output_file, mode='w')
    z['X'] = X
    z['y'] = y

    # Write processed file
    features = get_features(X, y, lineages, max_cells, crop_mode='fixed', mpps=mpps)

    z_proc = zarr.open(processed_file, mode='w')

    for k, v in features.items():
        z_proc[k]=v
    

if __name__ == "__main__":

    data_directory = Path.home() / '.deepcell/tracking/'
    print(data_directory)
    
    for filename in Path(data_directory).glob('*.trks'):
        print(f"Converting {os.path.basename(filename)}")
        convert_trk_to_zarr(filename)