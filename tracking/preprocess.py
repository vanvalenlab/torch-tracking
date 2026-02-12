import os
import io
import json
import tarfile
import zarr
import glob

from tracking.utils import relabel_sequential_lineage, get_image_features, histogram_normalization
from scipy.spatial.distance import cdist
from tqdm import tqdm

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
                 crop_mode='fixed', clahe=True, distance_threshold=64, verbose=True):
    
    n_batches = X.shape[0]
    n_frames = X.shape[1]
    max_tracks = max_cells
    
    batch_shape = (n_batches, n_frames, max_tracks)
    
    # Initialize feature arrays
    appearances = np.zeros(batch_shape + appearance_shape, dtype='float32')
    morphologies = np.zeros(batch_shape + (3,), dtype='float32')
    centroids = np.zeros(batch_shape + (2,), dtype='float32')
    adj_matrix = np.zeros(batch_shape + (max_tracks,), dtype='float32')
    temporal_adj_matrix = np.zeros(
        (n_batches, n_frames-1, max_tracks, max_tracks), 
        dtype='float32'
    )
    mask = np.zeros(batch_shape + (max_tracks,), dtype='bool')
    track_length = np.zeros((n_batches, max_tracks, 2), dtype='int32')

    # Histogram normalization on image
    if clahe:
        print("Preprocessing whole image with CLAHE...")
        X = histogram_normalization(X, data_format='channels_last')
    
    for batch in tqdm(range(n_batches), desc="Processing batches"):
        
        # =====================================================================
        # STEP 1: Extract per-frame features (appearances, morphologies, etc.)
        # =====================================================================
        
        for frame in range(n_frames):
            frame_features = get_image_features(
                X[batch, frame], 
                y[batch, frame],
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
        
        temporal_adj_matrix[batch] = get_temporal_adjacency(lineages[batch], max_frames=n_frames, max_cells=max_cells)

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

        # JSON only allows strings as keys, so convert them back to ints
        # for i, tracks in enumerate(lineages):
        #     lineages[i] = {str(k): v for k, v in tracks.items()}

    return {'lineages': lineages, 'X': raw, 'y': tracked}

def correct_lineages(X, y, lineages):
    """Ensure valid lineages and sequential labels for all batches"""
    new_X = []
    new_y = []
    new_lineages = []
    for batch in tqdm(range(y.shape[0])):
        # if is_valid_lineage(self.y[batch], self.lineages[batch]):

        y_relabel, new_lineage = relabel_sequential_lineage(
            y[batch], lineages[batch])

        new_X.append(X[batch])
        new_y.append(y_relabel)
        new_lineages.append(new_lineage)
        # else:
        #     print('Invalid lineage detected.')

    X = np.stack(new_X, axis=0)
    y = np.stack(new_y, axis=0)
    lineages = new_lineages

    return X, y, lineages

def convert_trk_to_zarr(filename, out_dir=None):

    data = load_trks(filename)

    X = data['X']
    y = data['y']
    lineages = data['lineages']

    # Step 1: correct lineages function
    X, y, lineages = correct_lineages(X, y, lineages)

    # Step 2: get maximum cells from lineages
    max_cells = 0
    for i in lineages:
        curr_m = len(i.keys())
        if curr_m > max_cells:
            max_cells = curr_m

    if 'test' in filename:

        if out_dir is None:
            file_dir = os.path.dirname(filename)
            split = os.path.splitext(os.path.basename(filename))[0]
            output_file = os.path.join(file_dir, split) + '.zarr'
            processed_file = os.path.join(file_dir, split) + '_proc.zarr'

        z = zarr.open(output_file)
        z['X'] = X
        z['y'] = y

        with open(os.path.join(file_dir, split) + '.json', 'w') as file:
            json.dump(lineages, file)

        features = get_features(X, y, lineages, max_cells, crop_mode='fixed')

        z2 = zarr.open(processed_file)

        for k, v in features.items():
            z2.create_array(k, data=v)

    else:

        if out_dir is None:
            file_dir = os.path.dirname(filename)
            split = os.path.splitext(os.path.basename(filename))[0]
            output_file = os.path.join(file_dir, split) + '_proc.zarr'
        
        features = get_features(X, y, lineages, max_cells, crop_mode='fixed')

        z = zarr.open(output_file)

        for k, v in features.items():
            z.create_array(k, data=v)

if __name__ == "__main__":

    data_directory = 'data/DynamicNuclearNet-tracking-v1_0/test.trks'
    
    for filename in glob.glob(data_directory):
        print(f"Converting {os.path.basename(filename)}")
        convert_trk_to_zarr(filename)