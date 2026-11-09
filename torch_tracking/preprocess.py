import os
import io
import shutil
import json
import tarfile
import zarr

from torch_tracking.utils import relabel_sequential_lineage, get_image_features, histogram_normalization, resize, scale_factor, LINEAGES_KEY
from tqdm import tqdm
from pathlib import Path

import numpy as np

def get_features(X, y, max_cells, appearance_shape=(32, 32, 1), 
                 clahe=False, mpps=None):
    """Per-cell features for every movie: appearances, morphologies and centroids.

    The temporal adjacency (the training targets) is not built here. The dataloader
    builds it per window from the lineage; see ``loader.get_temporal_adjacency``.
    """
    
    B, T, H, W, C = X.shape

    max_tracks = max_cells
    
    batch_shape = (B, T, max_tracks)
    
    # Initialize feature arrays
    appearances = np.zeros(batch_shape + appearance_shape, dtype='float32')
    morphologies = np.zeros(batch_shape + (3,), dtype='float32')
    centroids = np.zeros(batch_shape + (2,), dtype='float32')

    # Histogram normalization on image
    if clahe:
        print("Preprocessing whole image with CLAHE...")
        X = histogram_normalization(X, data_format='channels_last')

    for batch in tqdm(range(B), desc="Processing movie batches"):
        
        # Rescale to the model resolution with the same rule inference applies.
        factor = scale_factor(mpps[batch])
        new_size = (int(factor*H), int(factor*W))

        X_batch = resize(X[batch], shape=new_size, data_format='channels_last')
        y_batch = resize(y[batch], shape=new_size, data_format='channels_last', labeled_image=True)

        for frame in range(T):
            frame_features = get_image_features(
                X_batch[frame], 
                y_batch[frame],
                appearance_dim=appearance_shape[0]
            )
            
            track_ids = frame_features['labels'] - 1  # Convert to 0-indexed
            
            # Store features
            centroids[batch, frame, track_ids] = frame_features['centroids']
            morphologies[batch, frame, track_ids] = frame_features['morphologies']
            appearances[batch, frame, track_ids] = frame_features['appearances']

    features = {
        'appearances': appearances,
        'morphologies': morphologies,
        'centroids': centroids,
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

SIDECAR_NAME = 'data-source.npz'


def _resolutions_from_sidecar(trks_dir, split):
    """Per-movie resolutions from ``data-source.npz`` beside the ``.trks`` file.

    The file is keyed by split name; each value is a per-movie metadata array whose third
    column (index 2) is the movie's resolution in um/px. See ``docs.md``.
    """
    sidecar = os.path.join(trks_dir, SIDECAR_NAME)
    if not os.path.isfile(sidecar):
        raise FileNotFoundError(
            f"No per-movie resolutions were supplied and {sidecar} does not exist. "
            "Pass `mpps=` with one resolution per movie, or place data-source.npz beside "
            "the .trks file. No default resolution is assumed."
        )
    with np.load(sidecar, allow_pickle=True) as metadata:
        if split not in metadata.files:
            raise KeyError(
                f"{sidecar} has no entry for split '{split}' (it holds "
                f"{sorted(metadata.files)}). No default resolution is assumed."
            )
        return np.asarray(metadata[split][:, 2], dtype='float64')


def convert_trk_to_zarr(filename, out_dir=None, track_length=8, mpps=None):
    """Convert a ``.trks`` split to its unprocessed and processed zarrs.

    Writes ``<split>.zarr``, ``<split>_proc.zarr`` and ``<split>.json`` into ``out_dir``,
    or beside the ``.trks`` file when ``out_dir`` is None. ``out_dir`` must already exist.

    ``track_length`` is the training window the processed zarr is chunked for; it should
    match the ``track_length`` the dataloader will sample with.

    ``mpps`` is one declared resolution (um/px) per movie. When omitted, they are read
    from ``data-source.npz`` beside the ``.trks`` file; when neither is available,
    conversion fails rather than assume a resolution.
    """

    trks_dir = os.path.dirname(filename)
    split = os.path.splitext(os.path.basename(filename))[0]

    if out_dir is None:
        file_dir = trks_dir
    else:
        file_dir = os.fspath(out_dir)
        if not (os.path.isdir(file_dir) and os.access(file_dir, os.W_OK)):
            raise NotADirectoryError(
                f"Output directory {file_dir} does not exist or cannot be written. "
                "Outputs are not redirected beside the .trks file."
            )
    processed_file = os.path.join(file_dir, split) + '_proc.zarr'

    # Settle where the resolutions come from before the expensive load.
    if mpps is None:
        mpps = _resolutions_from_sidecar(trks_dir, split)
    mpps = np.asarray(mpps, dtype='float64')

    data = load_trks(filename)
    X = data['X']
    y = data['y']
    lineages = data['lineages']

    if mpps.shape != (X.shape[0],):
        raise ValueError(
            f"{split} holds {X.shape[0]} movies but {mpps.size} resolutions were given; "
            "exactly one per movie is required."
        )

    # Step 1: correct lineages function
    X, y, lineages = correct_lineages(X, y, lineages)

    # Step 2: get maximum cells from lineages and save
    max_cells = 0

    for i in lineages:
        curr_m = len(i.keys())
        if curr_m > max_cells:
            max_cells = curr_m

    # Compute features before writing anything, so a failure here leaves no stale output.
    features = get_features(X, y, max_cells, mpps=mpps)

    with open(os.path.join(file_dir, split) + '.json', 'w') as file:
        json.dump(lineages, file)

    output_file = os.path.join(file_dir, split) + '.zarr'
    _write_zarr_pair(output_file, processed_file, X, y, features, lineages, mpps, track_length)


#: Processed arrays indexed by (movie, frame, ...). These are read one window at a time.
FRAME_INDEXED_ARRAYS = ('appearances', 'morphologies', 'centroids')


def window_chunks(shape, track_length):
    """Chunk shape that makes one movie's ``track_length``-frame window a bounded read.

    One movie on the batch axis, ``track_length`` frames on the time axis (capped at the
    array's own length), and every remaining axis whole - every cell slot, and each crop
    intact. zarr's automatic choice splits a window across movies, frames and crop tiles.
    """
    frames = max(1, min(track_length, shape[1]))
    return (1, frames) + tuple(shape[2:])


def _write_zarr_pair(raw_path, processed_path, X, y, features, lineages, mpps, track_length=8):
    """Write the unprocessed and processed zarrs together, or neither.

    Both are staged beside their destinations and only moved into place once both are
    complete, so a failure part-way cannot leave one file holding ``mpp`` and the other
    not. Both receive the same ``mpps``. The processed zarr also carries ``lineages`` (the
    corrected lineage of every movie, as JSON in its attributes), so the dataloader can
    build each window's training targets without reading a sibling file.
    """
    staged = {raw_path: raw_path + '.partial', processed_path: processed_path + '.partial'}
    try:
        for path in staged.values():
            shutil.rmtree(path, ignore_errors=True)

        z = zarr.open(staged[raw_path], mode='w')
        z['X'] = X
        z['y'] = y
        z.create_array('mpp', data=mpps)

        z_proc = zarr.open(staged[processed_path], mode='w')
        # Record the window the layout was built for, so a reader can tell.
        z_proc.attrs['track_length'] = int(track_length)
        z_proc.attrs[LINEAGES_KEY] = json.loads(json.dumps(lineages))
        for k, v in features.items():
            if k in FRAME_INDEXED_ARRAYS:
                z_proc.create_array(k, data=v, chunks=window_chunks(v.shape, track_length))
            else:
                z_proc.create_array(k, data=v)
        z_proc.create_array('mpp', data=mpps)
    except BaseException:
        for path in staged.values():
            shutil.rmtree(path, ignore_errors=True)
        raise

    for final, partial in staged.items():
        shutil.rmtree(final, ignore_errors=True)
        os.rename(partial, final)


# Conversion is reached through `convert_trk_to_zarr`, which the test suite exercises. A
# `__main__` block used to sit here that converted every `.trks` file under a hardcoded
# `~/.deepcell/tracking/`; nothing ran it. Divergence 83.
