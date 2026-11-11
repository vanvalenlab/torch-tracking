"""L1 - model structural contracts. Needs torch only; no checkpoint, no dataset.

This layer pins structure that retraining must never change: the class ordering, the
temporal adjacency matrix shape, and the architecture `CellTracker` hardcodes. Random
weights are sufficient, because none of these depend on what the model learned.

The output pins at the top of the file guard against unintended changes to what the model
computes rather than stating a permanent contract - see their comment. See
tests/README.md.
"""

import hashlib

import pytest
import torch
import torch.nn as nn

from tests.conftest import MODEL_ARCH, UNTRAINED_CHECKPOINT

pytestmark = pytest.mark.l1


# --------------------------------------------------------------------------------------
# Output pin
#
# Changes to documentation, dead code or argument handling must not change a tensor the
# model computes, and that is only credible if it is checked. These digests fix the exact
# bytes of the model's outputs on a seeded input.
#
# Unlike the L2 goldens, which are deliberately not frozen because untrained weights make
# them a change-detector rather than a measurement, a change-detector is exactly what is
# wanted here. The caveat `tests/data/make_random_init.py` records applies equally: a torch
# upgrade can move the last bits of a float without any code changing. If these fail after
# a torch upgrade and nothing else, regenerate them and say so in the commit.
# --------------------------------------------------------------------------------------

#: Shape of the pinned training batch: one movie, three frames, four cells, 32px crops.
#: Cells 0 and 1 are adjacent real cells; 2 and 3 stand in for padding slots.
PIN_BATCH = {"batch": 1, "frames": 3, "cells": 4, "crop": 32}

# These digests were last re-taken when the model's normalization was corrected. Both
# training-forward digests moved:
#
#   * appearance crops are now normalized over the whole image rather than one row at a
#     time. This is the substantive cause.
#   * `TemporalMerge` now runs its LSTM along each track rather than across slots, which
#     moves the inference digest as well because that path uses the same layer.
#   * padding is now masked out of BatchNorm statistics and the graph no longer gives
#     padding slots edges, but on THIS fixture those are no-ops - all four cells have
#     positive morphology, so none is padding - apart from reordering the sums inside
#     BatchNorm, which moves the last bit of a float. That they are no-ops is what
#     `PIN_NO_APPEARANCE_NORM_ALL_REAL` checks, free of the appearance change.
#
# Their values before that correction were
#   eval mode   d9710afe5f5c35f7d7852037e978543432732bbadc15285c7e2e8cfc54e9f82f
#   train mode  58a007bc11dc3678fc267d652ca9ed9768e22ace20ae5e64d25d803cddb13e45
# `PIN_EVAL_INFERENCE_FORWARD` did NOT move for the masking, graph or crop changes -
# inference takes embeddings and never sees a crop, a graph, or a padded batch - and moved
# only for the `TemporalMerge` fix. Its value before that fix was
# 1ad2feebc10d2362d23198ad7d243df533916dea6931db07d9f10b4adaf04506.
PIN_EVAL_TRAINING_FORWARD = "83c9c4c93b77e673f924d9a76b3c0f302edac8ae4d7f425b8b42f6af55e6ea98"
PIN_EVAL_INFERENCE_FORWARD = "c854bd28802c372a9423a67255f3eb2f05e14af5e6d1319e8afa15fe67f89ea9"
PIN_TRAIN_MODE_TRAINING_FORWARD = "3c80ae6e914158b35011e6c4d9681d0524daecc4edcff496e3f945b171cdfbb4"


def _disable_dropout(model):
    """Switch off every dropout in the model, and report how many were found.

    Two kinds exist. `nn.Dropout` modules are the obvious ones. `GATConv` instead keeps its
    attention dropout as a plain float attribute, which no module sweep reaches — missing
    it is why the first attempt at the train-mode pin below was not reproducible.
    """
    modules = attributes = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            modules += 1
        elif isinstance(getattr(module, "dropout", None), float) and module.dropout:
            module.dropout = 0.0
            attributes += 1
    return modules, attributes


def _digest(tensor):
    """SHA-256 of a tensor's exact float32 bytes."""
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _pinned_model():
    """The deployed architecture carrying the committed untrained weights."""
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    state = torch.load(UNTRAINED_CHECKPOINT, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state)
    return model


def _pinned_training_inputs():
    """Seeded training-branch inputs. Cells 2 and 3 are isolated, standing in for padding."""
    b, t, n, crop = (PIN_BATCH[k] for k in ("batch", "frames", "cells", "crop"))
    gen = torch.Generator().manual_seed(1234)
    appearances = torch.rand((b, t, n, crop, crop, 1), generator=gen)
    morphologies = torch.rand((b, t, n, 3), generator=gen)
    centroids = torch.rand((b, t, n, 2), generator=gen) * 100.0
    adjacency = torch.zeros((b, t, n, n))
    adjacency[:, :, 0, 1] = adjacency[:, :, 1, 0] = 1.0
    return appearances, morphologies, centroids, adjacency


def _pinned_inference_inputs():
    """Seeded inference-branch inputs: three tracks with four frames of history, two detections."""
    gen = torch.Generator().manual_seed(5678)
    current_embeddings = torch.rand((1, 4, 3, 64), generator=gen)
    current_centroids = torch.rand((1, 4, 3, 2), generator=gen) * 100.0
    future_embeddings = torch.rand((1, 1, 2, 64), generator=gen)
    future_centroids = torch.rand((1, 1, 2, 2), generator=gen) * 100.0
    return current_embeddings, current_centroids, future_embeddings, future_centroids


def _assert_pinned(tensor, expected, label):
    actual = _digest(tensor)
    assert actual == expected, (
        f"{label} changed.\n"
        f"  expected sha256 {expected}\n"
        f"  actual   sha256 {actual}\n"
        f"  first values    {[float(v) for v in tensor.detach().flatten()[:4]]}\n"
        "A failure here means a code change altered model behavior, or a torch upgrade "
        "moved the last bits of a float. Do not regenerate without establishing which."
    )


def test_package_imports():
    """The package imports and reports a version."""
    import torch_tracking

    assert isinstance(torch_tracking.__version__, str)
    assert torch_tracking.__version__


def test_model_constructs_with_hardcoded_architecture():
    """GNNTrackingModel builds with the exact architecture CellTracker hardcodes.

    `CellTracker.__init__` constructs its model with these values and then loads
    whatever state dict it is handed into it, so a checkpoint trained with different
    hyperparameters cannot be loaded. This test is the tripwire for that coupling
    drifting apart.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    assert model is not None

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "model constructed with no parameters"


def test_training_forward_output_is_pinned():
    """`training_forward` in eval mode is byte-identical to the pinned output.

    Changes to how the graph's edge set is built, to construction, to feature widths or to
    unused state must not move this tensor.
    """
    model = _pinned_model().eval()

    with torch.no_grad():
        output = model.training_forward(*_pinned_training_inputs())

    assert output.shape == (1, 2, 4, 4, 3)
    _assert_pinned(output, PIN_EVAL_TRAINING_FORWARD, "training_forward (eval mode)")


def test_inference_forward_output_is_pinned():
    """`inference_forward` is byte-identical to the pinned output.

    This is the path that produces published tracking results, so the same inputs must
    give the same answer.
    """
    model = _pinned_model().eval()

    with torch.no_grad():
        output = model.inference_forward(*_pinned_inference_inputs())

    assert output.shape == (1, 1, 3, 2, 3)
    _assert_pinned(output, PIN_EVAL_INFERENCE_FORWARD, "inference_forward")


def test_training_mode_output_is_pinned():
    """`training_forward` in train mode, with dropout disabled, is byte-identical.

    Eval mode normalizes with the running statistics, so it cannot see a change to how
    BatchNorm aggregates a batch. Train mode can. This is the pin that would catch an
    unintended change to how BatchNorm pools its statistics: such a change moves this
    tensor and leaves the two eval-mode pins untouched.

    Dropout is switched off rather than seeded so the comparison does not depend on how
    many random numbers the layers happen to draw.
    """
    model = _pinned_model()
    modules, attributes = _disable_dropout(model)
    assert (modules, attributes) == (4, 4), (
        "the number of dropout sites changed; the pin below is only reproducible with "
        "every one of them disabled"
    )
    model.train()

    with torch.no_grad():
        output = model.training_forward(*_pinned_training_inputs())

    _assert_pinned(
        output, PIN_TRAIN_MODE_TRAINING_FORWARD, "training_forward (train mode)"
    )


def _padded_training_inputs():
    """`_pinned_training_inputs` with cells 2 and 3 made genuine padding slots.

    `_pinned_training_inputs` gives all four cells random positive morphologies, so under
    the rule the repository actually uses - a padding slot is an all-zero morphology - cells
    2 and 3 there are *real* cells that merely happen to be isolated in the adjacency. The
    isolation, masking and padding-count tests need slots that really are padding, so this
    zeroes their morphology. The original fixture is left untouched so the three output
    digests above stay valid.
    """
    appearances, morphologies, centroids, adjacency = _pinned_training_inputs()
    morphologies = morphologies.clone()
    morphologies[:, :, 2:] = 0.0
    return appearances, morphologies, centroids, adjacency


def test_padded_fixture_marks_exactly_cells_two_and_three_as_padding():
    """Only the zeroed slots are padding, and the original fixture has none."""
    _, original, _, _ = _pinned_training_inputs()
    _, padded, _, _ = _padded_training_inputs()

    assert (original != 0).any(dim=-1).all(), "the original fixture should have no padding"

    is_real = (padded != 0).any(dim=-1)
    assert is_real[:, :, :2].all()
    assert not is_real[:, :, 2:].any()


#: `training_forward` in train mode with `appearance_norm=False`, every cell real and
#: dropout off, as raw values (shape (1, 2, 4, 4, 3), flattened).
#:
#: History. First recorded before padding was masked out of the model's statistics (digest
#: of those exact bytes 5040ca83...28eb). It was re-checked after the masking, self-loop and
#: compaction changes and held to within 1e-7, which is the evidence that those are no-ops
#: when nothing is padding. It was then RE-TAKEN once, after the `TemporalMerge` fix,
#: because that changes any batch of more than one track and so moves this output too; the
#: values below are the ones after that fix.
#:
#: Masking padding out of the statistics, and refusing padding a self-loop, are both no-ops
#: when nothing in the batch is padding, so these values cannot move; and with appearance
#: normalization off, the whole-image crop normalization cannot reach them either. The other
#: pins above cannot show that, because the appearance normalization moves them regardless.
#:
#: It is compared within 1e-6 rather than by digest. Masking changes the *order* in which
#: a BatchNorm sums its rows (a flat list of real rows instead of a strided channel view),
#: and float addition is not associative, so the result moves by about 1e-7 in the last
#: bit even when the arithmetic is the same. A real change to which values are pooled moves
#: the output by orders of magnitude more than that, so the tolerance cannot hide one.
PIN_NO_APPEARANCE_NORM_ALL_REAL = [
    0.401696086, 0.289404362, 0.308899581, 0.369263589, 0.33018887, 0.30054763,
    0.515201449, 0.165965483, 0.318833053, 0.384329528, 0.220296681, 0.395373791,
    0.414872497, 0.298964173, 0.2861633, 0.37415722, 0.344707072, 0.281135619,
    0.622213125, 0.219575882, 0.158210963, 0.470424294, 0.306149572, 0.223426163,
    0.3725425, 0.26750955, 0.35994795, 0.3623555, 0.268621176, 0.369023293,
    0.396707356, 0.281297415, 0.321995199, 0.390737891, 0.202538401, 0.406723738,
    0.374044955, 0.245482817, 0.380472183, 0.370358437, 0.251775205, 0.377866387,
    0.465301186, 0.255466402, 0.279232353, 0.420400947, 0.270418555, 0.309180588,
    0.314867169, 0.459095985, 0.226036876, 0.331265926, 0.394010842, 0.274723291,
    0.425135583, 0.206899151, 0.367965251, 0.311430961, 0.303071916, 0.385497093,
    0.217863917, 0.462450862, 0.31968528, 0.37702322, 0.372137725, 0.250839025,
    0.369961739, 0.224954322, 0.405083954, 0.298263162, 0.25476411, 0.446972668,
    0.360865772, 0.385625809, 0.253508389, 0.362777293, 0.222253054, 0.414969653,
    0.396083862, 0.287874222, 0.316041917, 0.272539824, 0.249767348, 0.477692783,
    0.331009686, 0.433943361, 0.235046923, 0.373734027, 0.31213215, 0.314133823,
    0.423657745, 0.241160169, 0.335182101, 0.296061784, 0.267203987, 0.436734259,
]


def _model_without_appearance_norm():
    """The pinned model rebuilt with appearance normalization off, same weights."""
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH, appearance_norm=False)
    state = torch.load(UNTRAINED_CHECKPOINT, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state)
    return model


def test_all_real_batch_without_appearance_norm_is_pinned():
    """With nothing padding and no appearance normalization, the output must not move.

    Masking padding out of the statistics and refusing padding a self-loop are no-ops when
    every slot is a real cell. See the comment on `PIN_NO_APPEARANCE_NORM_ALL_REAL`.
    """
    model = _model_without_appearance_norm()
    _disable_dropout(model)
    model.train()

    with torch.no_grad():
        output = model.training_forward(*_pinned_training_inputs())

    expected = torch.tensor(PIN_NO_APPEARANCE_NORM_ALL_REAL).reshape(output.shape)
    worst = (output - expected).abs().max().item()
    assert worst <= 1e-6, (
        f"training_forward with every cell real and no appearance normalization moved by "
        f"{worst:.3g}. Masking and the self-loop change must be no-ops when nothing is "
        "padding, so this means one of them changed real cells' values."
    )


def test_normalized_crop_has_whole_image_statistics():
    """A crop is normalized over its full spatial extent.

    An earlier version normalized it one image row at a time, because the layer ignored the layout
    it was told about and took its statistics over (width, channels).
    """
    from torch_tracking.layers import ImageNormalization2D

    gen = torch.Generator().manual_seed(7)
    crop = torch.rand((2, 32, 32, 1), generator=gen)

    out = ImageNormalization2D(norm_method="whole_image", data_format="channels_last")(crop)

    assert torch.allclose(out.mean(dim=(1, 2)), torch.zeros(2, 1), atol=1e-5)
    assert torch.allclose(out.std(dim=(1, 2)), torch.ones(2, 1), atol=1e-4)


def test_normalizing_a_crop_does_not_flatten_its_rows():
    """Rows with different means stay different; the relative differences survive."""
    from torch_tracking.layers import ImageNormalization2D

    gen = torch.Generator().manual_seed(7)
    crop = torch.rand((2, 32, 32, 1), generator=gen)
    crop = crop + torch.linspace(0, 5, 32).view(1, 32, 1, 1)  # rows differ in brightness

    out = ImageNormalization2D(norm_method="whole_image", data_format="channels_last")(crop)

    row_means = out.mean(dim=2)  # (2, 32, 1)
    assert not torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-2), (
        "every row came back zero-mean, which is the row-wise behavior"
    )
    # Brightness still rises down the image by the same amount, up to one global scale.
    raw_means = crop.mean(dim=2)
    scale = out.std(dim=(1, 2), keepdim=True)  # == 1, kept for clarity
    ratio = (row_means[:, 1:] - row_means[:, :-1]) / (raw_means[:, 1:] - raw_means[:, :-1])
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), rtol=1e-2)
    assert torch.allclose(scale, torch.ones_like(scale), atol=1e-4)


def test_declared_layout_is_the_layout_used():
    """The same crop under each layout, each declared correctly, agrees up to transposition."""
    from torch_tracking.layers import ImageNormalization2D

    gen = torch.Generator().manual_seed(8)
    last = torch.rand((3, 16, 16, 2), generator=gen)  # (B, H, W, C)
    first = last.permute(0, 3, 1, 2)  # (B, C, H, W)

    for method in ("whole_image", "std", "max"):
        out_last = ImageNormalization2D(norm_method=method, data_format="channels_last")(last)
        out_first = ImageNormalization2D(norm_method=method, data_format="channels_first")(first)
        assert torch.allclose(out_last.permute(0, 3, 1, 2), out_first, atol=1e-6), method


def test_unsupported_image_layout_is_refused():
    """An unknown layout raises and names the two it accepts; it does not fall back."""
    from torch_tracking.layers import ImageNormalization2D

    with pytest.raises(ValueError, match="channels_first.*channels_last"):
        ImageNormalization2D(norm_method="whole_image", data_format="NHWC")

    assert not hasattr(ImageNormalization2D("whole_image"), "channel_axis")


def _encode_real_cells(pad_slots, *, seed=11, real=3):
    """Neighborhood-encode `real` cells beside `pad_slots` all-zero slots, in train mode.

    Returns the features of the real cells only. A fresh copy of the model is used each
    time because train mode updates running statistics in place. The real cells are drawn
    from the same generator regardless of `pad_slots`, so two calls differ only in padding.
    """
    import copy

    model = _pinned_model()
    _disable_dropout(model)
    model.train()
    encoder = copy.deepcopy(model.neighborhood_encoder)

    bt, crop = 2, 32
    n = real + pad_slots
    gen = torch.Generator().manual_seed(seed)
    appearance = torch.zeros((bt, n, crop, crop, 1))
    morphology = torch.zeros((bt, n, 3))
    centroid = torch.zeros((bt, n, 2))
    appearance[:, :real] = torch.rand((bt, real, crop, crop, 1), generator=gen)
    morphology[:, :real] = torch.rand((bt, real, 3), generator=gen) + 0.1
    centroid[:, :real] = torch.rand((bt, real, 2), generator=gen) * 100.0

    adjacency = torch.zeros((bt, n, n))
    adjacency[:, 0, 1] = adjacency[:, 1, 0] = 1.0
    adjacency[:, 1, 2] = adjacency[:, 2, 1] = 1.0

    with torch.no_grad():
        features, _ = encoder(appearance, morphology, centroid, adjacency)
    return features[:, :real]


def test_real_cell_features_do_not_move_with_padding_count():
    """A real cell's encoded features are the same however many padding slots surround it.

    In train mode every BatchNorm used to compute its mean and variance over
    all slots, padding included, so the same three real cells were normalized differently
    depending on how many padding slots sat beside them.
    """
    reference = _encode_real_cells(pad_slots=0)

    for pad_slots in (1, 5, 20):
        assert torch.allclose(
            reference, _encode_real_cells(pad_slots=pad_slots), atol=1e-5
        ), f"{pad_slots} padding slots moved a real cell's features"


def test_per_cell_encoders_give_the_same_values_compacted_or_full_when_all_real():
    """Compacting real cells changes nothing by itself. Only padding changes the answer.

    With every cell real, the three per-cell encoders run on the compacted tensor must give
    what they give on the full one, so any change in output is attributable to excluding
    padding and not to the reshape.
    """
    from torch_tracking.encoders import encode_real_cells

    model = _pinned_model()
    model.train()
    appearances, morphologies, centroids, _ = _pinned_training_inputs()
    bt, n = 3, 4
    app = appearances.reshape(bt, n, 32, 32, 1)
    morph = morphologies.reshape(bt, n, 3)
    cent = centroids.reshape(bt, n, 2)
    is_real = torch.ones(bt, n, dtype=torch.bool)

    import copy

    neighborhood = model.neighborhood_encoder
    for name, x in (
        ("appearance_encoder", app),
        ("morphology_encoder", morph),
        ("centroid_encoder", cent),
    ):
        full_encoder = copy.deepcopy(getattr(neighborhood, name))
        compact_encoder = copy.deepcopy(getattr(neighborhood, name))
        with torch.no_grad():
            full = full_encoder(x)
            compact = encode_real_cells(compact_encoder, x, is_real)
        assert torch.allclose(full, compact, atol=1e-5), name
        for (key, a), (_, b) in zip(
            full_encoder.state_dict().items(), compact_encoder.state_dict().items()
        ):
            assert torch.allclose(a.float(), b.float(), atol=1e-5), f"{name}.{key} buffer"


def test_masked_apply_degenerate_batches():
    """One real row does not raise; no real row leaves the running buffers untouched."""
    from torch_tracking.utils import masked_apply

    norm = nn.BatchNorm1d(4).train()
    x = torch.randn(2, 5, 4)

    one_real = torch.zeros(2, 5, dtype=torch.bool)
    one_real[0, 0] = True
    out = masked_apply(norm, x, one_real)  # must not raise
    assert out.shape == x.shape

    norm = nn.BatchNorm1d(4).train()
    before = {k: v.clone() for k, v in norm.state_dict().items()}
    out = masked_apply(norm, x, torch.zeros(2, 5, dtype=torch.bool))
    assert (out == 0).all() and out.shape == x.shape
    for key, value in norm.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} changed with no real row"


def test_masked_apply_matches_the_stock_module_on_real_rows():
    """Real rows are normalized exactly as the stock layer normalizes them alone."""
    from torch_tracking.utils import masked_apply

    x = torch.randn(3, 6, 4)
    is_real = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]]).bool()

    masked, stock = nn.BatchNorm1d(4).train(), nn.BatchNorm1d(4).train()
    out = masked_apply(masked, x, is_real)
    reference = stock(x[is_real])

    assert torch.allclose(out[is_real], reference, atol=1e-6)
    assert (out[~is_real] == 0).all()
    for key, value in masked.state_dict().items():
        assert torch.allclose(value, stock.state_dict()[key], atol=1e-6), key


# --------------------------------------------------------------------------------------
# The one definition of "padding": a slot whose whole morphology is zero.
# --------------------------------------------------------------------------------------


def test_real_cell_mask_rule():
    """A slot is real unless its whole morphology is zero; works for arrays and tensors."""
    import numpy as np

    from torch_tracking.utils import real_cell_mask

    rows = [[0.0, 0.0, 0.0], [12.0, 3.0, 0.5], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]

    expected = [False, True, True, False]
    assert real_cell_mask(torch.tensor(rows)).tolist() == expected
    assert real_cell_mask(np.array(rows)).tolist() == expected

    assert not real_cell_mask(torch.zeros(2, 5, 3)).any(), "all-padding input has no real cell"
    assert real_cell_mask(torch.ones(2, 5, 3)).all()
    assert real_cell_mask(torch.zeros(2, 5, 3)).shape == (2, 5)


def test_real_cell_mask_ignores_the_centroid():
    """The centroid cannot identify padding: augmentation translates padding rows too.

    A padding slot carries a zero morphology wherever its centroid has been moved to, and
    a real cell sitting at the origin is still real.
    """
    from torch_tracking.utils import real_cell_mask

    morphologies = torch.tensor([[0.0, 0.0, 0.0], [9.0, 2.0, 0.3]])
    # Two possible centroids per slot: the stored origin, and an augmented offset.
    for centroids in (torch.zeros(2, 2), torch.tensor([[300.0, -200.0], [0.0, 0.0]])):
        assert centroids.shape == (2, 2)
        assert real_cell_mask(morphologies).tolist() == [False, True]


def test_real_cell_mask_identifies_cells_two_and_three_of_the_padded_fixture():
    """Exactly the zeroed slots, and nothing in the original fixture."""
    from torch_tracking.utils import real_cell_mask

    _, original, _, _ = _pinned_training_inputs()
    _, padded, _, _ = _padded_training_inputs()

    assert real_cell_mask(original).all()
    mask = real_cell_mask(padded)
    assert mask[:, :, :2].all()
    assert not mask[:, :, 2:].any()


def test_spatial_adjacency_uses_the_shared_rule():
    """The graph rule excludes exactly the slots `real_cell_mask` calls padding."""
    import numpy as np

    from torch_tracking.utils import real_cell_mask, spatial_adjacency

    centroids = np.array([[10.0, 10.0], [20.0, 10.0], [12.0, 11.0], [0.0, 0.0]])
    morphologies = np.array([[5.0, 2.0, 1.0], [6.0, 2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    adjacency = spatial_adjacency(centroids, morphologies, distance_threshold=50)
    real = real_cell_mask(morphologies)

    assert adjacency[0, 1] and adjacency[1, 0]
    assert not adjacency[~real].any() and not adjacency[:, ~real].any()


def _capture_edge_index(model, inputs):
    """Record the `edge_index` the first graph layer is actually handed.

    Observed by wrapping the layer rather than recomputing the adjacency, so the test
    cannot agree with a buggy implementation by repeating it.
    """
    captured = []
    layer = model.neighborhood_encoder.graph_layers[0]
    original = layer.forward

    def spy(x, edge_index, *args, **kwargs):
        captured.append(edge_index.detach().clone())
        return original(x, edge_index, *args, **kwargs)

    layer.forward = spy
    try:
        with torch.no_grad():
            model.training_forward(*inputs)
    finally:
        layer.forward = original

    assert captured, "the first graph layer was never called"
    return captured[0]


def test_graph_edge_set_is_pinned():
    """The edge set the graph layers receive is pinned exactly.

    Self-loops are added by an explicit step rather than as a side effect of a
    `D^-1/2 A D^-1/2` normalization whose weights were then discarded. That replacement
    must not move the resulting edge set, so it is pinned exactly rather than described.

    The input has four cells per frame: 0 and 1 are adjacent real cells, 2 and 3 are
    isolated and stand in for padding slots. Three frames are batched into one graph, so
    node indices are offset by four per frame.
    """
    model = _pinned_model().eval()
    edge_index = _capture_edge_index(model, _pinned_training_inputs())

    edges = sorted(map(tuple, edge_index.t().tolist()))
    per_frame = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3)]
    expected = sorted(
        (a + 4 * frame, b + 4 * frame) for frame in range(3) for a, b in per_frame
    )

    assert edges == expected, (
        f"the graph's edge set changed.\n  expected {expected}\n  actual   {edges}"
    )


def test_padding_slots_are_isolated_in_the_graph():
    """A padding slot has no edge at all.

    An earlier version gave it a self-loop like every other slot. Only real cells get one,
    so every edge the graph layers receive has a real cell at both ends. Cells 2 and 3 of the
    padded fixture are genuine padding; cells 0 and 1 are adjacent real cells.
    """
    model = _pinned_model().eval()
    edge_index = _capture_edge_index(model, _padded_training_inputs())

    edges = sorted(map(tuple, edge_index.t().tolist()))
    per_frame = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected = sorted(
        (a + 4 * frame, b + 4 * frame) for frame in range(3) for a, b in per_frame
    )
    assert edges == expected, f"expected {expected}\n  actual   {edges}"

    padding_nodes = {n + 4 * frame for frame in range(3) for n in (2, 3)}
    touched = {node for edge in edges for node in edge}
    assert not (padding_nodes & touched), "a padding slot still appears in the edge set"


def test_isolated_slots_do_not_poison_the_graph_output():
    """Isolated slots leave real cells' output finite and independent of their number."""
    outputs = [_encode_real_cells(pad_slots=k) for k in (1, 4, 9)]

    for out in outputs:
        assert torch.isfinite(out).all()
    for out in outputs[1:]:
        assert torch.allclose(outputs[0], out, atol=1e-5), (
            "a real cell's graph output depends on how many isolated slots the batch has"
        )


def test_add_self_loops_gives_loops_to_real_cells_only():
    """Over a four-slot matrix, real cells keep self-loops and padding slots gain none."""
    from torch_tracking.utils import add_self_loops_dense

    adj = torch.zeros(1, 4, 4)
    adj[:, 0, 1] = adj[:, 1, 0] = 1.0
    is_real = torch.tensor([[True, True, True, False]])

    looped = add_self_loops_dense(adj, is_real)

    assert torch.diagonal(looped, dim1=-2, dim2=-1).tolist() == [[1.0, 1.0, 1.0, 0.0]]
    assert looped[0, 0, 1] == 1.0 and looped[0, 1, 0] == 1.0
    assert (looped[0, 3] == 0).all() and (looped[0, :, 3] == 0).all()


# --------------------------------------------------------------------------------------
# Construction and validation contract.
# --------------------------------------------------------------------------------------


def test_default_configuration_constructs():
    """The constructor's own default builds a working model.

    It used to default to `channels_first`, which made `appearance_shape` compare
    `crop_size == 1` and raise. Every working caller passed `channels_last`, so the
    default was a trap for anyone who did not.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel()
    assert model.data_format == "channels_last"


def test_unsupported_data_format_is_refused_at_construction():
    """An unsupported layout fails at construction, not at the first forward pass."""
    from torch_tracking.model import GNNTrackingModel

    with pytest.raises(ValueError, match="channels_last"):
        GNNTrackingModel(**{**MODEL_ARCH, "data_format": "channels_first"})

    with pytest.raises(ValueError, match="channels_last"):
        GNNTrackingModel(**{**MODEL_ARCH, "data_format": "nonsense"})


def test_no_data_format_reaches_unbound_local_error():
    """No configuration leaves a comparison unbound.

    The validator used to bind its squareness comparison inside two `if` branches and
    then read it unconditionally, so any third value raised `UnboundLocalError` from the
    validator rather than a `ValueError` describing the problem.
    """
    from torch_tracking.model import GNNTrackingModel

    for value in ["channels_first", "NHWC", "", None, 3]:
        with pytest.raises(ValueError):
            GNNTrackingModel(**{**MODEL_ARCH, "data_format": value})


def test_shape_arity_is_checked_before_the_shape_is_indexed():
    """The arity message states the length actually required.

    An earlier version reported "should be length 4" for a length-3 requirement, and the
    check ran after two statements had already indexed the shape.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    assert len(model.appearance_shape) == 3

    broken = GNNTrackingModel.__new__(GNNTrackingModel)
    broken.appearance_shape = (32, 32)
    broken.data_format = "channels_last"
    broken.graph_layer = "gat"
    broken.norm_layer = "batch"

    with pytest.raises(ValueError, match="length 3"):
        broken._validate_config()


def test_non_power_of_two_crop_is_refused():
    """Spatial dimensions must be a power of two, and the message says so."""
    from torch_tracking.model import GNNTrackingModel

    with pytest.raises(ValueError, match="power of 2"):
        GNNTrackingModel(**{**MODEL_ARCH, "crop_size": 24})


def test_model_docstring_example_runs():
    """The `GNNTrackingModel` docstring example executes.

    It used to pass `max_cells` and `appearance_shape`, neither of which is a parameter,
    so copying it raised `TypeError`. A documented example that does not run is worse than
    no example: it costs the reader the time to find out.
    """
    import doctest

    import torch_tracking.model as model_module

    results = doctest.testmod(model_module, verbose=False, report=False)
    assert results.failed == 0, f"{results.failed} doctest(s) failed in model.py"
    assert results.attempted > 0, "no doctests ran; the example may have been removed"


# --------------------------------------------------------------------------------------
# Normalization selection.
# --------------------------------------------------------------------------------------


def test_norm_layer_is_honored_in_every_module():
    """Nothing builds a normalization other than the one requested.

    Three encoders accepted `norm_layer` and built `BatchNorm` regardless; `DeltaEncoder`
    stored the choice, built only `BatchNorm1d`, and its `layer` branch applied that layer
    to a rank-4 tensor and raised. The parameter is now narrowed to the one value every
    module can build, so "honored" and "not offered" are the only two states.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)

    assert model.norm_layer == "batch"
    layer_norms = [
        name for name, mod in model.named_modules() if isinstance(mod, nn.LayerNorm)
    ]
    assert not layer_norms, (
        f"norm_layer='batch' built LayerNorm layers at {layer_norms}"
    )


def test_unbuildable_norm_layer_is_refused_at_construction():
    """`norm_layer='layer'` is refused at construction rather than silently half-built.

    An earlier version built a mostly-BatchNorm model that failed on the first forward
    pass: the setting was neither honored nor refused, and the problem surfaced late.
    """
    from torch_tracking.model import GNNTrackingModel

    with pytest.raises(ValueError, match="norm_layer"):
        GNNTrackingModel(**{**MODEL_ARCH, "norm_layer": "layer"})


def test_encoders_no_longer_accept_a_normalization_they_cannot_build():
    """The modules that ignored `norm_layer` no longer advertise it.

    A setting should be either honored or not offered. These four could not honor it, so
    they no longer take it, and passing it is now a loud `TypeError` rather than silence.
    """
    from torch_tracking.encoders import (
        AppearanceEncoder,
        CentroidEncoder,
        DeltaEncoder,
        MorphologyEncoder,
    )

    for factory in (
        lambda **kw: AppearanceEncoder(appearance_shape=(32, 32, 1), **kw),
        lambda **kw: MorphologyEncoder(input_dim=3, **kw),
        lambda **kw: CentroidEncoder(input_dim=2, **kw),
        lambda **kw: DeltaEncoder(input_dim=2, **kw),
    ):
        with pytest.raises(TypeError):
            factory(norm_layer="layer")


def test_delta_encoder_normalizes_rank_four_and_rank_five_input():
    """`DeltaEncoder` handles both shapes it is actually given.

    Within-frame deltas arrive as (B, T, N, 2) and cross-frame deltas as (B, T, N, N, 2).
    The deleted `layer` branch raised on both.
    """
    from torch_tracking.encoders import DeltaEncoder

    encoder = DeltaEncoder(input_dim=2, encoder_dim=64).eval()

    with torch.no_grad():
        assert encoder(torch.zeros(2, 3, 4, 2)).shape == (2, 3, 4, 64)
        assert encoder(torch.zeros(2, 3, 4, 4, 2)).shape == (2, 3, 4, 4, 64)


def test_add_self_loops_dense_preserves_the_edge_set():
    """The self-loop helper reproduces exactly what the discarded normalizer left behind.

    The removed `normalize_adjacency_symmetric` computed `D^-1/2 A D^-1/2` and its only
    caller threw the weights away with `.nonzero()`. The replacement does the one thing
    that survived. This asserts the two agree on the sparsity pattern, which is all the
    graph layers ever saw.
    """
    from torch_tracking.utils import add_self_loops_dense

    adj = torch.zeros(2, 4, 4)
    adj[:, 0, 1] = adj[:, 1, 0] = 1.0

    looped = add_self_loops_dense(adj, torch.ones(2, 4, dtype=torch.bool))

    # Reference: what the deleted normalizer produced, written out longhand.
    identity = torch.eye(4).expand(2, 4, 4)
    with_loops = adj + identity
    degree = with_loops.sum(dim=-1)
    inv_sqrt = degree.pow(-0.5)
    normalized = inv_sqrt.unsqueeze(-1) * with_loops * inv_sqrt.unsqueeze(-2)

    assert torch.equal(looped.nonzero(), normalized.nonzero()), (
        "the replacement changed which entries are non-zero, which is the only thing "
        "the graph layers read"
    )
    assert torch.equal(torch.diagonal(looped, dim1=-2, dim2=-1), torch.ones(2, 4))


def test_removed_normalizer_is_gone():
    """`normalize_adjacency_symmetric` is deleted, not left behind with no callers.

    Its 4-D branch bound `B, T, N` and then built its identity from `T`, so it was wrong
    for the shape its own docstring advertised. Rather than correct a function nothing
    calls, it was removed.
    """
    import torch_tracking.utils as utils

    assert not hasattr(utils, "normalize_adjacency_symmetric")


# --------------------------------------------------------------------------------------
# Feature widths are derived, not assumed.
# --------------------------------------------------------------------------------------


def test_neighborhood_encoder_accepts_encoders_of_different_widths():
    """Differing encoder widths build and run.

    `combined_dim` used to be `3 * embedding_dim`, which is right only while the three encoders
    and the embedding all happen to be the same width. The final layer in the same class
    already summed the real widths, so the class disagreed with itself.
    """
    from torch_tracking.encoders import (
        AppearanceEncoder,
        CentroidEncoder,
        MorphologyEncoder,
        NeighborhoodEncoder,
    )

    appearance = AppearanceEncoder(appearance_shape=(32, 32, 1), encoder_dim=48)
    morphology = MorphologyEncoder(input_dim=3, encoder_dim=16)
    centroid = CentroidEncoder(input_dim=2, encoder_dim=32)

    encoder = NeighborhoodEncoder(
        appearance_encoder=appearance,
        morphology_encoder=morphology,
        centroid_encoder=centroid,
        n_filters=64,
        embedding_dim=64,
        n_layers=2,
        graph_layer="gat",
    ).eval()

    assert encoder.initial_dense.in_features == 48 + 16 + 32

    bt, n = 2, 4
    with torch.no_grad():
        features, centroids = encoder(
            torch.zeros(bt, n, 32, 32, 1),
            torch.zeros(bt, n, 3),
            torch.zeros(bt, n, 2),
            torch.zeros(bt, n, n),
        )
    assert features.shape == (bt, n, 64)
    assert centroids.shape == (bt, n, 2)


def test_decoder_accepts_deltas_narrower_than_embeddings():
    """The decoder sizes itself from both widths it receives.

    `input_dim` used to be `4 * embedding_dim`, which assumed the delta encoders were as
    wide as the embeddings - the same defect as in `NeighborhoodEncoder` above.
    """
    from torch_tracking.decoder import TrackingDecoder

    decoder = TrackingDecoder(embedding_dim=64, encoder_dim=16, n_filters=32).eval()

    assert decoder.dense1.in_features == 2 * 64 + 2 * 16

    with torch.no_grad():
        output = decoder(torch.zeros(1, 2, 3, 4, 128), torch.zeros(1, 2, 3, 4, 32))
    assert output.shape == (1, 2, 3, 4, 3)


def test_decoder_defaults_delta_width_to_embedding_width():
    """Omitting `encoder_dim` reproduces the previous sizing, so no caller is surprised."""
    from torch_tracking.decoder import TrackingDecoder

    assert TrackingDecoder(embedding_dim=64).dense1.in_features == 4 * 64


# --------------------------------------------------------------------------------------
# Checkpoint, device and crop size.
# --------------------------------------------------------------------------------------


def test_checkpoint_parameter_names_a_file():
    """`CellTracker` takes a checkpoint file path, and the parameter says so.

    The parameter was called `checkpoint_dir` and documented as "the directory of the
    model weight file", while being handed straight to `torch.load`. The name mattered
    more than usual here because `TrainingConfig.checkpoint_dir` *is* a directory, so one
    name meant two things in one package.
    """
    import inspect

    from torch_tracking.tracker import CellTracker

    signature = inspect.signature(CellTracker.__init__)
    assert "checkpoint_path" in signature.parameters
    assert list(signature.parameters)[1] == "checkpoint_path", (
        "checkpoint_path should stay the first real argument so positional callers work"
    )
    assert "Path to the checkpoint *file*" in (CellTracker.__doc__ or ""), (
        "the parameter's documentation should say it takes a file, not a directory"
    )


def test_deprecated_checkpoint_dir_still_loads(untrained_checkpoint):
    """The old spelling keeps working, loudly.

    `checkpoint_dir` is kept as a deprecated alias rather than removed, so existing user
    code that passes it keeps working.
    """
    from torch_tracking.tracker import CellTracker

    with pytest.warns(DeprecationWarning, match="checkpoint_dir"):
        tracker = CellTracker(
            checkpoint_dir=str(untrained_checkpoint), device="cpu", verbose=False
        )

    assert str(tracker.checkpoint_path) == str(untrained_checkpoint)


def test_both_checkpoint_spellings_together_is_an_error(untrained_checkpoint):
    """Passing both names is refused rather than silently preferring one."""
    from torch_tracking.tracker import CellTracker

    with pytest.raises(ValueError, match="not both"):
        CellTracker(
            checkpoint_path=str(untrained_checkpoint),
            checkpoint_dir=str(untrained_checkpoint),
            device="cpu",
            verbose=False,
        )


def test_device_defaults_to_cpu(untrained_checkpoint):
    """The documented default and the implemented default are both CPU.

    The docstring used to claim `cuda`. CPU is the correct default, because inference must
    work on machines without a usable GPU, so the documentation was what was wrong.
    """
    from torch_tracking.tracker import CellTracker

    tracker = CellTracker(
        checkpoint_path=str(untrained_checkpoint), verbose=False
    )

    assert tracker.device == "cpu"
    assert "Default is `cpu`" in CellTracker.__doc__


# --------------------------------------------------------------------------------------
# Dead state.
# --------------------------------------------------------------------------------------


def test_every_built_layer_is_reachable_from_a_forward_pass(untrained_checkpoint):
    """The model builds no layer that nothing uses.

    `model.py` used to build `unmerge_embeddings` and `unmerge_centroids` that nothing read;
    `TrainingBranch` builds and uses its own. Dead layers mislead a reader about what the
    architecture contains and about what a checkpoint has to supply.

    Reachability is checked by running both forward paths with every module instrumented,
    rather than by listing names, so reintroducing a dead module fails this without
    anyone having to remember to update a list.
    """
    model = _pinned_model().eval()

    called = set()
    handles = [
        module.register_forward_hook(
            lambda mod, inp, out, name=name: called.add(name)
        )
        for name, module in model.named_modules()
        if name and not list(module.children())
    ]
    try:
        with torch.no_grad():
            model.training_forward(*_pinned_training_inputs())
            model.inference_forward(*_pinned_inference_inputs())
    finally:
        for handle in handles:
            handle.remove()

    leaves = {
        name
        for name, module in model.named_modules()
        if name and not list(module.children())
    }
    unreachable = leaves - called

    # Softmax is skipped when logits are requested, so it is legitimately conditional.
    unreachable -= {"tracking_decoder.softmax"}

    assert not unreachable, (
        f"these layers are built but never used: {sorted(unreachable)}"
    )


def test_removing_dead_layers_did_not_change_the_checkpoint_contract(untrained_checkpoint):
    """The committed checkpoint still loads strictly.

    The removed `Unmerge` layers hold no parameters, so the state dict is unaffected, but
    that is worth asserting rather than assuming.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    state = torch.load(untrained_checkpoint, map_location="cpu")["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected


# --------------------------------------------------------------------------------------
# Module shape contracts.
#
# These replace the `__main__` demo blocks in layers.py, branches.py, decoder.py and
# encoders.py, which were the de-facto shape checks for modules that had no tests. Every
# one of them raised: `Unmerge` was constructed with three arguments it does not take,
# `TrainingBranch` and `InferenceBranch` with parameters removed in an earlier refactor,
# and `AppearanceEncoder` with a channels-first shape. Repairing a block nothing runs
# would leave the same gap, so the checks moved here, where they run.
# --------------------------------------------------------------------------------------


def test_layer_shapes():
    """`layers.py`'s shape contracts, from its former `__main__` block."""
    from torch_tracking.layers import (
        Comparison,
        DeltaReshape,
        ImageNormalization2D,
        TemporalMerge,
        Unmerge,
        compute_deltas,
        compute_deltas_across_frames,
    )

    with torch.no_grad():
        assert ImageNormalization2D(norm_method="whole_image")(
            torch.randn(2, 3, 32, 32)
        ).shape == (2, 3, 32, 32)

        assert Comparison()(
            torch.randn(2, 4, 5, 64), torch.randn(2, 4, 3, 64)
        ).shape == (2, 4, 5, 3, 128)

        assert DeltaReshape()(
            torch.randn(2, 4, 5, 64), torch.randn(2, 4, 3, 64)
        ).shape == (2, 4, 5, 3, 64)

        # Unmerge takes no constructor arguments; the demo passed it three.
        assert Unmerge()(torch.randn(2, 8 * 39, 64), 2, 8, 39).shape == (2, 8, 39, 64)

        assert TemporalMerge(encoder_dim=64)(
            torch.randn(2, 8, 39, 64)
        ).shape == (2, 8, 39, 64)

        assert compute_deltas(torch.randn(2, 8, 39, 2)).shape == (2, 8, 39, 2)
        assert compute_deltas_across_frames(
            torch.randn(2, 8, 39, 2)
        ).shape == (2, 7, 39, 39, 2)


def test_encoder_shapes():
    """`encoders.py`'s shape contracts, from its former `__main__` block.

    The demo built `AppearanceEncoder(appearance_shape=(1, 32, 32))` — a channels-first
    shape — which made the encoder derive `log2(1) == 0` convolution blocks and then fail
    on a four-dimensional input it expected to be five-dimensional.
    """
    from torch_tracking.encoders import (
        AppearanceEncoder,
        CentroidEncoder,
        DeltaEncoder,
        MorphologyEncoder,
        NeighborhoodEncoder,
    )

    bt, n = 2 * 8, 39

    appearance = AppearanceEncoder(appearance_shape=(32, 32, 1)).eval()
    morphology = MorphologyEncoder(input_dim=3, encoder_dim=64).eval()
    centroid = CentroidEncoder(input_dim=2, encoder_dim=64).eval()

    with torch.no_grad():
        assert appearance(torch.randn(bt, n, 32, 32, 1)).shape == (bt, n, 64)
        assert morphology(torch.randn(bt, n, 3)).shape == (bt, n, 64)
        assert centroid(torch.randn(bt, n, 2)).shape == (bt, n, 64)

        encoder = NeighborhoodEncoder(
            appearance_encoder=appearance,
            morphology_encoder=morphology,
            centroid_encoder=centroid,
            graph_layer="gat",
            n_layers=2,
        ).eval()
        # A binary adjacency, not the `torch.randn` the demo passed: the graph is a
        # sparsity pattern, and random floats make every node adjacent to every other.
        adjacency = torch.zeros(bt, n, n)
        adjacency[:, 0, 1] = adjacency[:, 1, 0] = 1.0
        features, centroids = encoder(
            torch.randn(bt, n, 32, 32, 1),
            torch.randn(bt, n, 3),
            torch.randn(bt, n, 2),
            adjacency,
        )
        assert features.shape == (bt, n, 64)
        assert centroids.shape == (bt, n, 2)

        assert DeltaEncoder(input_dim=2, encoder_dim=64).eval()(
            torch.randn(16, n, 2)
        ).shape == (16, n, 64)


def test_branch_shapes():
    """`branches.py`'s shape contracts, from its former `__main__` block.

    The demo passed `max_cells`, `embedding_dim` and `encoder_dim` to both branches; none
    is a parameter of either, so both constructions raised `TypeError`.
    """
    from torch_tracking.branches import InferenceBranch, TrainingBranch
    from torch_tracking.encoders import (
        AppearanceEncoder,
        CentroidEncoder,
        DeltaEncoder,
        MorphologyEncoder,
        NeighborhoodEncoder,
    )
    from torch_tracking.layers import TemporalMerge

    batch, track_length, max_cells, dim = 2, 8, 6, 64

    neighborhood = NeighborhoodEncoder(
        appearance_encoder=AppearanceEncoder(appearance_shape=(32, 32, 1)),
        morphology_encoder=MorphologyEncoder(input_dim=3),
        centroid_encoder=CentroidEncoder(input_dim=2),
        graph_layer="gat",
        n_layers=2,
    )
    shared = dict(
        embedding_temporal_merge=TemporalMerge(dim),
        delta_temporal_merge=TemporalMerge(dim),
        delta_encoder=DeltaEncoder(input_dim=2, encoder_dim=dim),
        delta_across_frames_encoder=DeltaEncoder(input_dim=2, encoder_dim=dim),
    )

    training = TrainingBranch(
        neighborhood_encoder=neighborhood, track_length=track_length, **shared
    ).eval()

    adjacency = torch.zeros(batch, track_length, max_cells, max_cells)
    adjacency[:, :, 0, 1] = adjacency[:, :, 1, 0] = 1.0

    with torch.no_grad():
        comparisons, deltas, pair_mask = training(
            torch.randn(batch, track_length, max_cells, 32, 32, 1),
            torch.randn(batch, track_length, max_cells, 3),
            torch.randn(batch, track_length, max_cells, 2),
            adjacency,
        )
    assert comparisons.shape == (batch, track_length - 1, max_cells, max_cells, 2 * dim)
    assert deltas.shape == (batch, track_length - 1, max_cells, max_cells, 2 * dim)
    assert pair_mask.shape == (batch, track_length - 1, max_cells, max_cells)

    inference = InferenceBranch(**shared).eval()
    tracks, detections, history = 10, 15, 5
    with torch.no_grad():
        comparisons, deltas = inference(
            torch.randn(batch, history, tracks, dim),
            torch.randn(batch, history, tracks, 2),
            torch.randn(batch, 1, detections, dim),
            torch.randn(batch, 1, detections, 2),
        )
    assert comparisons.shape == (batch, 1, tracks, detections, 2 * dim)
    assert deltas.shape == (batch, 1, tracks, detections, 2 * dim)


def test_decoder_shapes():
    """`decoder.py`'s shape contracts, from its former `__main__` block."""
    from torch_tracking.decoder import TrackingDecoder

    decoder = TrackingDecoder(embedding_dim=64, n_filters=64, n_classes=3).eval()

    with torch.no_grad():
        probabilities = decoder(
            torch.randn(2, 7, 39, 39, 128), torch.randn(2, 7, 39, 39, 128)
        )
        assert probabilities.shape == (2, 7, 39, 39, 3)
        assert probabilities[0, 0, 0, 0].sum().item() == pytest.approx(1.0)

        logits = decoder(
            torch.randn(2, 7, 39, 39, 128),
            torch.randn(2, 7, 39, 39, 128),
            apply_softmax=False,
        )
        assert logits.shape == (2, 7, 39, 39, 3)
        assert logits[0, 0, 0, 0].sum().item() != pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Padding does not move what a real cell is scored as.
# --------------------------------------------------------------------------------------


def test_every_normalization_in_the_neighborhood_encoder_sees_real_cells_only():
    """`initial_norm`, both `graph_norms` and `final_norm` are each masked.

    Observed by hooking each layer and counting the rows it is handed, so a layer left
    unmasked fails here by name rather than being averaged away in an output comparison.
    """
    import copy

    model = _pinned_model()
    _disable_dropout(model)
    model.train()
    encoder = copy.deepcopy(model.neighborhood_encoder)

    seen = {}
    norms = {"initial_norm": encoder.initial_norm, "final_norm": encoder.final_norm}
    for i, norm in enumerate(encoder.graph_norms):
        norms[f"graph_norms.{i}"] = norm
    assert len(norms) == 4
    for name, norm in norms.items():
        norm.register_forward_hook(
            lambda mod, inp, out, name=name: seen.__setitem__(name, inp[0].shape[0])
        )

    bt, real, pad = 2, 3, 4
    n = real + pad
    gen = torch.Generator().manual_seed(3)
    appearance = torch.zeros((bt, n, 32, 32, 1))
    morphology = torch.zeros((bt, n, 3))
    centroid = torch.zeros((bt, n, 2))
    appearance[:, :real] = torch.rand((bt, real, 32, 32, 1), generator=gen)
    morphology[:, :real] = torch.rand((bt, real, 3), generator=gen) + 0.1
    centroid[:, :real] = torch.rand((bt, real, 2), generator=gen) * 50
    adjacency = torch.zeros((bt, n, n))
    adjacency[:, 0, 1] = adjacency[:, 1, 0] = 1.0

    with torch.no_grad():
        encoder(appearance, morphology, centroid, adjacency)

    assert seen == {name: bt * real for name in norms}, seen


def test_pair_masks_have_the_expected_shapes_and_exclude_padding_ends():
    """Rank-4 and rank-5 masks, and a pair with a padding end is excluded."""
    from torch_tracking.utils import pair_masks

    # One movie, three frames, three slots. Slot 2 is born in frame 1; slot 1 ends after 1.
    is_real = torch.tensor([[[1, 1, 0], [1, 0, 1], [1, 0, 1]]]).bool()
    within, across = pair_masks(is_real)

    assert within.shape == (1, 3, 3)
    assert across.shape == (1, 2, 3, 3)

    # across[t, i, j]: slot i real in frame t AND slot j real in frame t+1.
    assert across[0, 0].tolist() == [
        [True, False, True],
        [True, False, True],
        [False, False, False],
    ]
    assert across[0, 1].tolist() == [
        [True, False, True],
        [False, False, False],
        [True, False, True],
    ]
    # within: the first frame has no predecessor, so only the slot's own state counts.
    assert within[0, 0].tolist() == [True, True, False]
    assert within[0, 1].tolist() == [True, False, False]  # slot 2 is new: no real predecessor
    assert within[0, 2].tolist() == [True, False, True]


def test_pair_mask_agrees_with_the_entries_the_loss_drops():
    """The mask built from morphology is the set of entries the targets leave graded.

    Two independent computations decide what "padding pair" means: the model's mask,
    from all-zero morphology, and the dataset's targets, from the lineage, where `-1`
    is what `TrackingLoss` drops. They must be the same set.
    """
    import numpy as np

    from torch_tracking.loader import get_temporal_adjacency
    from torch_tracking.utils import pair_masks, real_cell_mask

    frames = 5
    lineage = {
        1: {"label": 1, "frames": [0, 1, 2, 3, 4], "daughters": []},
        2: {"label": 2, "frames": [1, 2, 3, 4], "daughters": []},  # born in frame 1
        3: {"label": 3, "frames": [0, 1, 2], "daughters": []},  # ends after frame 2
    }
    max_cells = 5
    targets = torch.from_numpy(
        get_temporal_adjacency(lineage, 0, frames, max_cells).astype(np.int64)
    )

    # A cell's features are non-zero exactly in the frames it is present.
    morphologies = torch.zeros((1, frames, max_cells, 3))
    for attr in lineage.values():
        morphologies[0, attr["frames"], attr["label"] - 1] = 1.0

    _, across = pair_masks(real_cell_mask(morphologies))

    assert across.shape == (1,) + tuple(targets.shape)
    assert torch.equal(across[0], targets >= 0)


def _padded_batch(pad_slots, *, real=3, frames=3, seed=21):
    """Real cells beside `pad_slots` padding slots, for the whole training forward pass.

    The real cells are drawn from the same generator whatever `pad_slots` is, so two
    batches differ only in padding. Real cell 0 and 1, and 1 and 2, are adjacent.
    """
    n = real + pad_slots
    gen = torch.Generator().manual_seed(seed)
    appearances = torch.zeros((1, frames, n, 32, 32, 1))
    morphologies = torch.zeros((1, frames, n, 3))
    centroids = torch.zeros((1, frames, n, 2))
    appearances[:, :, :real] = torch.rand((1, frames, real, 32, 32, 1), generator=gen)
    morphologies[:, :, :real] = torch.rand((1, frames, real, 3), generator=gen) + 0.1
    centroids[:, :, :real] = torch.rand((1, frames, real, 2), generator=gen) * 60.0
    adjacency = torch.zeros((1, frames, n, n))
    adjacency[:, :, 0, 1] = adjacency[:, :, 1, 0] = 1.0
    adjacency[:, :, 1, 2] = adjacency[:, :, 2, 1] = 1.0
    return appearances, morphologies, centroids, adjacency


def _train_forward(pad_slots):
    """One train-mode forward pass. Returns the output and the model afterwards."""
    model = _pinned_model()
    _disable_dropout(model)
    model.train()
    with torch.no_grad():
        output = model.training_forward(*_padded_batch(pad_slots))
    return output, model


def test_padding_does_not_move_the_models_graded_output():
    """Every entry the loss grades is the same with 0, 3 or 17 padding slots beside it."""
    reference, _ = _train_forward(0)

    for pad_slots in (3, 17):
        output, _ = _train_forward(pad_slots)
        graded = output[:, :, :3, :3]
        assert torch.isfinite(graded).all()
        assert torch.allclose(reference, graded, atol=1e-5), (
            f"{pad_slots} padding slots moved the graded output by "
            f"{(reference - graded).abs().max():.3g}"
        )


def test_running_statistics_are_accumulated_over_real_cells_only():
    """Padded batches leave every normalization buffer exactly as an unpadded one would."""
    _, unpadded = _train_forward(0)
    _, padded = _train_forward(14)

    checked = 0
    for (name, a), (_, b) in zip(unpadded.named_buffers(), padded.named_buffers()):
        assert torch.allclose(a.float(), b.float(), atol=1e-5), name
        checked += 1
    assert checked > 20, "expected the model's BatchNorm buffers to have been compared"


def test_a_batch_of_only_padding_completes_and_updates_no_statistic():
    """No real cell: the forward pass does not raise and no running statistic moves."""
    model = _pinned_model()
    _disable_dropout(model)
    model.train()
    before = {k: v.clone() for k, v in model.named_buffers()}

    appearances, morphologies, centroids, adjacency = _padded_batch(5, real=0)
    with torch.no_grad():
        output = model.training_forward(appearances, morphologies, centroids, adjacency)

    assert output.shape == (1, 2, 5, 5, 3)
    for name, value in model.named_buffers():
        assert torch.equal(value, before[name]), f"{name} was updated by a padding-only batch"


def test_checkpoint_layout_is_unchanged_by_the_normalization_correction():
    """The committed checkpoint loads strictly, with every name and shape matching."""
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    state = torch.load(UNTRAINED_CHECKPOINT, map_location="cpu")["model_state_dict"]

    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    saved = {k: tuple(v.shape) for k, v in state.items()}
    assert saved == expected

    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected


# --------------------------------------------------------------------------------------
# TemporalMerge runs one sequence per track.
# --------------------------------------------------------------------------------------


def test_temporal_merge_runs_each_track_as_its_own_sequence():
    """A track's output equals the LSTM run on that track's frames alone, in time order."""
    from torch_tracking.layers import TemporalMerge

    merge = TemporalMerge(encoder_dim=8).eval()
    x = torch.randn(2, 5, 3, 8)

    with torch.no_grad():
        merged = merge(x)
        for b in range(2):
            for track in range(3):
                alone, _ = merge.lstm(x[b : b + 1, :, track])  # (1, T, F)
                assert torch.allclose(merged[b, :, track], alone[0], atol=1e-6), (b, track)


def test_temporal_merge_is_independent_of_other_slots():
    """Four added padding slots leave every real track's output unchanged."""
    from torch_tracking.layers import TemporalMerge

    merge = TemporalMerge(encoder_dim=8).eval()
    x = torch.randn(1, 4, 3, 8)
    padded = torch.cat([x, torch.zeros(1, 4, 4, 8)], dim=2)

    with torch.no_grad():
        assert torch.allclose(merge(x), merge(padded)[:, :, :3], atol=1e-6)


def test_temporal_merge_with_one_track_matches_the_old_reshape():
    """Confinement check: with one track the old and corrected layouts coincide.

    So the per-track layout can only change a batch that holds several tracks, which is
    what separates its effect from any other movement in the model's output.
    """
    from torch_tracking.layers import TemporalMerge

    merge = TemporalMerge(encoder_dim=8).eval()
    x = torch.randn(2, 5, 1, 8)

    with torch.no_grad():
        old_style, _ = merge.lstm(x.reshape(2 * 1, 5, 8))
        assert torch.allclose(
            merge(x), old_style.reshape(2, 5, 1, 8), atol=1e-6
        )
