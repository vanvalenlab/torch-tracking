"""L1 - model structural contracts. Needs torch only; no checkpoint, no dataset.

This layer pins structure that retraining must never change: the class ordering, the
temporal adjacency matrix shape, and the architecture `CellTracker` hardcodes. Random
weights are sufficient, because none of these depend on what the model learned.

The `tracking-model` capability spec is what these assert; `specify-tracking-model` wrote
it and filled this file out. The output pins at the top of the file are that change's own
acceptance condition rather than a permanent contract - see their comment. See
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
# `specify-tracking-model` closes ten divergences that are documentation, dead code or
# argument handling. None of them may change a tensor the model computes, and "none of
# them" is only credible if it is checked. These digests were taken from the code as it
# stood before that change touched it; every task group in it re-runs them.
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

PIN_EVAL_TRAINING_FORWARD = "d9710afe5f5c35f7d7852037e978543432732bbadc15285c7e2e8cfc54e9f82f"
PIN_EVAL_INFERENCE_FORWARD = "1ad2feebc10d2362d23198ad7d243df533916dea6931db07d9f10b4adaf04506"
PIN_TRAIN_MODE_TRAINING_FORWARD = "58a007bc11dc3678fc267d652ca9ed9768e22ace20ae5e64d25d803cddb13e45"


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
        "`specify-tracking-model` must not change any tensor the model computes. A "
        "failure here means a task in it altered model behavior, or a torch upgrade "
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
    drifting apart; the `tracking-model` spec states it as a requirement.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    assert model is not None

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "model constructed with no parameters"


def test_training_forward_output_is_pinned():
    """`training_forward` in eval mode is byte-identical to the pre-change output.

    Divergence 60's correction rewrites how the graph's edge set is built, and groups 2-7
    touch construction, widths and dead state. None may move this tensor.
    """
    model = _pinned_model().eval()

    with torch.no_grad():
        output = model.training_forward(*_pinned_training_inputs())

    assert output.shape == (1, 2, 4, 4, 3)
    _assert_pinned(output, PIN_EVAL_TRAINING_FORWARD, "training_forward (eval mode)")


def test_inference_forward_output_is_pinned():
    """`inference_forward` is byte-identical to the pre-change output.

    This is the path that produces published tracking results, so it carries constraint 1:
    the same inputs must give the same answer.
    """
    model = _pinned_model().eval()

    with torch.no_grad():
        output = model.inference_forward(*_pinned_inference_inputs())

    assert output.shape == (1, 1, 3, 2, 3)
    _assert_pinned(output, PIN_EVAL_INFERENCE_FORWARD, "inference_forward")


def test_training_mode_output_is_pinned():
    """`training_forward` in train mode, with dropout disabled, is byte-identical.

    Eval mode normalizes with the running statistics, so it cannot see a change to how
    BatchNorm aggregates a batch. Train mode can. This is the pin that would catch
    divergence 34 being corrected by accident rather than deliberately: a masked BatchNorm
    changes this tensor and leaves the two eval-mode pins untouched.

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
    """The edge set the graph layers receive is byte-identical to the pre-change one.

    Divergence 60's correction replaces a discarded `D^-1/2 A D^-1/2` with an explicit
    add-self-loops step. The whole point is that the resulting edge set does not move, so
    it is pinned exactly rather than described.

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


def test_padding_slots_currently_carry_self_loops():
    """Padding slots are self-looped rather than isolated. Divergence 60, second half.

    This asserts today's behavior, not the intended behavior. The `tracking-model`
    requirement "Padding slots are isolated inside the graph" states that a padding slot
    SHALL NOT carry a self-loop; correcting that changes padding embeddings, and so the
    BatchNorm statistics divergence 34 is about, so it travels with 34 in
    `correct-model-normalization` rather than landing here.

    When that change lands, this test SHALL fail, and that failure is the intended
    correction rather than a regression. Replace it with the isolation assertion then.
    """
    model = _pinned_model().eval()
    edge_index = _capture_edge_index(model, _pinned_training_inputs())

    edges = set(map(tuple, edge_index.t().tolist()))
    padding_nodes = [2 + 4 * frame for frame in range(3)] + [
        3 + 4 * frame for frame in range(3)
    ]

    for node in padding_nodes:
        assert (node, node) in edges, (
            f"padding node {node} lost its self-loop. If this is "
            "`correct-model-normalization`, that is the intended fix: delete this test "
            "and assert isolation instead."
        )
        neighbours = {b for a, b in edges if a == node and b != node}
        assert not neighbours, (
            f"padding node {node} is connected to real cells {neighbours}; the isolation "
            "guarantee `training-data` states has been broken"
        )


# --------------------------------------------------------------------------------------
# Construction and validation contract. Divergences 58 and 63.
# --------------------------------------------------------------------------------------


def test_default_configuration_constructs():
    """The constructor's own default builds a working model. Divergence 58.

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
    """No configuration leaves a comparison unbound. Divergence 63.

    The validator used to bind its squareness comparison inside two `if` branches and
    then read it unconditionally, so any third value raised `UnboundLocalError` from the
    validator rather than a `ValueError` describing the problem.
    """
    from torch_tracking.model import GNNTrackingModel

    for value in ["channels_first", "NHWC", "", None, 3]:
        with pytest.raises(ValueError):
            GNNTrackingModel(**{**MODEL_ARCH, "data_format": value})


def test_shape_arity_is_checked_before_the_shape_is_indexed():
    """The arity message states the length actually required. Divergence 63.

    It reported "should be length 4" for a length-3 requirement, and the check ran after
    two statements had already indexed the shape.
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
    """The `GNNTrackingModel` docstring example executes. Divergence 64's sibling.

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
# Normalization selection. Divergence 59.
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
    """`norm_layer='layer'` is refused rather than silently half-built. Divergence 59.

    It used to construct a mostly-BatchNorm model that raised on the first delta forward,
    which is the worst of the three possible behaviors: not honored, not refused, and not
    discovered until a forward pass.
    """
    from torch_tracking.model import GNNTrackingModel

    with pytest.raises(ValueError, match="norm_layer"):
        GNNTrackingModel(**{**MODEL_ARCH, "norm_layer": "layer"})


def test_encoders_no_longer_accept_a_normalization_they_cannot_build():
    """The modules that ignored `norm_layer` no longer advertise it.

    The spec requirement is "honored or not offered". These four could not honor it, so
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

    Divergence 60: `normalize_adjacency_symmetric` computed `D^-1/2 A D^-1/2` and the only
    caller threw the weights away with `.nonzero()`. The replacement does the one thing
    that survived. This asserts the two agree on the sparsity pattern, which is all the
    graph layers ever saw.
    """
    from torch_tracking.utils import add_self_loops_dense

    adj = torch.zeros(2, 4, 4)
    adj[:, 0, 1] = adj[:, 1, 0] = 1.0

    looped = add_self_loops_dense(adj)

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
# Feature widths are derived, not assumed. Divergence 69.
# --------------------------------------------------------------------------------------


def test_neighborhood_encoder_accepts_encoders_of_different_widths():
    """Differing encoder widths build and run. Divergence 69.

    `combined_dim` was `3 * embedding_dim`, which is right only while the three encoders
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
    """The decoder sizes itself from both widths it receives. Divergence 69's sibling.

    `input_dim` was `4 * embedding_dim`, which assumed the delta encoders were as wide as
    the embeddings. The register records this defect only for `NeighborhoodEncoder`.
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
# Checkpoint, device and crop size. Divergences 3, 37 and 67.
# --------------------------------------------------------------------------------------


def test_checkpoint_parameter_names_a_file():
    """`CellTracker` takes a checkpoint file path, and the parameter says so. Divergence 3.

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
    """The old spelling keeps working, loudly. Divergence 3.

    `metrics/PROVENANCE.md` reproduction commands and existing scripts pass
    `checkpoint_dir`, so it is deprecated rather than removed.
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
    """The documented default and the implemented default are both CPU. Divergence 37.

    The docstring claimed `cuda`. CPU is the correct behavior under constraint 5, so the
    documentation was what was wrong.
    """
    from torch_tracking.tracker import CellTracker

    tracker = CellTracker(
        checkpoint_path=str(untrained_checkpoint), verbose=False
    )

    assert tracker.device == "cpu"
    assert "Default is `cpu`" in CellTracker.__doc__


def test_crop_size_is_passed_through_to_feature_extraction():
    """`appearance_dim` reaches the extractor rather than relying on its default.

    Divergence 67: `CellTracker` set `appearance_dim = 32`, sized its arrays from it, and
    then called `get_image_features` without it. The two defaults agreed, so the bug was
    invisible until someone changed one of them.
    """
    import inspect

    from torch_tracking import tracker as tracker_module

    source = inspect.getsource(tracker_module.CellTracker._extract_features) \
        if hasattr(tracker_module.CellTracker, "_extract_features") \
        else inspect.getsource(tracker_module)

    assert "appearance_dim=self.appearance_dim" in source, (
        "get_image_features is still relying on its own default for the crop size"
    )


# --------------------------------------------------------------------------------------
# Dead state. Divergence 66.
# --------------------------------------------------------------------------------------


def test_every_built_layer_is_reachable_from_a_forward_pass(untrained_checkpoint):
    """The model builds no layer that nothing uses. Divergence 66.

    `model.py` built `unmerge_embeddings` and `unmerge_centroids` that nothing read;
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
    """The committed checkpoint still loads strictly. Divergence 66.

    The removed `Unmerge` layers hold no parameters, so the state dict is unaffected, but
    that is worth asserting rather than assuming.
    """
    from torch_tracking.model import GNNTrackingModel

    model = GNNTrackingModel(**MODEL_ARCH)
    state = torch.load(untrained_checkpoint, map_location="cpu")["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected


def test_no_attribute_is_assigned_and_never_read():
    """`pad_value` is gone from both places it was assigned. Divergence 66.

    The register names only `utils.py`. `loss.py` had its own copy, and neither was ever
    read; the padding sentinel the loss actually uses is written in the targets.
    """
    import inspect

    from torch_tracking import loss as loss_module
    from torch_tracking import utils as utils_module

    for module in (loss_module, utils_module):
        assert "pad_value" not in inspect.getsource(module), (
            f"{module.__name__} still assigns pad_value"
        )


# --------------------------------------------------------------------------------------
# Module shape contracts. Divergence 64.
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
        comparisons, deltas = training(
            torch.randn(batch, track_length, max_cells, 32, 32, 1),
            torch.randn(batch, track_length, max_cells, 3),
            torch.randn(batch, track_length, max_cells, 2),
            adjacency,
        )
    assert comparisons.shape == (batch, track_length - 1, max_cells, max_cells, 2 * dim)
    assert deltas.shape == (batch, track_length - 1, max_cells, max_cells, 2 * dim)

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


def test_model_modules_have_no_unrunnable_demo_blocks():
    """No model module ships a `__main__` block that raises. Divergence 64.

    They were the only shape checks these modules had, and all four were stale. The
    checks above replace them; this asserts the blocks are gone rather than quietly
    rotting again.
    """
    import inspect

    from torch_tracking import branches, decoder, encoders, layers, model

    for module in (layers, encoders, branches, decoder, model):
        source = inspect.getsource(module)
        assert 'if __name__' not in source, (
            f"{module.__name__} still has a __main__ block; its shape checks belong in "
            "this file, where they run"
        )
