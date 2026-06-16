"""Unit tests for learned-LapPE ETNN coordinate refinement.

The learned-coordinate variant should remain a conservative extension of
ETNN-LapPE: start from rank-0 LapPE, add a small bounded learned correction,
normalize the resulting structural coordinate frame, and keep scalar squared
distance as the only geometric relation feature.
"""

import pytest
import torch

from test.nn.backbones.combinatorial.test_etnn_lappe import (
    create_lappe_complex_batch,
)
from topobench.nn.backbones.combinatorial.etnn_learned_lappe import (
    ETNNLearnedLapPE,
    _center_and_normalize_coordinates,
)


def create_learned_lappe_etnn():
    """Instantiate the first learned-coordinate ETNN experiment config."""
    return ETNNLearnedLapPE(
        in_channels=16,
        hidden_channels=8,
        out_channels=16,
        neighborhoods=[
            "up_adjacency-0",
            "up_adjacency-1",
            "up_adjacency-2",
            "up_incidence-0",
            "down_incidence-1",
            "up_incidence-1",
            "down_incidence-2",
        ],
        num_layers=2,
        coordinate_attr="LapPE",
        coordinate_dim=3,
        coord_encoder_hidden_channels=8,
        coord_encoder_layers=2,
        alpha_init=0.05,
        max_alpha=0.25,
    )


def test_learned_lappe_etnn_runs_without_physical_positions():
    """Learned-LapPE should preserve the ETNN-LapPE TopoBench contract."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    assert "pos" not in batch

    out = create_learned_lappe_etnn()(batch)

    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_learned_lappe_alpha_initialization_matches_config():
    """raw_alpha should initialize alpha to alpha_init."""
    model = create_learned_lappe_etnn()

    assert torch.allclose(model.alpha, torch.tensor(0.05), atol=1e-6)
    assert model.alpha.item() < model.max_alpha


def test_learned_lappe_rejects_coordinate_dimension_mismatch():
    """The learned correction must live in the same coordinate frame as LapPE."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 4)

    with pytest.raises(ValueError, match="coordinate_dim"):
        create_learned_lappe_etnn()(batch)


def test_learned_lappe_builds_rankwise_coordinates():
    """Learned rank-0 coordinates should lift to every visible cell rank."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    model = create_learned_lappe_etnn()

    coordinates = model._build_learned_cell_coordinates(
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coordinates[0].shape == (batch.x_0.shape[0], 3)
    assert coordinates[1].shape == (batch.x_1.shape[0], 3)
    assert coordinates[2].shape == (batch.x_2.shape[0], 3)


def test_learned_lappe_validates_coordinate_encoder_output_shape():
    """Coordinate refinement must match the base LapPE coordinate shape."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    model = create_learned_lappe_etnn()

    class BadCoordinateEncoder(torch.nn.Module):
        """Return a malformed coordinate correction for shape-guard testing."""

        def forward(self, features, edge_index):
            """Ignore inputs and return the wrong coordinate dimension."""
            return features.new_zeros((features.shape[0], 2))

    model.coordinate_encoder = BadCoordinateEncoder()

    with pytest.raises(ValueError, match="coordinate encoder returned shape"):
        model._build_learned_cell_coordinates(
            batch=batch,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_coordinate_normalization_is_per_graph():
    """Batched graph coordinates should be centered and scaled independently."""
    coordinates = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [10.0, 10.0],
            [10.0, 14.0],
        ]
    )
    batch_index = torch.tensor([0, 0, 1, 1])

    normalized = _center_and_normalize_coordinates(
        coordinates=coordinates,
        batch_index=batch_index,
        center=True,
        normalize=True,
        eps=1e-8,
    )

    for graph_id in [0, 1]:
        group = normalized[batch_index == graph_id]
        assert torch.allclose(group.mean(dim=0), torch.zeros(2), atol=1e-6)
        rms = torch.sqrt(group.pow(2).sum(dim=-1).mean())
        assert torch.allclose(rms, torch.tensor(1.0), atol=1e-6)
