"""Unit tests for the LapPE structural-coordinate ETNN backbone.

These tests document the main adaptation used for GraphUniverse-style data:
rank-0 LapPE vectors are kept separate from node features, lifted to
higher-rank cells through incidence averaging, and consumed only through
rigid-motion-invariant squared distances inside ETNN messages.
"""

import pytest
import torch

from test.nn.backbones.combinatorial.test_etnn import (
    create_mock_complex_batch,
)
from topobench.nn.backbones.combinatorial.etnn_lappe import (
    ETNNLapPE,
    _build_lappe_cell_coordinates,
    _LapPEDistanceEncoder,
    _squared_coordinate_distances,
)


def create_lappe_complex_batch():
    """Create a mock lifted complex with canonical incidence matrices.

    The coordinate-free ETNN only needs named sparse neighborhoods. The LapPE
    variant also needs the base incidence matrices so it can construct
    higher-rank structural coordinates before message passing.
    """
    batch = create_mock_complex_batch()

    # The coordinate-enabled backbone needs the base incidence matrices because
    # it lifts rank-0 LapPE coordinates to higher-rank cells. The baseline ETNN
    # only consumes the directional neighborhood aliases.
    batch.incidence_1 = batch["down_incidence-1"]
    batch.incidence_2 = batch["down_incidence-2"]
    return batch


def create_lappe_etnn():
    """Instantiate the LapPE-distance ETNN variant used by the public config."""
    return ETNNLapPE(
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
    )


def create_lappe_rbf_etnn():
    """Instantiate the LapPE-RBF distance ETNN variant."""
    return ETNNLapPE(
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
        distance_encoding="rbf",
        include_raw_distance=True,
        num_rbf=8,
        rbf_min=0.0,
        rbf_max=2.0,
    )


def test_lappe_coordinate_etnn_runs_without_physical_positions():
    """LapPE mode uses structural coordinates and still does not need ``pos``."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    assert "pos" not in batch

    out = create_lappe_etnn()(batch)

    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_lappe_rbf_coordinate_etnn_runs_without_physical_positions():
    """LapPE-RBF mode should preserve the same TopoBench output contract."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    assert "pos" not in batch

    out = create_lappe_rbf_etnn()(batch)

    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_lappe_coordinate_etnn_requires_lappe_attribute():
    """Coordinate mode should fail clearly when preprocessing is missing."""
    batch = create_lappe_complex_batch()

    with pytest.raises(AttributeError, match="LapPE"):
        create_lappe_etnn()(batch)


def test_lappe_coordinate_etnn_validates_rank_0_coordinate_count():
    """LapPE rows must align one-to-one with rank-0 cells."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0] + 1, 3)

    with pytest.raises(ValueError, match="one rank-0 coordinate row"):
        create_lappe_etnn()(batch)


def test_lappe_cell_coordinates_are_barycentric_by_rank():
    """Higher-rank coordinates should be incidence-weighted barycenters."""
    batch = create_lappe_complex_batch()

    # The four rank-0 cells form a square. Rank-1 coordinates should become
    # edge midpoints, and rank-2 coordinates should become averages over the
    # incident rank-1 coordinates encoded in incidence_2.
    batch.LapPE = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    coordinates = _build_lappe_cell_coordinates(
        batch=batch,
        coordinate_attr="LapPE",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_rank_1 = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )
    expected_rank_2 = torch.tensor(
        [
            [4.0 / 3.0, 1.0],
            [1.0, 4.0 / 3.0],
        ]
    )
    assert torch.allclose(coordinates[0], batch.LapPE)
    assert torch.allclose(coordinates[1], expected_rank_1)
    assert torch.allclose(coordinates[2], expected_rank_2)


def test_lappe_cell_coordinates_handle_empty_rank():
    """Empty ranks should receive empty coordinate tensors, not fail."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    batch.x_2 = torch.empty(0, 16)
    batch.batch_2 = torch.empty(0, dtype=torch.long)
    batch.incidence_2 = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(batch.x_1.shape[0], 0),
    ).coalesce()

    coordinates = _build_lappe_cell_coordinates(
        batch=batch,
        coordinate_attr="LapPE",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coordinates[2].shape == (0, 3)


def test_lappe_cell_coordinates_validate_incidence_source_axis():
    """Incidence source rows must align with lower-rank coordinates."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    batch.incidence_1 = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(batch.x_0.shape[0] + 1, batch.x_1.shape[0]),
    ).coalesce()

    with pytest.raises(ValueError, match="source rows"):
        _build_lappe_cell_coordinates(
            batch=batch,
            coordinate_attr="LapPE",
            max_rank=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_squared_lappe_distance_is_rigid_motion_invariant():
    """Distance features should ignore the arbitrary structural frame."""
    src = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    dst = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [0, 1]])
    base = _squared_coordinate_distances(src, dst, edge_index, torch.float32)

    # LapPE eigenvectors may flip signs or rotate within repeated eigenspaces.
    # Squared distances are the invariant scalar passed to the message MLP, so
    # they should survive rotations, reflections, and translations unchanged.
    rotation_reflection = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
    translation = torch.tensor([4.0, -2.0])
    transformed_src = src @ rotation_reflection.T + translation
    transformed_dst = dst @ rotation_reflection.T + translation
    transformed = _squared_coordinate_distances(
        transformed_src,
        transformed_dst,
        edge_index,
        torch.float32,
    )

    assert torch.allclose(base, transformed)


def test_lappe_rbf_distance_encoding_is_rigid_motion_invariant():
    """RBF-expanded distances should keep the scalar-distance invariance."""
    src = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    dst = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [0, 1]])
    encoder = _LapPEDistanceEncoder(
        distance_encoding="rbf",
        include_raw_distance=True,
        num_rbf=8,
        rbf_min=0.0,
        rbf_max=2.0,
        rbf_gamma=None,
        distance_eps=1e-8,
    )

    base = encoder(src, dst, edge_index, torch.float32)

    # The encoded features are functions only of Euclidean distance. They
    # should be unchanged by rotations, reflections, and translations of the
    # structural coordinate frame.
    rotation_reflection = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
    translation = torch.tensor([4.0, -2.0])
    transformed_src = src @ rotation_reflection.T + translation
    transformed_dst = dst @ rotation_reflection.T + translation
    transformed = encoder(
        transformed_src,
        transformed_dst,
        edge_index,
        torch.float32,
    )

    assert base.shape == (2, 9)
    assert torch.allclose(base, transformed)


def test_lappe_distance_encoder_rejects_unknown_mode():
    """Invalid distance encodings should fail at model construction time."""
    with pytest.raises(ValueError, match="distance_encoding"):
        _LapPEDistanceEncoder(
            distance_encoding="raw_coordinates",
            include_raw_distance=True,
            num_rbf=8,
            rbf_min=0.0,
            rbf_max=2.0,
            rbf_gamma=None,
            distance_eps=1e-8,
        )
