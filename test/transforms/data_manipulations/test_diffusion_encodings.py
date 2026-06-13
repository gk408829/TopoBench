"""Tests for heat-kernel diffusion coordinate encodings."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations import DiffusionPE


def create_cycle_graph(num_nodes: int = 4) -> Data:
    """Create a small undirected cycle graph for spectral-coordinate tests."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ]
    )
    x = torch.randn(num_nodes, 5)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def test_diffusion_pe_stores_coordinates_without_changing_features():
    """concat_to_x=False stores rank-0 coordinates as data.DiffusionPE."""
    data = create_cycle_graph()
    x_before = data.x.clone()

    out = DiffusionPE(
        max_pe_dim=3,
        diffusion_time=1.0,
        concat_to_x=False,
        method="exact",
    )(data)

    assert hasattr(out, "DiffusionPE")
    assert out.DiffusionPE.shape == (4, 3)
    assert torch.equal(out.x, x_before)
    assert not torch.isnan(out.DiffusionPE).any()


def test_diffusion_pe_can_concatenate_to_x():
    """concat_to_x=True follows the normal TopoBench encoding convention."""
    data = create_cycle_graph()

    out = DiffusionPE(
        max_pe_dim=2,
        diffusion_time=1.0,
        concat_to_x=True,
        method="exact",
    )(data)

    assert out.x.shape == (4, 7)


def test_larger_diffusion_time_smooths_coordinate_energy():
    """Heat-kernel scaling should damp higher-frequency coordinate energy."""
    data_fast = create_cycle_graph()
    data_slow = create_cycle_graph()
    data_slow.edge_index = data_fast.edge_index
    data_slow.x = data_fast.x.clone()

    fast = DiffusionPE(
        max_pe_dim=3,
        diffusion_time=0.0,
        concat_to_x=False,
        method="exact",
    )(data_fast).DiffusionPE
    slow = DiffusionPE(
        max_pe_dim=3,
        diffusion_time=2.0,
        concat_to_x=False,
        method="exact",
    )(data_slow).DiffusionPE

    assert torch.linalg.norm(slow) <= torch.linalg.norm(fast)


def test_diffusion_pe_rejects_negative_time():
    """Diffusion time is a heat-kernel scale and must be non-negative."""
    with pytest.raises(ValueError, match="non-negative diffusion_time"):
        DiffusionPE(max_pe_dim=3, diffusion_time=-1.0)
