"""Tests for generic structural-coordinate ETNN variants."""

import torch

from test.nn.backbones.combinatorial.test_etnn_lappe import (
    create_lappe_complex_batch,
)
from topobench.nn.backbones.combinatorial.etnn_structural import (
    ETNNStructuralCoordinates,
)


def test_structural_coordinate_etnn_consumes_diffusion_attribute():
    """The generic backbone should work with non-LapPE coordinate names."""
    batch = create_lappe_complex_batch()
    batch.DiffusionPE = torch.randn(batch.x_0.shape[0], 3)

    model = ETNNStructuralCoordinates(
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
        coordinate_attr="DiffusionPE",
    )

    out = model(batch)

    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape
