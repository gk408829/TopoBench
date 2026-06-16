"""Learned-LapPE refinement ETNN backbone for combinatorial complexes.

This module implements a small learned-coordinate extension of the fixed
LapPE structural-coordinate ETNN in ``etnn_lappe.py``. The goal is deliberately
narrow: keep the tested ETNN-LapPE feature update and scalar distance
bottleneck, but allow the rank-0 structural coordinates to move slightly in a
task-adaptive direction.

The rank-0 coordinate construction is:

    base_p  = LapPE
    delta_p = small graph coordinate encoder(x_0, rank-0 adjacency)
    alpha   = max_alpha * sigmoid(raw_alpha)
    p_0     = base_p + alpha * delta_p
    p_0     = center_and_rms_normalize(p_0)

Higher-rank coordinates then use the same recursive incidence averaging as
ETNN-LapPE:

    p_r(c) = mean_{d incident to c} p_{r-1}(d)

Messages still receive only scalar squared structural distances:

    z_{d,c,N} = concat(h_d, h_c, a_{d,c,N}, ||p_d - p_c||^2)

This keeps the first learned-coordinate experiment anchored to LapPE rather
than learning arbitrary geometry from scratch. It also keeps the message input
invariant to translations, rotations, and reflections of the learned structural
coordinate frame. No auxiliary coordinate loss is introduced here; regularizers
should be added only if diagnostics show collapse, explosion, or instability.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from topobench.nn.backbones.combinatorial.etnn import (
    _neighborhood_to_edge_index,
)
from topobench.nn.backbones.combinatorial.etnn_lappe import (
    ETNNLapPE,
    _average_coordinates_through_incidence,
)


class ETNNLearnedLapPE(ETNNLapPE):
    """ETNN-LapPE with a bounded learned refinement of rank-0 coordinates.

    The class reuses ``ETNNLapPE`` message passing, distance encoding, and
    output contract. The only change is the coordinate source: rank-0 LapPE
    coordinates are refined by a small graph encoder before higher-rank
    incidence averaging.

    Parameters
    ----------
    in_channels : int
        Input feature dimension for every visible cell rank.
    hidden_channels : int
        Hidden dimension used by ETNN layers.
    out_channels : int
        Output feature dimension for every visible cell rank.
    neighborhoods : list[str]
        TopoBench neighborhood names used as typed ETNN relations.
    num_layers : int, optional
        Number of ETNN message-passing layers.
    dropout : float, optional
        Dropout probability used inside ETNN and coordinate-encoder blocks.
    activation : str, optional
        Activation function name.
    use_batch_norm : bool, optional
        Whether to use batch normalization inside ETNN MLP blocks.
    coordinate_attr : str, optional
        Batch attribute containing rank-0 LapPE coordinates.
    coordinate_dim : int, optional
        Dimension of the structural coordinate frame.
    coord_encoder_hidden_channels : int, optional
        Hidden dimension of the coordinate refinement encoder.
    coord_encoder_layers : int, optional
        Number of mean-aggregation graph layers used by the coordinate encoder.
    coordinate_encoder_neighborhood : str, optional
        Rank-0 sparse neighborhood used by the coordinate encoder.
    alpha_init : float, optional
        Initial effective refinement strength.
    max_alpha : float, optional
        Upper bound on the effective refinement strength.
    learn_alpha : bool, optional
        Whether ``raw_alpha`` is trainable.
    center_coordinates : bool, optional
        Whether to center refined rank-0 coordinates before lifting.
    normalize_coordinates : bool, optional
        Whether to RMS-normalize refined rank-0 coordinates before lifting.
    coordinate_eps : float, optional
        Numerical epsilon used during coordinate normalization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        neighborhoods: list[str],
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "silu",
        use_batch_norm: bool = False,
        coordinate_attr: str = "LapPE",
        coordinate_dim: int = 3,
        coord_encoder_hidden_channels: int = 32,
        coord_encoder_layers: int = 2,
        coordinate_encoder_neighborhood: str = "up_adjacency-0",
        alpha_init: float = 0.05,
        max_alpha: float = 0.25,
        learn_alpha: bool = True,
        center_coordinates: bool = True,
        normalize_coordinates: bool = True,
        coordinate_eps: float = 1e-8,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            neighborhoods=neighborhoods,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
            coordinate_attr=coordinate_attr,
            distance_encoding="scalar",
        )
        if coordinate_dim < 1:
            raise ValueError("ETNNLearnedLapPE requires coordinate_dim >= 1.")
        if coord_encoder_layers < 1:
            raise ValueError(
                "ETNNLearnedLapPE requires coord_encoder_layers >= 1."
            )
        if max_alpha <= 0:
            raise ValueError("ETNNLearnedLapPE requires max_alpha > 0.")
        if not 0 < alpha_init < max_alpha:
            raise ValueError(
                "ETNNLearnedLapPE requires 0 < alpha_init < max_alpha."
            )
        if coordinate_eps <= 0:
            raise ValueError("ETNNLearnedLapPE requires coordinate_eps > 0.")

        self.coordinate_dim = coordinate_dim
        self.coordinate_encoder_neighborhood = coordinate_encoder_neighborhood
        self.max_alpha = float(max_alpha)
        self.center_coordinates = center_coordinates
        self.normalize_coordinates = normalize_coordinates
        self.coordinate_eps = coordinate_eps

        self.coordinate_encoder = _MeanGraphCoordinateEncoder(
            in_channels=in_channels,
            hidden_channels=coord_encoder_hidden_channels,
            out_channels=coordinate_dim,
            num_layers=coord_encoder_layers,
            dropout=dropout,
            activation=activation,
        )

        # alpha = max_alpha * sigmoid(raw_alpha). Initialize raw_alpha by the
        # inverse sigmoid so the first correction has the requested magnitude.
        alpha_ratio = alpha_init / max_alpha
        raw_alpha_value = math.log(alpha_ratio / (1.0 - alpha_ratio))
        raw_alpha = torch.tensor(float(raw_alpha_value))
        if learn_alpha:
            self.raw_alpha = nn.Parameter(raw_alpha)
        else:
            self.register_buffer("raw_alpha", raw_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        """Current bounded refinement strength.

        Returns
        -------
        torch.Tensor
            Scalar tensor equal to ``max_alpha * sigmoid(raw_alpha)``.
        """
        return self.max_alpha * torch.sigmoid(self.raw_alpha)

    def forward(self, batch) -> dict[int, torch.Tensor]:
        """Run learned-LapPE-distance ETNN message passing.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing rank-wise features, sparse
            neighborhoods, incidence matrices, and rank-0 LapPE coordinates.

        Returns
        -------
        dict[int, torch.Tensor]
            Rank-indexed output embeddings compatible with ``TuneWrapper``.
        """
        x = {}
        for rank in range(self.max_rank + 1):
            key = f"x_{rank}"
            if not hasattr(batch, key):
                raise AttributeError(
                    "ETNNLearnedLapPE expected rank-"
                    f"{rank} features at `{key}`."
                )
            x[rank] = self.input_projection[str(rank)](getattr(batch, key))

        coordinates = self._build_learned_cell_coordinates(
            batch=batch,
            device=x[0].device,
            dtype=x[0].dtype,
        )

        for layer in self.layers:
            x = layer(x, batch, coordinates=coordinates)

        return {
            rank: self.output_projection[str(rank)](features)
            for rank, features in x.items()
        }

    def _build_learned_cell_coordinates(
        self,
        batch,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, torch.Tensor]:
        """Construct refined rank-wise coordinates for the current batch.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing rank-wise features, LapPE
            coordinates, sparse neighborhoods, and incidence matrices.
        device : torch.device
            Device for returned coordinate tensors.
        dtype : torch.dtype
            Floating dtype for returned coordinate tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            Rank-indexed refined structural coordinates.
        """
        if not hasattr(batch, self.coordinate_attr):
            raise AttributeError(
                "ETNNLearnedLapPE expected rank-0 coordinates at "
                f"`{self.coordinate_attr}`."
            )

        base_coordinates = getattr(batch, self.coordinate_attr).to(
            device=device, dtype=dtype
        )
        if base_coordinates.ndim != 2:
            raise ValueError(
                "ETNNLearnedLapPE expected a rank-0 coordinate matrix at "
                f"`{self.coordinate_attr}`, but found shape "
                f"{tuple(base_coordinates.shape)}."
            )
        if base_coordinates.shape[1] != self.coordinate_dim:
            raise ValueError(
                "ETNNLearnedLapPE expected coordinate_dim="
                f"{self.coordinate_dim}, but `{self.coordinate_attr}` has "
                f"{base_coordinates.shape[1]} channels."
            )
        if base_coordinates.shape[0] != batch.x_0.shape[0]:
            raise ValueError(
                "ETNNLearnedLapPE expected one rank-0 coordinate row per "
                f"rank-0 cell, but found {base_coordinates.shape[0]} "
                f"coordinates for {batch.x_0.shape[0]} cells."
            )

        # The coordinate encoder sees the rank-0 input features and uses a
        # TopoBench rank-0 sparse relation to exchange local graph information
        # before predicting the coordinate correction. This keeps the learned
        # coordinate path small and separate from the ETNN feature-update
        # hidden state.
        edge_index, _ = _neighborhood_to_edge_index(
            batch=batch,
            neighborhood=self.coordinate_encoder_neighborhood,
            src_rank=0,
            dst_rank=0,
            device=device,
            dtype=dtype,
            num_src_cells=batch.x_0.shape[0],
            num_dst_cells=batch.x_0.shape[0],
        )
        delta_coordinates = self.coordinate_encoder(
            features=batch.x_0.to(device=device, dtype=dtype),
            edge_index=edge_index,
        )
        if delta_coordinates.shape != base_coordinates.shape:
            raise ValueError(
                "ETNNLearnedLapPE coordinate encoder returned shape "
                f"{tuple(delta_coordinates.shape)}, expected "
                f"{tuple(base_coordinates.shape)}."
            )

        rank_0_coordinates = (
            base_coordinates
            + self.alpha.to(device=device, dtype=dtype) * delta_coordinates
        )
        rank_0_coordinates = _center_and_normalize_coordinates(
            coordinates=rank_0_coordinates,
            batch_index=getattr(batch, "batch_0", None),
            center=self.center_coordinates,
            normalize=self.normalize_coordinates,
            eps=self.coordinate_eps,
        )

        coordinates = {0: rank_0_coordinates}
        for rank in range(1, self.max_rank + 1):
            feature_key = f"x_{rank}"
            if not hasattr(batch, feature_key):
                raise AttributeError(
                    "ETNNLearnedLapPE expected rank-"
                    f"{rank} features at `{feature_key}`."
                )

            incidence_key = f"incidence_{rank}"
            if not hasattr(batch, incidence_key):
                raise AttributeError(
                    "ETNNLearnedLapPE needs incidence matrices to lift "
                    f"coordinates, but `{incidence_key}` is missing."
                )
            coordinates[rank] = _average_coordinates_through_incidence(
                lower_coordinates=coordinates[rank - 1],
                incidence=getattr(batch, incidence_key).coalesce().to(device),
                num_cells=getattr(batch, feature_key).shape[0],
            )
        return coordinates


class _MeanGraphCoordinateEncoder(nn.Module):
    """Small mean-aggregation graph encoder for coordinate refinements.

    Parameters
    ----------
    in_channels : int
        Input feature dimension for rank-0 cells.
    hidden_channels : int
        Hidden dimension used in intermediate graph layers.
    out_channels : int
        Output coordinate-refinement dimension.
    num_layers : int
        Number of mean-aggregation graph layers.
    dropout : float
        Dropout probability applied between hidden graph layers.
    activation : str
        Activation function name applied between hidden graph layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError(
                "ETNNLearnedLapPE requires coord_encoder_hidden_channels >= 1."
            )

        channels = [in_channels]
        channels.extend([hidden_channels] * max(num_layers - 1, 0))
        channels.append(out_channels)

        self.layers = nn.ModuleList(
            [
                _MeanGraphCoordinateLayer(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                )
                for idx in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation(activation)

    def forward(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Predict coordinate corrections from rank-0 features and adjacency.

        Parameters
        ----------
        features : torch.Tensor
            Rank-0 cell features with shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Rank-0 relation edges in ``[sender, receiver]`` format.

        Returns
        -------
        torch.Tensor
            Coordinate corrections with shape ``[num_nodes, out_channels]``.
        """
        h = features
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if layer_idx != len(self.layers) - 1:
                h = self.dropout(self.activation(h))
        return h


class _MeanGraphCoordinateLayer(nn.Module):
    """One mean-neighborhood layer used by the coordinate encoder.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output feature dimension.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels)
        self.neighbor_linear = nn.Linear(in_channels, out_channels)

    def forward(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Combine self features with mean sender features at each receiver.

        Parameters
        ----------
        features : torch.Tensor
            Cell features with shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Relation edges in ``[sender, receiver]`` format.

        Returns
        -------
        torch.Tensor
            Updated features with shape ``[num_nodes, out_channels]``.
        """
        if edge_index.numel() == 0:
            return self.self_linear(features)

        sender, receiver = edge_index
        aggregated = features.new_zeros(features.shape)
        aggregated.index_add_(0, receiver, features[sender])

        degree = features.new_zeros((features.shape[0], 1))
        degree.index_add_(
            0,
            receiver,
            torch.ones(
                (receiver.numel(), 1),
                device=features.device,
                dtype=features.dtype,
            ),
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        return self.self_linear(features) + self.neighbor_linear(aggregated)


def _center_and_normalize_coordinates(
    coordinates: torch.Tensor,
    batch_index: torch.Tensor | None,
    center: bool,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    """Center and RMS-normalize coordinates per graph when possible.

    Parameters
    ----------
    coordinates : torch.Tensor
        Coordinate matrix with shape ``[num_cells, coordinate_dim]``.
    batch_index : torch.Tensor | None
        Optional graph-id vector for rank-0 cells.
    center : bool
        Whether to subtract the coordinate mean.
    normalize : bool
        Whether to divide by RMS coordinate norm.
    eps : float
        Numerical floor for RMS normalization.

    Returns
    -------
    torch.Tensor
        Centered and/or normalized coordinates.
    """
    if not center and not normalize:
        return coordinates
    if coordinates.numel() == 0:
        return coordinates

    if batch_index is None:
        return _center_and_normalize_single_group(
            coordinates=coordinates,
            center=center,
            normalize=normalize,
            eps=eps,
        )

    batch_index = batch_index.to(device=coordinates.device)
    out = coordinates.clone()
    for graph_id in batch_index.unique(sorted=True):
        mask = batch_index == graph_id
        out[mask] = _center_and_normalize_single_group(
            coordinates=out[mask],
            center=center,
            normalize=normalize,
            eps=eps,
        )
    return out


def _center_and_normalize_single_group(
    coordinates: torch.Tensor,
    center: bool,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    """Apply centering and RMS normalization to one coordinate group.

    Parameters
    ----------
    coordinates : torch.Tensor
        Coordinate matrix for one graph or group.
    center : bool
        Whether to subtract the coordinate mean.
    normalize : bool
        Whether to divide by RMS coordinate norm.
    eps : float
        Numerical floor for RMS normalization.

    Returns
    -------
    torch.Tensor
        Centered and/or normalized coordinate matrix.
    """
    out = coordinates
    if center:
        out = out - out.mean(dim=0, keepdim=True)
    if normalize:
        rms = torch.sqrt(out.pow(2).sum(dim=-1).mean().clamp_min(eps))
        out = out / rms
    return out


def _get_activation(name: str) -> nn.Module:
    """Return the activation used by the coordinate encoder.

    Parameters
    ----------
    name : str
        Activation function name.

    Returns
    -------
    nn.Module
        Instantiated activation module.
    """
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported ETNNLearnedLapPE activation: {name!r}.")
