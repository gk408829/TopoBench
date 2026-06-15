"""LapPE structural-coordinate ETNN backbone for combinatorial complexes.

This module implements a coordinate-enabled TopoBench adaptation of
E(n)-Equivariant Topological Neural Networks (ETNNs) from Battiloro et al.,
``E(n) Equivariant Topological Neural Networks``, arXiv:2405.15429, and the
official implementation at
``https://github.com/NSAPH-Projects/topological-equivariant-networks``.

The coordinate-free backbone in ``etnn.py`` implements the ETNN/CCMPN feature
update over TopoBench neighborhoods. This file keeps that feature update but
adds structural pseudo-coordinates for graph datasets, such as GraphUniverse,
that do not provide physical Euclidean coordinates.

The construction is deliberately conservative:

1. A preprocessing transform computes normalized graph Laplacian eigenvectors
   as rank-0 structural coordinates:

       p_0(v) = LapPE(v)

2. Higher-rank cell coordinates are obtained recursively by incidence
   averaging:

       p_r(c) = mean_{d incident to c} p_{r-1}(d)

   In TopoBench combinatorial complexes, ``incidence_r`` has rank ``r-1`` cells
   on rows and rank ``r`` cells on columns. Absolute incidence values are used
   because orientation signs are not part of the coordinate barycenter.

3. The ETNN relation message receives the original sparse-neighborhood scalar
   and a rigid-motion invariant structural distance encoding. By default this
   encoding is the scalar squared distance:

       z_{d,c,N} = concat(h_d, h_c, a_{d,c,N}, ||p_d - p_c||^2)

   The optional ``distance_encoding="rbf"`` mode replaces this single channel
   with a Euclidean-distance RBF expansion:

       d_{d,c} = sqrt(||p_d - p_c||^2 + eps)
       e_k     = exp(-gamma * (d_{d,c} - mu_k)^2)
       z       = concat(h_d, h_c, a_{d,c,N}, [d_{d,c}], e_1, ..., e_K)

   The feature update is therefore:

       m_{c,N} = sum_{d in N(c)} psi_N(z_{d,c,N})
       h'_c    = h_c + beta_rank(c)(h_c, concat_N m_{c,N})

This follows the CCMPN/ETNN neighborhood aggregation and feature-update
structure while specializing the geometric invariant input to distances in a
structural graph embedding. The coordinate update from ETNN is omitted:
coordinates are fixed auxiliary features, not learned dynamical states. Thus
the model is invariant to rigid transformations of the structural coordinate
frame, but the coordinates should be interpreted as graph-derived structural
embeddings rather than physical Euclidean positions.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from topobench.data.utils import get_routes_from_neighborhoods
from topobench.nn.backbones.combinatorial.etnn import (
    _ETNNMessagePassing,
    _make_mlp,
    _neighborhood_to_edge_index,
)


class ETNNLapPE(nn.Module):
    """ETNN feature backbone with LapPE structural distance messages.

    The class mirrors the coordinate-free ``ETNN`` backbone but requires a
    rank-0 coordinate attribute, usually ``LapPE``, produced before
    graph-to-combinatorial lifting. For each relation edge from sender cell
    ``d`` to receiver cell ``c``, the relation-specific message function sees
    the sparse TopoBench relation value ``a_{d,c,N}`` together with an
    E(n)-invariant structural-distance encoding derived from
    ``||p_d - p_c||``.

    By default, this encoding is the scalar squared distance. Optionally, the
    distance can be expanded with radial basis functions, which keeps the
    geometric input invariant while giving the message MLP a richer distance
    representation.

    This variant keeps ETNN's topological feature update structure:

        m_{c,N} = sum_{d in N(c)} psi_N(h_d, h_c, a_{d,c,N}, enc(||p_d-p_c||))
        h'_c    = h_c + beta_rank(c)(h_c, concat_N m_{c,N})

    Here ``enc`` is either the default squared distance or an RBF expansion of
    Euclidean distance. These equations follow the ETNN feature-update and
    neighborhood-aggregation structure. The LapPE term is used only as an
    invariant message feature. Coordinates are not updated, so this module
    should be understood as a structural-coordinate adaptation rather than a
    full coordinate-dynamical ETNN.

    The backbone expects the lifting/feature-encoding pipeline to provide
    feature tensors for every rank from 0 to ``max_rank``. Empty ranks should
    still be represented by zero-row tensors such as ``x_2.shape == [0, d]``;
    missing rank attributes are treated as malformed input.

    Parameters
    ----------
    in_channels : int
        Input feature dimension for every visible cell rank.
    hidden_channels : int
        Hidden dimension used by ETNN layers.
    out_channels : int
        Output feature dimension for every visible cell rank.
    neighborhoods : list[str]
        TopoBench neighborhood names, e.g. ``"up_adjacency-0"`` or
        ``"down_incidence-1"``.
    num_layers : int, optional
        Number of ETNN message-passing layers.
    dropout : float, optional
        Dropout probability used inside message and update blocks.
    activation : str, optional
        Activation function name.
    use_batch_norm : bool, optional
        Whether to use batch normalization inside MLP blocks.
    coordinate_attr : str, optional
        Batch attribute containing rank-0 structural coordinates.
    distance_encoding : {"scalar", "rbf"}, optional
        Structural distance encoding appended to each relation message. The
        default ``"scalar"`` preserves the original LapPE variant and appends
        ``||p_d - p_c||^2``. The ``"rbf"`` option appends Euclidean distance
        and a fixed radial-basis expansion of that distance.
    include_raw_distance : bool, optional
        Whether ``"rbf"`` mode includes the raw Euclidean distance before the
        RBF channels.
    num_rbf : int, optional
        Number of RBF centers used when ``distance_encoding="rbf"``.
    rbf_min : float, optional
        Minimum Euclidean distance center for RBF encoding.
    rbf_max : float, optional
        Maximum Euclidean distance center for RBF encoding.
    rbf_gamma : float | None, optional
        RBF width parameter. If ``None``, the value is derived from center
        spacing as ``1 / spacing^2``.
    distance_eps : float, optional
        Numerical epsilon used before the square root in RBF mode.
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
        distance_encoding: str = "scalar",
        include_raw_distance: bool = True,
        num_rbf: int = 8,
        rbf_min: float = 0.0,
        rbf_max: float = 2.0,
        rbf_gamma: float | None = None,
        distance_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(
                "ETNNLapPE requires at least one message-passing layer."
            )
        if len(neighborhoods) == 0:
            raise ValueError("ETNNLapPE requires at least one neighborhood.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.coordinate_attr = coordinate_attr
        self.distance_encoder = _LapPEDistanceEncoder(
            distance_encoding=distance_encoding,
            include_raw_distance=include_raw_distance,
            num_rbf=num_rbf,
            rbf_min=rbf_min,
            rbf_max=rbf_max,
            rbf_gamma=rbf_gamma,
            distance_eps=distance_eps,
        )

        # Keep the public neighborhood config identical to ETNN, then derive
        # source/destination ranks for relation-wise message passing.
        self.neighborhoods = list(neighborhoods)
        self.routes = get_routes_from_neighborhoods(self.neighborhoods)
        self.num_layers = num_layers
        self.max_rank = max(max(route) for route in self.routes)

        # AllCellFeatureEncoder projects every selected rank to the same hidden
        # size, so one projection shape works across ranks.
        self.input_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(in_channels, hidden_channels)
                for rank in range(self.max_rank + 1)
            }
        )

        # LapPE layers use the same topological relations as ETNN, then append
        # the configured structural-distance encoding per relation edge. Scalar
        # mode preserves the original LapPE variant; RBF mode gives the message
        # MLP a richer invariant distance basis without exposing raw
        # coordinate-frame-dependent vectors.
        self.layers = nn.ModuleList(
            [
                _ETNNLapPELayer(
                    neighborhoods=self.neighborhoods,
                    routes=self.routes,
                    hidden_channels=hidden_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    distance_encoder=self.distance_encoder,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.ModuleDict(
            {
                str(rank): nn.Linear(hidden_channels, out_channels)
                for rank in range(self.max_rank + 1)
            }
        )

    def forward(self, batch) -> dict[int, torch.Tensor]:
        """Run LapPE-distance ETNN message passing.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing rank-wise features ``x_i``,
            sparse neighborhood tensors, incidence matrices, and rank-0
            structural coordinates.

        Returns
        -------
        dict[int, torch.Tensor]
            Rank-indexed output embeddings compatible with ``TuneWrapper``.
        """
        x = {}
        for rank in range(self.max_rank + 1):
            key = f"x_{rank}"

            # The model config determines which ranks are visible through its
            # neighborhoods. Every visible rank must already have been encoded
            # by AllCellFeatureEncoder as ``x_0``, ``x_1``, ...
            if not hasattr(batch, key):
                raise AttributeError(
                    f"ETNNLapPE expected rank-{rank} features at `{key}`."
                )

            # Project every rank into the common hidden space used by ETNN
            # message and update functions.
            x[rank] = self.input_projection[str(rank)](getattr(batch, key))

        # Build structural coordinates once so every layer uses the same
        # coordinate frame and the same rank-wise cell summaries. This is done
        # inside the forward pass for the first variant because TopoBench
        # batches are assembled dynamically and moved across devices by
        # Lightning. If profiling shows this is a bottleneck, the same
        # coordinate lifting can be moved into a preprocessing transform.
        coordinates = _build_lappe_cell_coordinates(
            batch=batch,
            coordinate_attr=self.coordinate_attr,
            max_rank=self.max_rank,
            device=x[0].device,
            dtype=x[0].dtype,
        )

        # Apply relation-wise message passing. Coordinates are fixed auxiliary
        # features in this variant; only cell features are updated by layers.
        for layer in self.layers:
            x = layer(x, batch, coordinates=coordinates)

        # Return rank-indexed embeddings for the standard TopoBench wrapper.
        return {
            rank: self.output_projection[str(rank)](features)
            for rank, features in x.items()
        }


class _ETNNLapPELayer(nn.Module):
    """One ETNN layer with LapPE structural distance edge attributes.

    The layer has the same two aggregation levels as the coordinate-free ETNN:

    1. For each configured neighborhood relation ``N``, sender-cell messages
       are summed into receiver cells.
    2. For each destination rank, messages from all incoming relation types are
       concatenated and passed through a rank-specific update MLP.

    The only architectural difference is the relation edge attribute. Instead
    of using only the sparse neighborhood value ``a_{d,c,N}``, each message
    also sees an invariant structural-distance encoding computed from
    LapPE-derived cell coordinates. This keeps the geometric signal invariant
    to translations, rotations, and reflections of the structural coordinate
    frame.

    Relation-message order is part of the model definition: messages are
    concatenated in the order given by ``self.neighborhoods``. The Hydra config
    therefore fixes both which relations are used and the channel order seen by
    the rank-wise update MLPs.

    Parameters
    ----------
    neighborhoods : list[str]
        TopoBench sparse neighborhood names used as ETNN relation types.
    routes : list[list[int]]
        Source and destination rank pairs inferred from ``neighborhoods``.
    hidden_channels : int
        Hidden feature dimension for every rank.
    dropout : float
        Dropout probability used in message and update MLPs.
    activation : str
        Activation name used in message and update MLPs.
    use_batch_norm : bool
        Whether to insert batch normalization in MLP blocks.
    distance_encoder : _LapPEDistanceEncoder
        Module that converts sender/receiver structural coordinates into the
        invariant distance features appended to each relation edge.
    """

    def __init__(
        self,
        neighborhoods: list[str],
        routes: list[list[int]],
        hidden_channels: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
        distance_encoder: _LapPEDistanceEncoder,
    ) -> None:
        super().__init__()
        self.neighborhoods = list(neighborhoods)
        self.routes = [tuple(route) for route in routes]
        self.distance_encoder = distance_encoder
        if len(self.neighborhoods) != len(self.routes):
            raise ValueError(
                "ETNNLapPE expected one route per neighborhood, but found "
                f"{len(self.neighborhoods)} neighborhoods and "
                f"{len(self.routes)} routes."
            )

        # One sparse-neighborhood scalar plus the configured structural
        # distance encoding are supplied to each relation message.
        self.message_passing = nn.ModuleList(
            [
                _ETNNMessagePassing(
                    hidden_channels=hidden_channels,
                    edge_channels=1 + distance_encoder.out_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for _ in self.neighborhoods
            ]
        )

        # Count how many relation types send messages into each rank. The
        # update MLP input is the current state plus one aggregated message per
        # incoming relation.
        incoming_counts = defaultdict(int)
        for _, dst_rank in self.routes:
            incoming_counts[dst_rank] += 1

        # ETNN uses rank-wise update functions. A node, edge, and face can
        # receive different numbers and types of relation messages.
        ranks = sorted({rank for route in self.routes for rank in route})
        self.update = nn.ModuleDict(
            {
                str(rank): _make_mlp(
                    in_channels=(1 + incoming_counts[rank]) * hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for rank in ranks
            }
        )

    def forward(
        self,
        x: dict[int, torch.Tensor],
        batch,
        coordinates: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Apply one LapPE-distance ETNN layer.

        Parameters
        ----------
        x : dict[int, torch.Tensor]
            Rank-indexed hidden cell features.
        batch : torch_geometric.data.Data
            Lifted TopoBench batch containing sparse neighborhoods.
        coordinates : dict[int, torch.Tensor]
            Rank-indexed structural cell coordinates. The keys must match the
            ranks in ``x``.

        Returns
        -------
        dict[int, torch.Tensor]
            Updated rank-indexed hidden cell features.
        """
        # Accumulate one aggregated message tensor per destination rank and per
        # incoming relation type. The list order follows ``self.neighborhoods``
        # and is therefore deterministic for a fixed config.
        messages_by_rank = defaultdict(list)

        for route_idx, (neighborhood, route) in enumerate(
            zip(self.neighborhoods, self.routes, strict=True)
        ):
            src_rank, dst_rank = route

            # Convert the sparse TopoBench relation into explicit sender and
            # receiver indices. The returned edge attribute is the scalar sparse
            # neighborhood value ``a_{d,c,N}``.
            edge_index, edge_attr = _neighborhood_to_edge_index(
                batch=batch,
                neighborhood=neighborhood,
                src_rank=src_rank,
                dst_rank=dst_rank,
                device=x[src_rank].device,
                dtype=x[src_rank].dtype,
                num_src_cells=x[src_rank].shape[0],
                num_dst_cells=x[dst_rank].shape[0],
            )
            if edge_attr.ndim != 2 or edge_attr.shape[1] != 1:
                raise ValueError(
                    "ETNNLapPE expected `_neighborhood_to_edge_index` to "
                    "return scalar edge attributes with shape [num_edges, 1], "
                    f"but found shape {tuple(edge_attr.shape)}."
                )

            # Add the invariant structural-coordinate distance encoding. This
            # is the only place where LapPE coordinates enter the feature
            # update. RBF mode increases distance-channel capacity while
            # preserving invariance to coordinate-frame transformations.
            distance_attr = self.distance_encoder(
                src_coordinates=coordinates[src_rank],
                dst_coordinates=coordinates[dst_rank],
                edge_index=edge_index,
                dtype=edge_attr.dtype,
            )
            edge_attr = torch.cat([edge_attr, distance_attr], dim=-1)

            # Apply the relation-specific ETNN message function and store the
            # aggregated receiver messages under the destination rank.
            message = self.message_passing[route_idx](
                x_src=x[src_rank],
                x_dst=x[dst_rank],
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            messages_by_rank[dst_rank].append(message)

        out = {}
        for rank, features in x.items():
            # Defensive fallback: if a future config exposes a rank that is
            # never touched by any configured relation, keep that rank's
            # features unchanged rather than failing at lookup time. The
            # standard ETNN-LapPE config touches ranks 0, 1, and 2.
            if str(rank) not in self.update:
                out[rank] = features
                continue

            # Concatenate the current rank state with all messages arriving at
            # that rank, then apply the rank-specific residual update.
            update_input = torch.cat(
                [features, *messages_by_rank.get(rank, [])], dim=-1
            )
            out[rank] = features + self.update[str(rank)](update_input)
        return out


def _build_lappe_cell_coordinates(
    batch,
    coordinate_attr: str,
    max_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Construct rank-wise structural cell coordinates from rank-0 LapPE.

    Rank 0 uses the coordinate matrix stored in ``coordinate_attr``. Each
    higher rank is obtained by averaging the coordinates of incident
    lower-rank cells through the corresponding absolute incidence matrix.

    The expected incidence convention is the TopoBench convention used by
    combinatorial complexes: ``incidence_r`` has rows for rank ``r-1`` cells
    and columns for rank ``r`` cells. Absolute values are used because
    orientation signs are meaningful for boundary operators but not for
    barycentric coordinate averaging.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Lifted TopoBench batch containing rank-wise features, incidence
        matrices, and the rank-0 coordinate attribute.
    coordinate_attr : str
        Attribute containing rank-0 structural coordinates, usually ``LapPE``.
    max_rank : int
        Maximum cell rank that needs coordinates.
    device : torch.device
        Device for returned coordinate tensors.
    dtype : torch.dtype
        Floating dtype for returned coordinate tensors.

    Returns
    -------
    dict[int, torch.Tensor]
        Rank-indexed structural coordinate tensors.
    """
    if not hasattr(batch, coordinate_attr):
        raise AttributeError(
            f"ETNNLapPE expected rank-0 coordinates at `{coordinate_attr}`."
        )

    # Rank-0 coordinates come directly from the preprocessing transform. We
    # move them to the active model device because Lightning may transfer the
    # batch before the backbone sees it.
    rank_0_coordinates = getattr(batch, coordinate_attr).to(
        device=device, dtype=dtype
    )

    # Fail loudly if preprocessing produced a malformed coordinate tensor. A
    # silent row mismatch would attach coordinates to the wrong cells.
    if rank_0_coordinates.ndim != 2:
        raise ValueError(
            "ETNNLapPE expected a rank-0 coordinate matrix at "
            f"`{coordinate_attr}`, but found shape "
            f"{tuple(rank_0_coordinates.shape)}."
        )
    if rank_0_coordinates.shape[0] != batch.x_0.shape[0]:
        raise ValueError(
            "ETNNLapPE expected one rank-0 coordinate row per rank-0 cell, "
            f"but found {rank_0_coordinates.shape[0]} coordinates for "
            f"{batch.x_0.shape[0]} cells."
        )

    coordinates = {0: rank_0_coordinates}
    for rank in range(1, max_rank + 1):
        feature_key = f"x_{rank}"
        if not hasattr(batch, feature_key):
            raise AttributeError(
                f"ETNNLapPE expected rank-{rank} features at `{feature_key}`."
            )

        incidence_key = f"incidence_{rank}"
        if not hasattr(batch, incidence_key):
            raise AttributeError(
                "ETNNLapPE needs incidence matrices to lift coordinates, but "
                f"`{incidence_key}` is missing."
            )

        # Lift coordinates one rank at a time: rank 0 -> rank 1 -> rank 2.
        # This recursive incidence averaging avoids adding a new preprocessing
        # contract for direct vertex-to-cell incidence while still producing a
        # coordinate for every visible cell.
        incidence = getattr(batch, incidence_key).coalesce().to(device)
        coordinates[rank] = _average_coordinates_through_incidence(
            lower_coordinates=coordinates[rank - 1],
            incidence=incidence,
            num_cells=getattr(batch, feature_key).shape[0],
        )
    return coordinates


def _average_coordinates_through_incidence(
    lower_coordinates: torch.Tensor,
    incidence: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    """Average lower-rank coordinates over each incident higher-rank cell.

    Parameters
    ----------
    lower_coordinates : torch.Tensor
        Coordinates for rank ``r-1`` cells with shape
        ``[num_lower_cells, coordinate_dim]``.
    incidence : torch.Tensor
        Sparse incidence matrix with rows as rank ``r-1`` cells and columns as
        rank ``r`` cells.
    num_cells : int
        Number of real rank ``r`` cells in the current batch.

    Returns
    -------
    torch.Tensor
        Coordinate matrix for rank ``r`` cells with shape
        ``[num_cells, coordinate_dim]``.
    """
    if num_cells == 0:
        return lower_coordinates.new_empty((0, lower_coordinates.shape[1]))

    incidence = incidence.coalesce()

    # Check both sparse axes before indexing. These errors usually indicate an
    # incompatible lifting or a malformed batch, so failing here is better than
    # silently producing incorrect distances.
    if incidence.shape[0] != lower_coordinates.shape[0]:
        raise ValueError(
            "Cannot lift ETNNLapPE structural coordinates: incidence has "
            f"{incidence.shape[0]} source rows, but the lower rank has "
            f"{lower_coordinates.shape[0]} coordinate rows."
        )
    if incidence.shape[1] != num_cells:
        raise ValueError(
            "Cannot lift ETNNLapPE structural coordinates: incidence has "
            f"{incidence.shape[1]} columns, but the target rank has "
            f"{num_cells} cells."
        )

    # Drop explicit zero entries. They may appear as empty-rank placeholders and
    # should not contribute to coordinate averages.
    indices = incidence.indices()
    values = incidence.values().abs().to(lower_coordinates.dtype)
    nonzero_mask = values != 0
    indices = indices[:, nonzero_mask]
    values = values[nonzero_mask]

    # Accumulate weighted coordinate sums and total incidence weights for each
    # target cell. Absolute incidence values ignore orientation signs.
    coordinates = lower_coordinates.new_zeros(
        (num_cells, lower_coordinates.shape[1])
    )
    weights = lower_coordinates.new_zeros((num_cells, 1))
    if values.numel() == 0:
        return coordinates

    lower_idx = indices[0]
    cell_idx = indices[1]
    coordinates.index_add_(
        0, cell_idx, lower_coordinates[lower_idx] * values.unsqueeze(-1)
    )
    weights.index_add_(0, cell_idx, values.unsqueeze(-1))

    # Isolated or degenerate target cells keep zero coordinates. The clamp
    # avoids division by zero without changing nonzero averages.
    weights = weights.clamp_min(torch.finfo(weights.dtype).eps)
    return coordinates / weights


class _LapPEDistanceEncoder(nn.Module):
    """Encode invariant structural distances for ETNN relation messages.

    Parameters
    ----------
    distance_encoding : {"scalar", "rbf"}
        Distance feature type. ``"scalar"`` returns squared Euclidean distance.
        ``"rbf"`` returns Euclidean distance and fixed RBF features.
    include_raw_distance : bool
        Whether RBF mode includes the Euclidean distance channel.
    num_rbf : int
        Number of fixed RBF centers in RBF mode.
    rbf_min : float
        Minimum Euclidean distance center.
    rbf_max : float
        Maximum Euclidean distance center.
    rbf_gamma : float | None
        RBF width parameter. If ``None``, it is derived from center spacing.
    distance_eps : float
        Numerical epsilon used before square root in RBF mode.
    """

    def __init__(
        self,
        distance_encoding: str,
        include_raw_distance: bool,
        num_rbf: int,
        rbf_min: float,
        rbf_max: float,
        rbf_gamma: float | None,
        distance_eps: float,
    ) -> None:
        super().__init__()
        if distance_encoding not in {"scalar", "rbf"}:
            raise ValueError(
                "ETNNLapPE distance_encoding must be 'scalar' or 'rbf', "
                f"but found {distance_encoding!r}."
            )
        if num_rbf < 1:
            raise ValueError("ETNNLapPE requires num_rbf >= 1.")
        if rbf_max <= rbf_min:
            raise ValueError("ETNNLapPE requires rbf_max > rbf_min.")
        if distance_eps < 0:
            raise ValueError("ETNNLapPE requires non-negative distance_eps.")

        self.distance_encoding = distance_encoding
        self.include_raw_distance = include_raw_distance
        self.num_rbf = num_rbf
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.distance_eps = distance_eps

        if rbf_gamma is None:
            if num_rbf == 1:
                spacing = rbf_max - rbf_min
            else:
                spacing = (rbf_max - rbf_min) / (num_rbf - 1)
            rbf_gamma = 1.0 / (spacing**2)
        if rbf_gamma <= 0:
            raise ValueError("ETNNLapPE requires positive rbf_gamma.")
        self.rbf_gamma = float(rbf_gamma)

        centers = torch.linspace(rbf_min, rbf_max, num_rbf)
        self.register_buffer("rbf_centers", centers)

    @property
    def out_channels(self) -> int:
        """Number of distance channels appended to each relation edge.

        Returns
        -------
        int
            Number of distance-encoding channels.
        """
        if self.distance_encoding == "scalar":
            return 1
        return self.num_rbf + int(self.include_raw_distance)

    def forward(
        self,
        src_coordinates: torch.Tensor,
        dst_coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode distances for relation edges.

        Parameters
        ----------
        src_coordinates : torch.Tensor
            Coordinates for sender-rank cells.
        dst_coordinates : torch.Tensor
            Coordinates for receiver-rank cells.
        edge_index : torch.Tensor
            Relation edges in ``[sender, receiver]`` format.
        dtype : torch.dtype
            Floating dtype for the returned distance features.

        Returns
        -------
        torch.Tensor
            Distance feature matrix with shape
            ``[num_edges, self.out_channels]``.
        """
        squared_distance = _squared_coordinate_distances(
            src_coordinates=src_coordinates,
            dst_coordinates=dst_coordinates,
            edge_index=edge_index,
            dtype=dtype,
        )
        if self.distance_encoding == "scalar":
            return squared_distance

        distance = torch.sqrt(squared_distance.clamp_min(self.distance_eps))
        centers = self.rbf_centers.to(device=distance.device, dtype=dtype)
        rbf = torch.exp(
            -self.rbf_gamma * (distance - centers.unsqueeze(0)).pow(2)
        )
        if self.include_raw_distance:
            return torch.cat([distance, rbf], dim=-1)
        return rbf


def _squared_coordinate_distances(
    src_coordinates: torch.Tensor,
    dst_coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute invariant squared distances for relation edges.

    Parameters
    ----------
    src_coordinates : torch.Tensor
        Coordinates for sender-rank cells.
    dst_coordinates : torch.Tensor
        Coordinates for receiver-rank cells.
    edge_index : torch.Tensor
        Relation edges in ``[sender, receiver]`` format.
    dtype : torch.dtype
        Floating dtype for the returned distance feature.

    Returns
    -------
    torch.Tensor
        Squared distance feature with shape ``[num_edges, 1]``.
    """
    if edge_index.numel() == 0:
        return src_coordinates.new_empty((0, 1), dtype=dtype)

    sender, receiver = edge_index

    # Squared Euclidean distance is invariant to translations, rotations, and
    # reflections of the structural coordinate frame.
    delta = src_coordinates[sender] - dst_coordinates[receiver]
    return delta.pow(2).sum(dim=-1, keepdim=True).to(dtype)
