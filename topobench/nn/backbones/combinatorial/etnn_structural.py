"""Generic structural-coordinate ETNN backbone.

This module exposes a small generic entry point for ETNN variants whose
rank-0 coordinates are produced by a preprocessing transform. The feature
update is intentionally the same as ``ETNNLapPE``:

    p_0 = structural rank-0 coordinates
    p_r = mean_{d incident to c} p_{r-1,d}
    z_{d,c,N} = concat(h_d, h_c, a_{d,c,N}, ||p_d - p_c||^2)

The class is useful when comparing different structural coordinate choices,
for example Laplacian eigenvectors versus diffusion coordinates, without
duplicating the ETNN message-passing code.
"""

from __future__ import annotations

from topobench.nn.backbones.combinatorial.etnn_lappe import ETNNLapPE


class ETNNStructuralCoordinates(ETNNLapPE):
    """ETNN backbone parameterized by a rank-0 coordinate attribute.

    Parameters are identical to ``ETNNLapPE``. The only semantic difference is
    naming: ``coordinate_attr`` may point to any rank-0 structural coordinate
    tensor with shape ``[num_rank_0_cells, coordinate_dim]``. The coordinates
    are lifted to higher ranks by recursive incidence averaging, then used only
    through squared-distance scalar features in ETNN relation messages.

    The model is equivariant to rigid transformations of the chosen structural
    coordinate system. These coordinates are not assumed to be physical
    Euclidean coordinates unless the dataset provides such geometry.

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
        Dropout probability used inside message and update blocks.
    activation : str, optional
        Activation function name.
    use_batch_norm : bool, optional
        Whether to use batch normalization inside MLP blocks.
    coordinate_attr : str, optional
        Batch attribute containing rank-0 structural coordinates.
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
        coordinate_attr: str = "StructuralPE",
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
        )
