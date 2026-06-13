"""Diffusion structural coordinate transform."""

import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.laplacian_encodings import LapPE


class DiffusionPE(LapPE):
    r"""Heat-kernel diffusion coordinates from the normalized graph Laplacian.

    For eigenpairs ``(lambda_i, phi_i)`` of the normalized graph Laplacian, the
    rank-0 structural coordinate for node ``v`` is

        p_v(t) = [exp(-t lambda_1) phi_1(v), ..., exp(-t lambda_d) phi_d(v)]

    where ``t`` is the diffusion time and ``d`` is ``max_pe_dim``. Small
    eigenvalues describe slowly varying graph structure; the heat-kernel factor
    downweights higher-frequency modes as ``t`` grows.

    Parameters
    ----------
    max_pe_dim : int
        Number of diffusion coordinate dimensions to keep.
    diffusion_time : float, optional
        Heat-kernel time ``t``. Larger values smooth coordinates more strongly.
    include_first : bool, optional
        If False, removes eigenvectors corresponding to near-zero eigenvalues.
    concat_to_x : bool, optional
        If True, concatenates coordinates to ``data.x``. If False, stores them
        in ``data.DiffusionPE`` for coordinate-aware backbones.
    eps : float, optional
        Tolerance used to identify near-zero eigenvalues.
    tolerance : float, optional
        Numerical tolerance passed to the eigenvalue solver.
    method : str, optional
        Computation method inherited from ``LapPE``: ``"exact"`` or ``"gpu"``.
    debug : bool, optional
        If True, uses the inherited solver debug path.
    **kwargs : dict
        Additional unused arguments accepted for config compatibility.
    """

    def __init__(
        self,
        max_pe_dim: int,
        diffusion_time: float = 1.0,
        include_first: bool = False,
        concat_to_x: bool = True,
        eps: float = 1e-6,
        tolerance: float = 0.001,
        method: str = "gpu",
        debug: bool = False,
        **kwargs,
    ) -> None:
        if diffusion_time < 0:
            raise ValueError(
                "DiffusionPE requires non-negative diffusion_time."
            )

        self.diffusion_time = float(diffusion_time)
        super().__init__(
            max_pe_dim=max_pe_dim,
            include_eigenvalues=False,
            include_first=include_first,
            concat_to_x=concat_to_x,
            eps=eps,
            tolerance=tolerance,
            method=method,
            debug=debug,
            **kwargs,
        )

    def forward(self, data: Data) -> Data:
        """Compute and attach diffusion coordinates.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data with diffusion coordinates concatenated to ``x`` or
            stored as ``data.DiffusionPE``.
        """
        # Reuse LapPE's normalized-Laplacian eigensolver. The overridden
        # _pad_and_concat method below turns eigenpairs into diffusion
        # coordinates by multiplying each eigenvector by exp(-t lambda).
        if self.method == "exact":
            pe = self._compute_exact(data.edge_index, data.num_nodes)
        else:
            pe = self._compute_gpu(data.edge_index, data.num_nodes)

        if self.concat_to_x:
            if data.x is None:
                data.x = pe
            else:
                data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.DiffusionPE = pe

        return data

    def _pad_and_concat(
        self,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert eigenpairs to padded diffusion coordinates.

        ``LapPE`` calls this helper after solving for eigenpairs. DiffusionPE
        keeps the same padding convention but replaces raw eigenvectors with
        heat-kernel weighted eigenvectors.

        Parameters
        ----------
        evals : torch.Tensor
            Selected Laplacian eigenvalues with shape ``[num_modes]``.
        evecs : torch.Tensor
            Selected Laplacian eigenvectors with shape
            ``[num_nodes, num_modes]``.
        num_nodes : int
            Number of nodes in the graph. Included to match the inherited
            ``LapPE`` helper signature.
        device : torch.device
            Device on which the returned coordinate tensor should live.

        Returns
        -------
        torch.Tensor
            Diffusion coordinate matrix with shape ``[num_nodes, max_pe_dim]``.
        """
        # Apply the heat-kernel weight before padding. Padding then appends
        # exactly zero coordinate columns when the graph has too few usable
        # eigenvectors.
        weights = torch.exp(-self.diffusion_time * evals).to(device=device)
        pe = evecs * weights.unsqueeze(0)

        pad_width = self.max_pe_dim - pe.shape[1]
        if pad_width > 0:
            pe = torch.nn.functional.pad(
                pe, (0, pad_width), mode="constant", value=0
            )

        return pe
