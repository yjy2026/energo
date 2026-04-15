"""Parametric Return Model for electricity price distribution prediction.

Predicts (μ, σ) parameters of the price distribution using
Negative Log-Likelihood loss, adapted from the Parametric Return Modeling
approach.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ParametricPriceModel(nn.Module):
    """Predicts price distribution parameters (μ, σ) from input features.

    Architecture:
        Input → LayerNorm → LSTM Encoder → MLP Head → (μ, σ)

    The model outputs a Gaussian (or optionally Student-t) distribution
    parameterized by mean and scale.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        distribution: str = "gaussian",
    ) -> None:
        """Initialize the model.

        Args:
            input_dim: Number of input features.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            distribution: "gaussian" or "student_t".
        """
        super().__init__()

        if distribution not in ("gaussian", "student_t"):
            msg = f"Unknown distribution: {distribution}"
            raise ValueError(msg)

        self.distribution = distribution
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(input_dim)

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # MLP head: hidden → (μ, σ) or (μ, σ, ν)
        head_output_dim = 3 if distribution == "student_t" else 2
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, head_output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Dictionary with:
                "mu": Predicted mean (batch,)
                "sigma": Predicted scale, always positive (batch,)
                "nu": Degrees of freedom (only for student_t) (batch,)
        """
        # Normalize input
        x = self.input_norm(x)

        # Encode sequence
        lstm_out, _ = self.encoder(x)

        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]

        # Predict distribution parameters
        params = self.head(last_hidden)

        mu = params[:, 0]
        sigma = nn.functional.softplus(params[:, 1]) + 1e-6  # Ensure positive

        result = {"mu": mu, "sigma": sigma}

        if self.distribution == "student_t":
            nu = nn.functional.softplus(params[:, 2]) + 2.1  # ν > 2 for finite variance
            result["nu"] = nu

        return result


class ParametricLoss(nn.Module):
    """Negative Log-Likelihood loss for parametric distributions.

    Supports Gaussian and Student-t distributions.
    """

    def __init__(self, distribution: str = "gaussian") -> None:
        super().__init__()
        self.distribution = distribution

    def forward(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NLL loss.

        Args:
            params: Model output dict with "mu", "sigma", and optionally "nu".
            target: True price values (batch,).

        Returns:
            Scalar loss tensor.
        """
        mu = params["mu"]
        sigma = params["sigma"]

        if self.distribution == "gaussian":
            return self._gaussian_nll(mu, sigma, target)
        else:
            nu = params["nu"]
            return self._student_t_nll(mu, sigma, nu, target)

    @staticmethod
    def _gaussian_nll(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian NLL: -log N(target | mu, sigma^2)."""
        variance = sigma ** 2
        nll = 0.5 * (
            torch.log(2 * math.pi * variance)
            + (target - mu) ** 2 / variance
        )
        return nll.mean()

    @staticmethod
    def _student_t_nll(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        nu: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Student-t NLL for heavier tails (better for price spikes)."""
        z = (target - mu) / sigma
        nll = (
            torch.lgamma(nu / 2)
            - torch.lgamma((nu + 1) / 2)
            + 0.5 * torch.log(nu * math.pi)
            + torch.log(sigma)
            + (nu + 1) / 2 * torch.log(1 + z ** 2 / nu)
        )
        return nll.mean()


def create_sequences(
    features: torch.Tensor,
    targets: torch.Tensor,
    seq_len: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sliding window sequences for time-series modeling.

    Args:
        features: Feature tensor (total_steps, input_dim).
        targets: Target tensor (total_steps,).
        seq_len: Number of past steps to use as input.

    Returns:
        Tuple of (sequences, targets):
            sequences: (num_samples, seq_len, input_dim)
            targets: (num_samples,)
    """
    num_samples = len(features) - seq_len
    if num_samples <= 0:
        msg = f"Not enough data: {len(features)} steps < seq_len={seq_len}"
        raise ValueError(msg)

    seqs = []
    tgts = []
    for i in range(num_samples):
        seqs.append(features[i : i + seq_len])
        tgts.append(targets[i + seq_len])

    return torch.stack(seqs), torch.stack(tgts)
