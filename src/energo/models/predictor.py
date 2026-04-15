"""Inference predictor for trained parametric models.

Loads a trained model and produces probabilistic price forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from energo.models.parametric import ParametricPriceModel

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PriceForecast:
    """A single probabilistic price forecast."""

    mean: float
    std: float
    ci_lower_90: float
    ci_upper_90: float
    ci_lower_95: float
    ci_upper_95: float

    @property
    def ci_width_90(self) -> float:
        return self.ci_upper_90 - self.ci_lower_90


class Predictor:
    """Loads a trained model and produces price distribution forecasts."""

    def __init__(
        self,
        model: ParametricPriceModel,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        distribution: str = "gaussian",
    ) -> Predictor:
        """Create a Predictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            input_dim: Number of input features.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            distribution: Distribution type.

        Returns:
            Initialized Predictor.
        """
        model = ParametricPriceModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            distribution=distribution,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info("Loaded model from %s", checkpoint_path)
        return cls(model=model, device=device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> list[PriceForecast]:
        """Generate probabilistic forecasts for input sequences.

        Args:
            x: Input tensor (batch, seq_len, input_dim).

        Returns:
            List of PriceForecast objects.
        """
        x = x.to(self.device)
        params = self.model(x)

        mu = params["mu"].cpu().numpy()
        sigma = params["sigma"].cpu().numpy()

        forecasts = []
        for i in range(len(mu)):
            m = float(mu[i])
            s = float(sigma[i])

            forecasts.append(
                PriceForecast(
                    mean=m,
                    std=s,
                    ci_lower_90=m - 1.645 * s,
                    ci_upper_90=m + 1.645 * s,
                    ci_lower_95=m - 1.960 * s,
                    ci_upper_95=m + 1.960 * s,
                )
            )

        return forecasts

    @torch.no_grad()
    def predict_raw(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """Get raw distribution parameters.

        Args:
            x: Input tensor (batch, seq_len, input_dim).

        Returns:
            Dict with "mu" and "sigma" numpy arrays.
        """
        x = x.to(self.device)
        params = self.model(x)

        return {
            k: v.cpu().numpy()
            for k, v in params.items()
        }
