"""Tests for the parametric price model."""

from __future__ import annotations

import pytest
import torch

from energo.models.parametric import (
    ParametricLoss,
    ParametricPriceModel,
    create_sequences,
)


class TestParametricPriceModel:
    """Tests for model forward pass and output shape."""

    @pytest.fixture()
    def model(self) -> ParametricPriceModel:
        return ParametricPriceModel(
            input_dim=10,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
        )

    def test_forward_shape(self, model: ParametricPriceModel) -> None:
        """Output should have mu and sigma with correct batch size."""
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 10)

        output = model(x)

        assert "mu" in output
        assert "sigma" in output
        assert output["mu"].shape == (batch_size,)
        assert output["sigma"].shape == (batch_size,)

    def test_sigma_positive(self, model: ParametricPriceModel) -> None:
        """Sigma must always be positive."""
        x = torch.randn(8, 16, 10)
        output = model(x)

        assert (output["sigma"] > 0).all()

    def test_student_t_output(self) -> None:
        """Student-t model should also output nu."""
        model = ParametricPriceModel(
            input_dim=5,
            hidden_dim=16,
            num_layers=1,
            distribution="student_t",
        )
        x = torch.randn(4, 8, 5)
        output = model(x)

        assert "nu" in output
        assert (output["nu"] > 2).all()  # finite variance requires ν > 2

    def test_invalid_distribution(self) -> None:
        with pytest.raises(ValueError, match="Unknown distribution"):
            ParametricPriceModel(input_dim=5, distribution="poisson")

    def test_gradient_flow(self, model: ParametricPriceModel) -> None:
        """Verify gradients flow through the model."""
        x = torch.randn(4, 16, 10)
        output = model(x)
        loss = output["mu"].sum() + output["sigma"].sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestParametricLoss:
    """Tests for NLL loss computation."""

    def test_gaussian_nll_finite(self) -> None:
        loss_fn = ParametricLoss(distribution="gaussian")
        params = {
            "mu": torch.tensor([10.0, 11.0]),
            "sigma": torch.tensor([1.0, 2.0]),
        }
        target = torch.tensor([10.5, 12.0])

        loss = loss_fn(params, target)

        assert torch.isfinite(loss)
        assert loss > 0

    def test_gaussian_nll_perfect_prediction(self) -> None:
        """Loss should be lower when prediction matches target."""
        loss_fn = ParametricLoss(distribution="gaussian")

        params_good = {
            "mu": torch.tensor([10.0]),
            "sigma": torch.tensor([0.1]),
        }
        params_bad = {
            "mu": torch.tensor([15.0]),
            "sigma": torch.tensor([0.1]),
        }
        target = torch.tensor([10.0])

        loss_good = loss_fn(params_good, target)
        loss_bad = loss_fn(params_bad, target)

        assert loss_good < loss_bad

    def test_student_t_nll_finite(self) -> None:
        loss_fn = ParametricLoss(distribution="student_t")
        params = {
            "mu": torch.tensor([10.0]),
            "sigma": torch.tensor([1.0]),
            "nu": torch.tensor([5.0]),
        }
        target = torch.tensor([10.5])

        loss = loss_fn(params, target)
        assert torch.isfinite(loss)


class TestCreateSequences:
    """Tests for sliding window sequence creation."""

    def test_output_shapes(self) -> None:
        features = torch.randn(100, 10)
        targets = torch.randn(100)

        seqs, tgts = create_sequences(features, targets, seq_len=16)

        assert seqs.shape == (84, 16, 10)  # 100 - 16 = 84
        assert tgts.shape == (84,)

    def test_no_future_leakage(self) -> None:
        """Target at index i should correspond to step i+seq_len."""
        features = torch.arange(50).float().unsqueeze(1)
        targets = torch.arange(50).float()

        seqs, tgts = create_sequences(features, targets, seq_len=10)

        # Target for first sequence should be at index 10
        assert tgts[0].item() == 10.0
        # Last feature in first sequence should be at index 9
        assert seqs[0, -1, 0].item() == 9.0

    def test_insufficient_data_raises(self) -> None:
        features = torch.randn(5, 10)
        targets = torch.randn(5)

        with pytest.raises(ValueError, match="Not enough data"):
            create_sequences(features, targets, seq_len=10)
