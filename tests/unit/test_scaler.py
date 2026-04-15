"""Tests for feature scaler — additional coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energo.features.scaler import FeatureScaler


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.standard_normal(100),
        "b": rng.uniform(0, 100, 100),
        "c": rng.integers(0, 10, 100).astype(float),
    })


class TestRobustScaler:
    """Tests for robust scaling mode."""

    def test_robust_fit_transform(self, sample_df: pd.DataFrame) -> None:
        scaler = FeatureScaler(method="robust")
        scaler.fit(sample_df, ["a", "b", "c"])
        result = scaler.transform(sample_df)

        # Median-centered, IQR-scaled
        assert abs(result["a"].median()) < 0.5  # Close to 0

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            FeatureScaler(method="minmax")


class TestInverseTransform:
    """Tests for inverse transform round-trip."""

    def test_standard_roundtrip(self, sample_df: pd.DataFrame) -> None:
        scaler = FeatureScaler(method="standard")
        scaler.fit(sample_df, ["a", "b"])

        scaled = scaler.transform(sample_df)
        recovered = scaler.inverse_transform(scaled)

        np.testing.assert_array_almost_equal(
            sample_df["a"].values, recovered["a"].values, decimal=10,
        )
        np.testing.assert_array_almost_equal(
            sample_df["b"].values, recovered["b"].values, decimal=10,
        )

    def test_robust_roundtrip(self, sample_df: pd.DataFrame) -> None:
        scaler = FeatureScaler(method="robust")
        scaler.fit(sample_df, ["a", "b"])

        scaled = scaler.transform(sample_df)
        recovered = scaler.inverse_transform(scaled)

        np.testing.assert_array_almost_equal(
            sample_df["a"].values, recovered["a"].values, decimal=10,
        )


class TestScalerEdgeCases:
    """Tests for edge cases."""

    def test_unfitted_transform_raises(self) -> None:
        scaler = FeatureScaler()
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(df)

    def test_unfitted_inverse_raises(self) -> None:
        scaler = FeatureScaler()
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.inverse_transform(df)

    def test_unfitted_save_raises(self, tmp_path) -> None:
        scaler = FeatureScaler()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.save(tmp_path / "scaler.json")

    def test_empty_column(self) -> None:
        scaler = FeatureScaler()
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        scaler.fit(df, ["a"])
        assert scaler.is_fitted

    def test_missing_column_ignored(self, sample_df: pd.DataFrame) -> None:
        scaler = FeatureScaler()
        scaler.fit(sample_df, ["a", "nonexistent"])
        result = scaler.transform(sample_df)
        # "a" should be transformed, "nonexistent" ignored
        assert "a" in result.columns
