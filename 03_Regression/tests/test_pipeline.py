"""Unit tests for pipeline.py components."""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import Normalizer, OnlineStatistics, hours_to_index


class TestNormalizer:
    """Tests for Normalizer class."""

    def test_normalize_sdo_range(self):
        """SDO normalization should map [0, 255] to [-1, 1]."""
        normalizer = Normalizer()

        # Test boundary values
        data_min = np.array([0.0])
        data_max = np.array([255.0])
        data_mid = np.array([127.5])

        assert np.isclose(normalizer.normalize_sdo(data_min), -1.0)
        assert np.isclose(normalizer.normalize_sdo(data_max), 1.0)
        assert np.isclose(normalizer.normalize_sdo(data_mid), 0.0, atol=0.01)

    def test_normalize_sdo_shape_preserved(self):
        """SDO normalization should preserve input shape."""
        normalizer = Normalizer()

        data = np.random.randint(0, 256, size=(4, 3, 64, 64)).astype(np.float32)
        normalized = normalizer.normalize_sdo(data)

        assert normalized.shape == data.shape

    def test_normalize_omni_zscore(self):
        """OMNI normalization should apply z-score correctly."""
        stat_dict = {
            "test_var": {"mean": 10.0, "std": 2.0}
        }
        normalizer = Normalizer(stat_dict=stat_dict)

        data = np.array([10.0, 12.0, 8.0])
        normalized = normalizer.normalize_omni(data, "test_var")

        expected = np.array([0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_denormalize_omni_inverse(self):
        """Denormalization should be inverse of normalization."""
        stat_dict = {
            "test_var": {"mean": 50.0, "std": 10.0}
        }
        normalizer = Normalizer(stat_dict=stat_dict)

        original = np.array([30.0, 50.0, 70.0, 100.0])
        normalized = normalizer.normalize_omni(original, "test_var")
        recovered = normalizer.denormalize_omni(normalized, "test_var")

        np.testing.assert_array_almost_equal(original, recovered)

    def test_normalize_omni_missing_variable(self):
        """Should raise KeyError for unknown variable."""
        stat_dict = {"known_var": {"mean": 0.0, "std": 1.0}}
        normalizer = Normalizer(stat_dict=stat_dict)

        with pytest.raises(KeyError):
            normalizer.normalize_omni(np.array([1.0]), "unknown_var")

    def test_denormalize_omni_missing_variable(self):
        """Should raise KeyError for unknown variable."""
        stat_dict = {"known_var": {"mean": 0.0, "std": 1.0}}
        normalizer = Normalizer(stat_dict=stat_dict)

        with pytest.raises(KeyError):
            normalizer.denormalize_omni(np.array([1.0]), "unknown_var")


class TestOnlineStatistics:
    """Tests for OnlineStatistics class."""

    def test_single_value(self):
        """Test with single value."""
        stats = OnlineStatistics()
        stats.update(np.array([5.0]))

        assert stats.mean == 5.0
        assert stats.std == 1.0  # Fallback for n < 2

    def test_multiple_values(self):
        """Test with multiple values."""
        stats = OnlineStatistics()
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        stats.update(data)

        assert stats.mean == 6.0
        assert np.isclose(stats.std, np.std(data), atol=0.01)

    def test_batch_updates(self):
        """Test incremental updates produce same result."""
        stats_batch = OnlineStatistics()
        stats_incremental = OnlineStatistics()

        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])
        all_data = np.concatenate([data1, data2])

        # Batch update
        stats_batch.update(all_data)

        # Incremental updates
        stats_incremental.update(data1)
        stats_incremental.update(data2)

        assert np.isclose(stats_batch.mean, stats_incremental.mean)
        assert np.isclose(stats_batch.std, stats_incremental.std)

    def test_ignores_nan_inf(self):
        """Should ignore NaN and Inf values."""
        stats = OnlineStatistics()
        data = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0, -np.inf, 5.0])
        stats.update(data)

        # Only finite values: [1, 2, 3, 4, 5]
        assert stats.n == 5
        assert stats.mean == 3.0

    def test_get_stats_format(self):
        """get_stats should return dict with mean and std."""
        stats = OnlineStatistics()
        stats.update(np.array([1.0, 2.0, 3.0]))

        result = stats.get_stats()

        assert "mean" in result
        assert "std" in result
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)

    def test_multidimensional_input(self):
        """Should flatten and process multidimensional arrays."""
        stats = OnlineStatistics()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats.update(data)

        assert stats.n == 4
        assert stats.mean == 2.5


class TestHoursToIndex:
    """Tests for hours_to_index function."""

    def test_sdo_start_hours(self):
        """Test SDO start hours conversion."""
        # SDO: -96h from reference, 6h interval, base offset -168h
        # Index = (-96 - (-168)) / 6 = 72 / 6 = 12
        assert hours_to_index(-96, 6, -168) == 12

    def test_sdo_end_hours(self):
        """Test SDO end hours conversion."""
        # SDO: 0h (reference time), 6h interval, base offset -168h
        # Index = (0 - (-168)) / 6 = 168 / 6 = 28
        assert hours_to_index(0, 6, -168) == 28

    def test_omni_input_start(self):
        """Test OMNI input start hours conversion."""
        # OMNI input: -96h, 3h interval, base offset -168h
        # Index = (-96 - (-168)) / 3 = 72 / 3 = 24
        assert hours_to_index(-96, 3, -168) == 24

    def test_omni_input_end(self):
        """Test OMNI input end hours conversion."""
        # OMNI input: +72h, 3h interval, base offset -168h
        # Index = (72 - (-168)) / 3 = 240 / 3 = 80
        assert hours_to_index(72, 3, -168) == 80

    def test_omni_target_start(self):
        """Test OMNI target start hours conversion."""
        # OMNI target: +72h, 3h interval, base offset -168h
        # Index = (72 - (-168)) / 3 = 240 / 3 = 80
        assert hours_to_index(72, 3, -168) == 80

    def test_omni_target_end(self):
        """Test OMNI target end hours conversion."""
        # OMNI target: +144h, 3h interval, base offset -168h
        # Index = (144 - (-168)) / 3 = 312 / 3 = 104
        assert hours_to_index(144, 3, -168) == 104


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
