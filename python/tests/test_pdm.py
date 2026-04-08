import numpy as np
import pytest
from phasedm import pdm, beta_test


# ===== pdm() basic functionality =====


class TestPdmBasic:
    def test_pdm_returns_correct_shapes(self):
        t = np.linspace(0, 10, 500)
        y = np.sin(2 * np.pi * t)
        n_freqs = 100
        freq, theta = pdm(t, y, 0.5, 2.0, n_freqs, n_bins=10)
        assert freq.shape == (n_freqs,)
        assert theta.shape == (n_freqs,)

    def test_pdm_basic_sine_recovery(self):
        """sin(2*pi*t) has period=1s, frequency=1Hz. PDM should find minimum near 1Hz."""
        t = np.linspace(0, 20, 10000)
        y = np.sin(2 * np.pi * t) + np.random.default_rng(42).normal(0, 0.1, len(t))
        freq, theta = pdm(t, y, 0.5, 1.5, 1000, n_bins=10)
        best_freq = freq[np.argmin(theta)]
        assert abs(best_freq - 1.0) < 0.05, f"Expected ~1.0 Hz, got {best_freq}"

    def test_pdm_with_sigma(self):
        t = np.linspace(0, 10, 500)
        y = np.sin(2 * np.pi * t)
        sigma = np.full(500, 0.01)
        freq, theta = pdm(t, y, 0.5, 2.0, 50, sigma=sigma, n_bins=10)
        assert freq.shape == (50,)
        assert theta.shape == (50,)

    def test_pdm_single_frequency(self):
        t = np.linspace(0, 10, 500)
        y = np.sin(2 * np.pi * t)
        freq, theta = pdm(t, y, 1.0, 1.0, 1, n_bins=10)
        assert freq.shape == (1,)
        assert theta.shape == (1,)

    def test_pdm_theta_range(self):
        t = np.linspace(0, 10, 500)
        y = np.sin(2 * np.pi * t) + np.random.default_rng(0).normal(0, 0.1, 500)
        freq, theta = pdm(t, y, 0.5, 2.0, 100, n_bins=10)
        assert np.all(theta > 0), "All theta values should be positive"


# ===== pdm() input type variants =====


class TestPdmInputTypes:
    def test_pdm_datetime64_input(self):
        base = np.datetime64("2020-01-01T00:00:00", "ns")
        dt = np.array(
            [base + np.timedelta64(int(i * 1e9), "ns") for i in range(500)],
            dtype="datetime64[ns]",
        )
        y = np.sin(2 * np.pi * np.arange(500) / 100.0)
        freq, theta = pdm(dt, y, 0.005, 0.02, 50, n_bins=10)
        assert freq.shape == (50,)
        assert theta.shape == (50,)

    def test_pdm_float32_time_coercion(self):
        t = np.linspace(0, 10, 500).astype(np.float32)
        y = np.sin(2 * np.pi * t.astype(np.float64))
        freq, theta = pdm(t, y, 0.5, 2.0, 50, n_bins=10)
        assert freq.shape == (50,)

    def test_pdm_astropy_time(self):
        astropy_time = pytest.importorskip("astropy.time")
        from astropy.time import Time

        times = Time(np.linspace(2460000, 2460010, 500), format="jd")
        y = np.sin(2 * np.pi * np.linspace(0, 10, 500))
        freq, theta = pdm(times, y, 0.5, 2.0, 50, n_bins=10)
        assert freq.shape == (50,)

    def test_pdm_astropy_quantity_signal(self):
        astropy_units = pytest.importorskip("astropy.units")
        import astropy.units as u

        t = np.linspace(0, 10, 500)
        y = np.sin(2 * np.pi * t) * u.electron / u.s
        sigma = np.full(500, 0.01) * u.electron / u.s
        freq, theta = pdm(t, y, 0.5, 2.0, 50, sigma=sigma, n_bins=10)
        assert freq.shape == (50,)


# ===== pdm() error cases =====


class TestPdmErrors:
    def test_pdm_mismatched_lengths(self):
        t = np.linspace(0, 10, 100)
        y = np.ones(50)
        with pytest.raises(ValueError):
            pdm(t, y, 0.5, 2.0, 10, n_bins=5)

    def test_pdm_sigma_mismatched_length(self):
        t = np.linspace(0, 10, 100)
        y = np.sin(t)
        sigma = np.ones(50)
        with pytest.raises(ValueError):
            pdm(t, y, 0.5, 2.0, 10, sigma=sigma, n_bins=5)

    def test_pdm_min_freq_zero(self):
        t = np.linspace(0, 10, 100)
        y = np.sin(t)
        with pytest.raises(ValueError):
            pdm(t, y, 0.0, 2.0, 10, n_bins=5)

    def test_pdm_negative_freq(self):
        t = np.linspace(0, 10, 100)
        y = np.sin(t)
        with pytest.raises(ValueError):
            pdm(t, y, -1.0, 2.0, 10, n_bins=5)

    def test_pdm_min_greater_than_max(self):
        t = np.linspace(0, 10, 100)
        y = np.sin(t)
        with pytest.raises(ValueError):
            pdm(t, y, 5.0, 2.0, 10, n_bins=5)

    def test_pdm_n_bins_zero(self):
        t = np.linspace(0, 10, 100)
        y = np.sin(t)
        with pytest.raises(ValueError):
            pdm(t, y, 0.5, 2.0, 10, n_bins=0)

    def test_pdm_n_bins_exceeds_data(self):
        t = np.linspace(0, 10, 10)
        y = np.sin(t)
        with pytest.raises(ValueError):
            pdm(t, y, 0.5, 2.0, 10, n_bins=10)

    def test_pdm_invalid_time_type(self):
        t = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = np.sin(np.array(t))
        with pytest.raises(TypeError):
            pdm(t, y, 0.5, 2.0, 10, n_bins=2)


# ===== beta_test() tests =====


class TestBetaTest:
    def test_beta_test_basic(self):
        result = beta_test(100, 10, 0.5)
        assert 0.0 < result < 1.0

    def test_beta_test_p_zero(self):
        result = beta_test(100, 10, 0.0)
        assert result == 0.0

    def test_beta_test_p_one(self):
        result = beta_test(100, 10, 1.0)
        assert result == 1.0

    def test_beta_test_p_negative(self):
        with pytest.raises(ValueError):
            beta_test(100, 10, -0.1)

    def test_beta_test_p_above_one(self):
        with pytest.raises(ValueError):
            beta_test(100, 10, 1.1)

    def test_beta_test_n_less_than_nbins(self):
        with pytest.raises(ValueError):
            beta_test(5, 10, 0.5)

    def test_beta_test_n_equals_nbins(self):
        with pytest.raises(ValueError):
            beta_test(10, 10, 0.5)

    def test_beta_test_monotonic(self):
        p_values = [0.01, 0.05, 0.1, 0.5, 0.9]
        results = [beta_test(1000, 10, p) for p in p_values]
        for i in range(1, len(results)):
            assert results[i] > results[i - 1], (
                f"beta_test should be monotonically increasing: "
                f"p={p_values[i-1]}→{results[i-1]}, p={p_values[i]}→{results[i]}"
            )
