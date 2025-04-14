import numpy as np
from typing import Optional, Tuple

def pdm(
    time: np.ndarray,
    signal: np.ndarray,
    min_freq: float,
    max_freq: float,
    n_freqs: int,
    sigma: Optional[np.ndarray] = None,  # None by default
    n_bins: int = 10,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Phase Dispersion Minimisation (PDM) analysis on a time series signal.

    This function computes the periodogram decomposition, which allows for spectral
    analysis of the input signal across a specified frequency range.

    Parameters
    ----------
    time : numpy.ndarray
        Array of time points corresponding to the signal.
        Must be a 1D numpy array of numeric or be a datetime64[ns] values.
        Cannot contain non-numeric values.

    signal : numpy.ndarray
        Input signal to be analyzed.
        Must be a 1D numpy array of float values representing the signal amplitudes.

    min_freq : float
        Minimum frequency for analysis.
        Must be a positive float value (> 0).

    max_freq : float
        Maximum frequency for analysis.
        Must be greater than or equal to min_freq.

    n_freqs : int
        Number of frequency points to compute in the analysis.
        Must be a positive integer (> 0).

    sigma: np.ndarray, optional
        Array of measurement uncertainty corresponding to the signal.
        Must be a 1D numpy array of numeric values.
        Cannot contain non-numeric values.

    n_bins : int, optional
        Number of bins to use in the analysis.
        Must be a positive integer.
        Default is 10.

    verbose : int, optional
        Verbosity level for timing information:
        - 0 (default): Silent, no timing output
        - Any non-zero value: Outputs timing information during computation

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - freqs: np.ndarray - Frequency values (first element) corresponding to the computed theta statistics
        - theta: np.ndarray - PDM statistic values (second element), where lower values indicate stronger periodicity

    Raises
    ------
    TypeError
        If input parameters do not meet the specified constraints:
        - time or signal are not 1D numpy arrays
        - time must be an array of floats or datetime64

    ValueError
        If input parameters do not meet the specified constraints:
        - min_freq is not positive
        - max_freq is less than min_freq
        - n_freqs is not positive
        - n_bins is not a positive integer

    Notes
    -----
    - Ensure input arrays have matching lengths
    - The function does not require uniform time sampling
    - Computational complexity increases with n_freqs and n_bins

    Examples
    --------
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 1000)
    >>> signal = np.sin(2 * np.pi * 2 * time) + np.random.normal(0, 0.1, time.shape)
    >>> theta,freqs = pdm(time, signal, min_freq=1, max_freq=10, n_freqs=100, n_bins=20)
    """

def beta_test(n: int, n_bins: int, p: float) -> float:
    """
    Calculate a significance theta value for a given p-value using the inverse incomplete beta function.

    This function computes the critical theta value corresponding to a specified probability level
    in Phase Dispersion Minimisation (PDM) analysis. It uses the inverse incomplete beta function
    to determine the threshold at which a periodogram peak can be considered statistically significant.

    Parameters
    ----------
    n : int
        Number of data points in the time series.
        Must be a positive integer (> 0).

    n_bins : int
        Number of phase bins used in the PDM analysis.
        Must be a positive integer (> 0).

    p : float
        Target probability value (p-value).
        Must be in the range (0, 1).
        Typically represents the desired significance level (e.g., 0.01 for 99% confidence).

    Returns
    -------
    float
        The critical theta value corresponding to the specified p-value.
        Values below this threshold in a PDM periodogram are considered statistically significant
        at the specified probability level.

    Raises
    ------
    ValueError
        If input parameters do not meet the specified constraints:
        - n is not a positive integer
        - n_bins is not a positive integer
        - p is not in the range (0, 1)

    Notes
    -----
    - The inverse incomplete beta function is used to determine the critical value
    - The degrees of freedom for the beta function are derived from n and n_bins
    - Lower theta values indicate stronger periodic signals

    Examples
    --------
    >>> critical_theta = beta_test(1000, 20, 0.01)
    >>> print(f"Threshold value at 99% confidence: {critical_theta:.4f}")
    Threshold value at 99% confidence: 0.8752

    >>> # Use with PDM analysis
    >>> theta, freqs = pdm(time, signal, min_freq=1, max_freq=10, n_freqs=100, n_bins=20)
    >>> significant_freqs = freqs[theta < beta_test(len(time), 20, 0.01)]
    """
