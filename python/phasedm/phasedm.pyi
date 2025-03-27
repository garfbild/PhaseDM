import numpy as np

def pdm(
    time: np.ndarray,
    signal: np.ndarray,
    min_freq: float,
    max_freq: float,
    n_freqs: int,
    n_bins: int = 10,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
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
    (numpy.ndarray, numpy.ndarray)
        Resulting periodogram decomposition array.
        The shape and specific content depend on the input parameters and
        internal implementation of the PDM algorithm.

    Raises
    ------
    ValueError
        If input parameters do not meet the specified constraints:
        - time or signal are not 1D numpy arrays
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
    "stuff"
