import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as f_distribution

def pdm_test(times, measurements, period_range, n_bins=10, n_covers=2):
    """
    Implementation of Phase Dispersion Minimisation (PDM) for period finding
    
    Parameters:
    -----------
    times : array-like
        Array of observation times
    measurements : array-like
        Array of measurements corresponding to the times
    period_range : tuple or array-like
        (min_period, max_period, n_periods) or array of periods to test
    n_bins : int, optional
        Number of phase bins to use (default: 10)
    n_covers : int, optional
        Number of covers (overlapping sets of bins) to use (default: 2)
    
    Returns:
    --------
    periods : numpy.ndarray
        Array of tested periods
    thetas : numpy.ndarray
        Array of PDM statistics (smaller values indicate better periods)
    best_period : float
        Period with the minimum theta statistic
    """
    # Convert inputs to numpy arrays
    times = np.asarray(times)
    measurements = np.asarray(measurements)
    
    # Generate periods to test
    if len(period_range) == 3:
        periods = np.linspace(period_range[0], period_range[1], int(period_range[2]))
    else:
        periods = np.asarray(period_range)
    
    # Calculate overall variance of the data
    overall_variance = np.var(measurements, ddof=1)
    if overall_variance == 0:
        raise ValueError("No variance in measurements")
    
    # Initialize array to store PDM statistics
    thetas = np.zeros_like(periods)
    
    # Loop through periods
    for i, period in enumerate(periods):
        # Calculate phases
        phases = (times / period) % 1.0
        
        # Initialize variables for this period
        s2_sum = 0.0
        M_sum = 0
        
        # Create n_covers samples
        for cover in range(n_covers):
            # Calculate bin edges for this cover (offset between covers)
            offset = cover / (n_covers * n_bins)
            bin_edges = np.linspace(0, 1, n_bins + 1) + offset
            
            # Handle wraparound for last cover
            if cover == n_covers - 1:
                bin_edges = bin_edges % 1.0
                # Sort bin edges and phases for the last cover
                sort_idx = np.argsort(bin_edges)
                bin_edges = bin_edges[sort_idx]
                # Add 1.0 to the last edge to ensure it's greater than all phases
                bin_edges[-1] = 1.0
            
            # Get bin assignments for each phase
            bin_indices = np.digitize(phases, bin_edges) - 1
            
            # Calculate variance in each bin
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                n_points = np.sum(bin_mask)
                
                # Skip empty bins
                if n_points <= 1:
                    continue
                
                # Calculate variance of points in this bin
                bin_variance = np.var(measurements[bin_mask], ddof=1)
                s2_sum += bin_variance * (n_points - 1)
                M_sum += n_points - 1
        
        # Calculate theta statistic for this period
        if M_sum == 0:
            thetas[i] = 1.0  # Default to maximum value if no valid bins
        else:
            thetas[i] = s2_sum / (overall_variance * M_sum)
    
    # Find best period (minimum theta)
    best_idx = np.argmin(thetas)
    best_period = periods[best_idx]
    
    return periods, thetas, best_period

def calculate_significance(theta, n_samples, n_bins, n_covers):
    """
    Calculate the statistical significance of a PDM result
    
    Parameters:
    -----------
    theta : float
        PDM statistic value
    n_samples : int
        Number of data points
    n_bins : int
        Number of bins used
    n_covers : int
        Number of covers used
    
    Returns:
    --------
    significance : float
        Probability that the result is due to chance
    """
    # Calculate degrees of freedom
    n_params = 1  # We're estimating one parameter (the period)
    n_bins_eff = n_bins * n_covers
    
    # Prevent division by zero
    if n_bins_eff <= 1 or n_samples <= n_bins_eff + n_params:
        return 1.0
    
    # Calculate F-statistic
    nu1 = n_samples - n_bins_eff
    nu2 = (n_bins_eff - 1)
    
    if nu1 <= 0 or nu2 <= 0:
        return 1.0
    
    f_value = (1.0 - theta) / theta * nu1 / nu2
    
    # Calculate significance using F-distribution
    significance = 1.0 - f_distribution.cdf(f_value, nu2, nu1)
    return significance

def plot_pdm_results(periods, thetas, best_period, times, measurements):
    """Plot PDM results and phase-folded data"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot PDM spectrum
    ax1.plot(periods, thetas, 'k-')
    ax1.axvline(best_period, color='r', linestyle='--', 
                label=f'Best period: {best_period:.5f}')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Theta (smaller is better)')
    ax1.set_title('PDM Spectrum')
    ax1.legend()
    
    # Plot phase-folded data
    phases = (times / best_period) % 1.0
    # Sort by phase for the plot
    sort_idx = np.argsort(phases)
    phases = phases[sort_idx]
    folded_measurements = measurements[sort_idx]
    
    # Plot once
    ax2.scatter(phases, folded_measurements, alpha=0.5, label='Folded data')
    # Plot again shifted by 1 phase to see the continuity
    ax2.scatter(phases + 1, folded_measurements, alpha=0.5, color='gray')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Measurement')
    ax2.set_title(f'Data Folded with Period = {best_period:.5f}')
    ax2.set_xlim(0, 2)
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Create synthetic data - a sine wave with some noise and uneven sampling
    np.random.seed(42)
    true_period = 0.75
    n_points = 100
    
    # Create uneven time sampling
    times = np.sort(np.random.uniform(0, 10, n_points))
    
    # Create signal with the known period
    signal = np.sin(2 * np.pi * times / true_period)
    
    # Add some noise
    noise = np.random.normal(0, 0.2, n_points)
    measurements = signal + noise
    
    # Define period range to search (min, max, number of periods)
    period_range = (0.5, 1.0, 1000)
    
    # Run PDM algorithm
    periods, thetas, best_period = pdm_test(times, measurements, period_range)
    
    # Calculate significance
    min_theta = np.min(thetas)
    significance = calculate_significance(min_theta, n_points, 10, 2)
    
    print(f"True period: {true_period}")
    print(f"Best period found: {best_period}")
    print(f"PDM statistic (theta): {min_theta:.6f}")
    print(f"Significance (p-value): {significance:.6e}")
    
    # Plot the results
    plt.figure(figsize=(10, 10))
    plot_pdm_results(periods, thetas, best_period, times, measurements)
    plt.show()