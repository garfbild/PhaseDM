use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::time_section;

pub fn generate_freqs(min_freq: f64, max_freq: f64, n_freqs: u64) -> Vec<f64> {
    if n_freqs <= 1 {
        return vec![min_freq];
    }

    let step = (max_freq - min_freq) / (n_freqs as f64 - 1.0);
    let mut result = Vec::with_capacity(n_freqs as usize);

    (0..n_freqs as usize)
        .into_par_iter()
        .map(|i| min_freq + (i as f64) * step)
        .collect_into_vec(&mut result);

    result
}

fn compute_phase(time: ArrayView1<f64>, inv_freq: f64) -> Vec<f64> {
    // try and use par_iter for phase calculation otherwise use serial iter
    let mut result = Vec::with_capacity(time.len());
    //rem euclid is more precise but also slow
    if let Some(time_slice) = time.as_slice() {
        time_slice
            .par_iter()
            .map(|&x| x % inv_freq)
            .collect_into_vec(&mut result);
    } else {
        result.extend(time.iter().map(|&x| x % inv_freq));
    }

    result
}

fn binning_operation(phase: &Vec<f64>, inv_freq: f64, n_bins: u64) -> Vec<u64> {
    let s = n_bins as f64 / inv_freq;
    // for some reason it is possible for x = ~inv_freq and so we get an index out of range error
    // modulo shouldn't be necassary
    phase.par_iter().map(|&x| (x * s) as u64 % n_bins).collect()
}

fn bin_count_sum_operation(
    bin_counts: &mut Vec<u64>,
    bin_sums: &mut Vec<f64>,
    bin_index: &Vec<u64>,
    signal: &ArrayView1<f64>,
) -> () {
    for (i, &bin) in bin_index.iter().enumerate() {
        bin_counts[bin as usize] += 1;
        bin_sums[bin as usize] += signal[i];
    }
}

fn squared_diff_calculation(
    bin_squared_difference: &mut Vec<f64>,
    squared_difference: &mut f64,
    bin_index: &Vec<u64>,
    bin_means: &Vec<f64>,
    signal: &ArrayView1<f64>,
    mean: &f64,
) {
    for (i, &bin) in bin_index.iter().enumerate() {
        // Do I check both independantly??????
        bin_squared_difference[bin as usize] += f64::powi(bin_means[bin as usize] - signal[i], 2);
        *squared_difference += f64::powi(mean - signal[i], 2);
    }
}

fn squared_diff_sigma_calculation(
    bin_squared_difference: &mut Vec<f64>,
    squared_difference: &mut f64,
    bin_index: &Vec<u64>,
    bin_means: &Vec<f64>,
    signal: &ArrayView1<f64>,
    mean: &f64,
    sigma: &ArrayView1<f64>,
) {
    for (i, &bin) in bin_index.iter().enumerate() {
        let bin_diff = bin_means[bin as usize] - signal[i];
        if bin_diff.abs() >= sigma[i].abs() {
            bin_squared_difference[bin as usize] += f64::powi(bin_diff, 2);
        }
        let diff = mean - signal[i];
        if diff.abs() >= sigma[i].abs() {
            *squared_difference += f64::powi(diff, 2);
        }
    }
}

fn compute_theta_core(
    time: ArrayView1<f64>,
    signal: ArrayView1<f64>,
    freq: f64,
    n_bins: u64,
) -> PyResult<(Vec<u64>, Vec<f64>, f64)> {
    let inv_freq = if freq != 0.0 {
        1.0 / freq
    } else {
        return Err(PyValueError::new_err(format!(
            "cannot evaluate frequency = 0. undefined behaviour."
        )));
    };

    let phase: Vec<f64> = time_section!("compute_phase", compute_phase(time, inv_freq));

    let bin_index: Vec<u64> = time_section!("binning_operation", {
        binning_operation(&phase, inv_freq, n_bins)
    });

    let mut bin_counts = vec![0_u64; n_bins as usize];
    let mut bin_sums = vec![0_f64; n_bins as usize];
    time_section!("bin_count_sum_operation", {
        bin_count_sum_operation(&mut bin_counts, &mut bin_sums, &bin_index, &signal);
    });

    // calculate the mean of each of the bins
    let bin_means: Vec<f64> = bin_sums
        .iter()
        .zip(bin_counts.iter())
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / count as f64
            } else {
                0.0 // or f64::NAN
            }
        })
        .collect();

    // calculate the total mean
    let mean = bin_sums.iter().sum::<f64>() / (bin_counts.iter().sum::<u64>() as f64);

    Ok((bin_index, bin_means, mean))
}

// Version without sigma
pub fn compute_theta(
    time: ArrayView1<f64>,
    signal: ArrayView1<f64>,
    freq: f64,
    n_bins: u64,
) -> PyResult<f64> {
    let (bin_index, bin_means, mean) = compute_theta_core(time, signal, freq, n_bins)?;

    let mut bin_squared_difference = vec![0_f64; n_bins as usize];
    let mut squared_difference: f64 = 0.0;

    time_section!("squared_diff_calculation", {
        squared_diff_calculation(
            &mut bin_squared_difference,
            &mut squared_difference,
            &bin_index,
            &bin_means,
            &signal,
            &mean,
        );
    });

    if squared_difference == 0.0 {
        return Err(PyValueError::new_err(
            "total squared difference is zero (all signal values are identical), cannot compute theta",
        ));
    }

    Ok(bin_squared_difference.iter().sum::<f64>() / squared_difference)
}

// Version with sigma
pub fn compute_theta_sigma(
    time: ArrayView1<f64>,
    signal: ArrayView1<f64>,
    sigma: ArrayView1<f64>,
    freq: f64,
    n_bins: u64,
) -> PyResult<f64> {
    let (bin_index, bin_means, mean) = compute_theta_core(time, signal, freq, n_bins)?;

    let mut bin_squared_difference = vec![0_f64; n_bins as usize];
    let mut squared_difference: f64 = 0.0;

    time_section!("squared_diff_calculation", {
        squared_diff_sigma_calculation(
            &mut bin_squared_difference,
            &mut squared_difference,
            &bin_index,
            &bin_means,
            &signal,
            &mean,
            &sigma,
        );
    });

    if squared_difference == 0.0 {
        return Err(PyValueError::new_err(
            "total squared difference is zero (all deviations within sigma), cannot compute theta",
        ));
    }

    Ok(bin_squared_difference.iter().sum::<f64>() / squared_difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    // ===== generate_freqs tests =====

    #[test]
    fn test_generate_freqs() {
        let freqs = generate_freqs(10.0, 20.0, 5);
        assert_eq!(freqs.len(), 5);
        assert_relative_eq!(freqs[0], 10.0);
        assert_relative_eq!(freqs[4], 20.0);
        assert_relative_eq!(freqs[2], 15.0);

        let single_freq = generate_freqs(10.0, 20.0, 1);
        assert_eq!(single_freq.len(), 1);
        assert_relative_eq!(single_freq[0], 10.0);
    }

    #[test]
    fn test_generate_freqs_zero_n() {
        let freqs = generate_freqs(5.0, 10.0, 0);
        assert_eq!(freqs.len(), 1);
        assert_relative_eq!(freqs[0], 5.0);
    }

    #[test]
    fn test_generate_freqs_same_bounds() {
        let freqs = generate_freqs(7.0, 7.0, 1);
        assert_eq!(freqs.len(), 1);
        assert_relative_eq!(freqs[0], 7.0);
    }

    #[test]
    fn test_generate_freqs_two() {
        let freqs = generate_freqs(1.0, 5.0, 2);
        assert_eq!(freqs.len(), 2);
        assert_relative_eq!(freqs[0], 1.0);
        assert_relative_eq!(freqs[1], 5.0);
    }

    #[test]
    fn test_generate_freqs_spacing() {
        let freqs = generate_freqs(0.0, 10.0, 11);
        assert_eq!(freqs.len(), 11);
        for i in 1..freqs.len() {
            let spacing = freqs[i] - freqs[i - 1];
            assert_relative_eq!(spacing, 1.0, epsilon = 1e-10);
        }
    }

    // ===== compute_phase tests =====

    #[test]
    fn test_compute_phase() {
        let time = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let phases = compute_phase(time.view(), 3.0);

        assert_eq!(phases.len(), 5);
        assert_relative_eq!(phases[0], 0.0);
        assert_relative_eq!(phases[1], 1.0);
        assert_relative_eq!(phases[2], 2.0);
        assert_relative_eq!(phases[3], 0.0);
        assert_relative_eq!(phases[4], 1.0);
    }

    #[test]
    fn test_compute_phase_single_element() {
        let time = Array1::from_vec(vec![2.5]);
        let phases = compute_phase(time.view(), 3.0);
        assert_eq!(phases.len(), 1);
        assert_relative_eq!(phases[0], 2.5);
    }

    #[test]
    fn test_compute_phase_all_zeros() {
        let time = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let phases = compute_phase(time.view(), 5.0);
        assert_eq!(phases.len(), 3);
        for &p in &phases {
            assert_relative_eq!(p, 0.0);
        }
    }

    // ===== binning_operation tests =====

    #[test]
    fn test_binning_operation() {
        let phase: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let bins = binning_operation(&phase, 3.0, 6);

        assert_eq!(bins.len(), 6);
        assert_eq!(bins[0], 0);
        assert_eq!(bins[1], 1);
        assert_eq!(bins[2], 2);
        assert_eq!(bins[3], 3);
        assert_eq!(bins[4], 4);
        assert_eq!(bins[5], 5);
    }

    #[test]
    fn test_binning_single_bin() {
        let phase: Vec<f64> = vec![0.0, 0.3, 0.7, 0.99];
        let bins = binning_operation(&phase, 1.0, 1);
        assert_eq!(bins.len(), 4);
        for &b in &bins {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn test_binning_phase_at_boundary() {
        // Phase exactly at inv_freq boundary wraps via modulo
        let inv_freq = 2.0;
        let n_bins: u64 = 4;
        let phase: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5];
        let bins = binning_operation(&phase, inv_freq, n_bins);
        assert_eq!(bins.len(), 4);
        // All bin indices should be in [0, n_bins)
        for &b in &bins {
            assert!(b < n_bins);
        }
    }

    // ===== bin_count_sum_operation tests =====

    #[test]
    fn test_bin_count_sum_operation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let bin_index = vec![0, 1, 0, 1];
        let mut bin_counts = vec![0; 2];
        let mut bin_sums = vec![0.0; 2];

        bin_count_sum_operation(&mut bin_counts, &mut bin_sums, &bin_index, &signal.view());

        assert_eq!(bin_counts[0], 2);
        assert_eq!(bin_counts[1], 2);
        assert_relative_eq!(bin_sums[0], 4.0);
        assert_relative_eq!(bin_sums[1], 6.0);
    }

    #[test]
    fn test_bin_count_sum_all_one_bin() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let bin_index = vec![0, 0, 0];
        let mut bin_counts = vec![0; 3];
        let mut bin_sums = vec![0.0; 3];

        bin_count_sum_operation(&mut bin_counts, &mut bin_sums, &bin_index, &signal.view());

        assert_eq!(bin_counts[0], 3);
        assert_relative_eq!(bin_sums[0], 6.0);
        // Other bins empty
        assert_eq!(bin_counts[1], 0);
        assert_eq!(bin_counts[2], 0);
        assert_relative_eq!(bin_sums[1], 0.0);
        assert_relative_eq!(bin_sums[2], 0.0);
    }

    #[test]
    fn test_bin_count_sum_empty_bins() {
        // 5 bins but data only in bins 0 and 4
        let signal = Array1::from_vec(vec![10.0, 20.0]);
        let bin_index = vec![0, 4];
        let mut bin_counts = vec![0; 5];
        let mut bin_sums = vec![0.0; 5];

        bin_count_sum_operation(&mut bin_counts, &mut bin_sums, &bin_index, &signal.view());

        assert_eq!(bin_counts[0], 1);
        assert_eq!(bin_counts[1], 0);
        assert_eq!(bin_counts[2], 0);
        assert_eq!(bin_counts[3], 0);
        assert_eq!(bin_counts[4], 1);
        assert_relative_eq!(bin_sums[0], 10.0);
        assert_relative_eq!(bin_sums[4], 20.0);
    }

    // ===== squared_diff_calculation tests =====

    #[test]
    fn test_squared_diff_calculation() {
        let signal = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let bin_index = vec![0, 1, 0, 1];
        let bin_means = vec![3.0, 5.0];
        let mean = 4.0;

        let mut bin_squared_difference = vec![0.0; 2];
        let mut squared_difference = 0.0;

        squared_diff_calculation(
            &mut bin_squared_difference,
            &mut squared_difference,
            &bin_index,
            &bin_means,
            &signal.view(),
            &mean,
        );

        assert_relative_eq!(bin_squared_difference[0], 8.0);
        assert_relative_eq!(bin_squared_difference[1], 8.0);
        assert_relative_eq!(squared_difference, 20.0);
    }

    // ===== squared_diff_sigma_calculation tests =====

    #[test]
    fn test_squared_diff_sigma_calculation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 4.5, 5.5]);
        let sigma = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let bin_index = vec![0, 0, 1, 1];
        let bin_means = vec![3.0, 5.0];
        let mean = 4.0;

        let mut bin_squared_difference = vec![0.0; 2];
        let mut squared_difference = 0.0;

        squared_diff_sigma_calculation(
            &mut bin_squared_difference,
            &mut squared_difference,
            &bin_index,
            &bin_means,
            &signal.view(),
            &mean,
            &sigma.view(),
        );

        // Bin 0: |3-1|=2>=1 → 4, |3-2|=1>=1 → 1 → 5.0
        assert_relative_eq!(bin_squared_difference[0], 5.0);
        // Bin 1: |5-4.5|=0.5<1 → skip, |5-5.5|=0.5<1 → skip → 0.0
        assert_relative_eq!(bin_squared_difference[1], 0.0);
        // Total: 9 + 4 + 0 + 2.25 = 15.25
        assert_relative_eq!(squared_difference, 15.25);
    }

    #[test]
    fn test_sigma_large_all_filtered() {
        // Sigma so large that all differences are within sigma → both accumulators stay 0
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sigma = Array1::from_vec(vec![1000.0, 1000.0, 1000.0]);
        let bin_index = vec![0, 1, 0];
        let bin_means = vec![2.0, 2.0];
        let mean = 2.0;

        let mut bin_squared_difference = vec![0.0; 2];
        let mut squared_difference = 0.0;

        squared_diff_sigma_calculation(
            &mut bin_squared_difference,
            &mut squared_difference,
            &bin_index,
            &bin_means,
            &signal.view(),
            &mean,
            &sigma.view(),
        );

        assert_relative_eq!(bin_squared_difference[0], 0.0);
        assert_relative_eq!(bin_squared_difference[1], 0.0);
        assert_relative_eq!(squared_difference, 0.0);
    }

    #[test]
    fn test_sigma_tiny_none_filtered() {
        // Sigma near zero → all differences are counted (same as no-sigma path)
        let signal = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let sigma = Array1::from_vec(vec![1e-15, 1e-15, 1e-15, 1e-15]);
        let bin_index = vec![0, 1, 0, 1];
        let bin_means = vec![3.0, 5.0];
        let mean = 4.0;

        let mut bin_squared_difference = vec![0.0; 2];
        let mut squared_difference = 0.0;

        squared_diff_sigma_calculation(
            &mut bin_squared_difference,
            &mut squared_difference,
            &bin_index,
            &bin_means,
            &signal.view(),
            &mean,
            &sigma.view(),
        );

        // Same as no-sigma: bin0=8, bin1=8, total=20
        assert_relative_eq!(bin_squared_difference[0], 8.0);
        assert_relative_eq!(bin_squared_difference[1], 8.0);
        assert_relative_eq!(squared_difference, 20.0);
    }

    // ===== compute_theta tests =====

    #[test]
    fn test_compute_theta_known_frequency() {
        // Sine wave at frequency 1.0 Hz → period = 1.0 s
        // At the true frequency, theta should be close to 0
        let n = 1000;
        let freq = 1.0;
        let period = 1.0 / freq;
        let time_vec: Vec<f64> = (0..n).map(|i| i as f64 * period / n as f64 * 10.0).collect();
        let signal_vec: Vec<f64> = time_vec
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * freq * t).sin())
            .collect();
        let time = Array1::from_vec(time_vec);
        let signal = Array1::from_vec(signal_vec);

        let theta = compute_theta(time.view(), signal.view(), freq, 10).unwrap();
        assert!(theta < 0.3, "theta at true frequency should be small, got {}", theta);
    }

    #[test]
    fn test_compute_theta_wrong_frequency() {
        // Sine wave at frequency 1.0 Hz, test at a very different frequency
        let n = 1000;
        let true_freq = 1.0;
        let period = 1.0 / true_freq;
        let time_vec: Vec<f64> = (0..n).map(|i| i as f64 * period / n as f64 * 10.0).collect();
        let signal_vec: Vec<f64> = time_vec
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * true_freq * t).sin())
            .collect();
        let time = Array1::from_vec(time_vec);
        let signal = Array1::from_vec(signal_vec);

        let wrong_freq = 3.7; // Not a harmonic
        let theta = compute_theta(time.view(), signal.view(), wrong_freq, 10).unwrap();
        assert!(theta > 0.7, "theta at wrong frequency should be close to 1, got {}", theta);
    }

    #[test]
    fn test_compute_theta_freq_zero_errors() {
        let time = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = compute_theta(time.view(), signal.view(), 0.0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_theta_identical_signal_errors() {
        // All signal values identical → total squared difference = 0 → error
        let time = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let signal = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

        let result = compute_theta(time.view(), signal.view(), 1.0, 2);
        assert!(result.is_err());
    }

    // ===== compute_theta_sigma tests =====

    #[test]
    fn test_compute_theta_sigma_basic() {
        let n = 500;
        let freq = 2.0;
        let time_vec: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let signal_vec: Vec<f64> = time_vec
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * freq * t).sin())
            .collect();
        // Small sigma so most differences are counted
        let sigma_vec: Vec<f64> = vec![0.001; n];
        let time = Array1::from_vec(time_vec);
        let signal = Array1::from_vec(signal_vec);
        let sigma = Array1::from_vec(sigma_vec);

        let theta = compute_theta_sigma(time.view(), signal.view(), sigma.view(), freq, 10).unwrap();
        assert!(theta > 0.0, "theta should be positive, got {}", theta);
        assert!(theta <= 1.0, "theta should be <= 1, got {}", theta);
    }

    #[test]
    fn test_compute_theta_sigma_all_within_sigma_errors() {
        // Large sigma → all filtered → squared_difference = 0 → error
        let time = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let sigma = Array1::from_vec(vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);

        let result = compute_theta_sigma(time.view(), signal.view(), sigma.view(), 1.0, 2);
        assert!(result.is_err());
    }
}
