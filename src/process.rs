use pyo3::prelude::*;
use numpy::ndarray::ArrayView1;
use rayon::prelude::*;

use crate::time_section;

pub fn generate_freqs(min_freq: f64, max_freq: f64, n_freqs: u64) -> Vec<f64>{
    let step = (max_freq - min_freq) / (n_freqs as f64 - 1.0);
    let freq_vec: Vec<f64> = (0..n_freqs).into_par_iter().map(|i| min_freq + (i as f64) * step).collect();
    freq_vec
}

pub fn compute_theta_st(time: ArrayView1<f64>, signal: ArrayView1<f64>, freq: f64, n_bins: u64) -> PyResult<f64>{
    let inv_freq = 1.0/freq;
    let phase: Vec<f64>= time_section!("phase calculation", {
        time.iter().map(|&x| x % inv_freq).collect()
    });

    let s = n_bins as f64/inv_freq;
    let bin_index: Vec<u64> = time_section!("binning_operation", {
        phase.iter().map(|x| (x * s) as u64).collect()
    });
    
    let mut bin_counts = vec![0_u64; n_bins as usize];
    let mut bin_sums = vec![0_f64; n_bins as usize];
    time_section!("bin_count_sum_operation", {
    
    // Serial implementation for smaller arrays
    for (i, &bin) in bin_index.iter().enumerate() {
        bin_counts[bin as usize] += 1;
        bin_sums[bin as usize] += signal[i];
    }});

    // calculate the mean of each of the bins
    let bin_means: Vec<f64> = bin_sums.iter()
    .zip(bin_counts.iter())
    .map(|(&sum, &count)| {
        if count > 0 {
            sum / count as f64
        } else {
            0.0 // or f64::NAN
        }
    }).collect::<Vec<f64>>();
    
    // calculate the total mean
    let mean = bin_sums.iter().sum::<f64>()/(bin_counts.iter().sum::<u64>() as f64);

    let mut bin_squared_difference = vec![0_f64; n_bins as usize];
    let mut squared_difference: f64 = 0.0;

    time_section!("squared_diff_calculation", { 
    for (i, &bin) in bin_index.iter().enumerate() {
        bin_squared_difference[bin as usize] += f64::powi(bin_means[bin as usize] - signal[i],2);
        squared_difference += f64::powi(mean - signal[i], 2);
    }});
    
    Ok(bin_squared_difference.iter().sum::<f64>()/squared_difference)
    
}

