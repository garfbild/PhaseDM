use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::ndarray::ArrayView1;
use rayon::prelude::*;

use crate::time_section;

pub fn generate_freqs(min_freq: f64, max_freq: f64, n_freqs: u64) -> Vec<f64>{
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

fn compute_phase(time:ArrayView1<f64>, inv_freq: f64) -> Vec<f64>{
    // try and use par_iter for phase calculation otherwise use serial iter
    let mut result = Vec::with_capacity(time.len());
    //rem euclid is more precise but also slow
    if let Some(time_slice) = time.as_slice() {
        time_slice.par_iter()
            .map(|&x| x % inv_freq)
            .collect_into_vec(&mut result);
    } else {
        result.extend(time.iter().map(|&x| x % inv_freq));
    }
    
    result
}

fn binning_operation(phase:&Vec<f64>,inv_freq:f64,n_bins: u64) -> Vec<u64>{
    let s = n_bins as f64/inv_freq;
    phase.iter().map(|&x| (x * s) as u64).collect()
}

fn bin_count_sum_operation(bin_counts: &mut Vec<u64>, bin_sums: &mut Vec<f64>, bin_index: &Vec<u64>, signal: &ArrayView1<f64>) -> (){
    for (i, &bin) in bin_index.iter().enumerate() {
        bin_counts[bin as usize] += 1;
        bin_sums[bin as usize] += signal[i];
    }
}

fn squared_diff_calculation(bin_squared_difference: &mut Vec<f64>, squared_difference: &mut f64, bin_index: &Vec<u64>, bin_means: &Vec<f64>, signal: &ArrayView1<f64>, mean: &f64){
    for (i, &bin) in bin_index.iter().enumerate() {
        bin_squared_difference[bin as usize] += f64::powi(bin_means[bin as usize] - signal[i],2);
        *squared_difference += f64::powi(mean - signal[i], 2);
    }
}

pub fn compute_theta(time: ArrayView1<f64>, signal: ArrayView1<f64>, freq: f64, n_bins: u64) -> PyResult<f64>{
    let inv_freq = if freq != 0.0{
        1.0/freq
    } else {
        return Err(PyValueError::new_err(format!(
            "cannot evalutate frequency = 0. undefined behaviour."
        )));
    };

    let phase: Vec<f64>= time_section!("compute_phase", {
        compute_phase(time, inv_freq)
    });

    let bin_index: Vec<u64> = time_section!("binning_operation", {
        binning_operation(&phase, inv_freq, n_bins)
    });
    
    let mut bin_counts = vec![0_u64; n_bins as usize];
    let mut bin_sums = vec![0_f64; n_bins as usize];
    time_section!("bin_count_sum_operation", {
        bin_count_sum_operation(&mut bin_counts, &mut bin_sums, &bin_index, &signal);
    });

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
        squared_diff_calculation(&mut bin_squared_difference, &mut squared_difference, &bin_index, &bin_means, &signal, &mean);
    });
    
    Ok(bin_squared_difference.iter().sum::<f64>()/squared_difference)
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use approx::assert_relative_eq;

    #[test]
    fn test_generate_freqs() {
        // Test with 5 frequencies from 10 to 20
        let freqs = generate_freqs(10.0, 20.0, 5);
        assert_eq!(freqs.len(), 5);
        assert_relative_eq!(freqs[0], 10.0);
        assert_relative_eq!(freqs[4], 20.0);
        assert_relative_eq!(freqs[2], 15.0);
        
        // Test edge case: single frequency
        let single_freq = generate_freqs(10.0, 20.0, 1);
        assert_eq!(single_freq.len(), 1);
        assert_relative_eq!(single_freq[0], 10.0);
    }
    #[test]
    fn test_compute_phase() {
        // Simple test with known values
        let time = Array1::from_vec(vec![0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]);
        let time_view: ArrayView1<f64> = time.view();
        let inv_freq = 3.0;
        let phases = compute_phase(time_view, inv_freq);
        
        assert_eq!(phases.len(), 5);
        assert_relative_eq!(phases[0], 0.0);
        assert_relative_eq!(phases[1], 1.0);
        assert_relative_eq!(phases[2], 2.0);
        assert_relative_eq!(phases[3], 0.0);  // 3.0 % 3.0 = 0.0
        assert_relative_eq!(phases[4], 1.0);  // 4.0 % 3.0 = 1.0
    }

    #[test]
    fn test_binning_operation() {
        let phase: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let inv_freq: f64 = 3.0;
        let n_bins: u64 = 6;
        
        let bins = binning_operation(&phase, inv_freq, n_bins);
        
        assert_eq!(bins.len(), 6);
        assert_eq!(bins[0], 0);  // 0.0 * (6/3) = 0
        assert_eq!(bins[1], 1);  // 0.5 * (6/3) = 1
        assert_eq!(bins[2], 2);  // 1.0 * (6/3) = 2
        assert_eq!(bins[3], 3);  // 1.5 * (6/3) = 3
        assert_eq!(bins[4], 4);  // 2.0 * (6/3) = 4
        assert_eq!(bins[5], 5);  // 2.5 * (6/3) = 5
    }

    #[test]
    fn test_bin_count_sum_operation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let bin_index = vec![0, 1, 0, 1];
        let mut bin_counts = vec![0; 2];
        let mut bin_sums = vec![0.0; 2];
        
        bin_count_sum_operation(&mut bin_counts, &mut bin_sums, &bin_index, &signal.view());
        
        assert_eq!(bin_counts[0], 2);  // Two values in bin 0
        assert_eq!(bin_counts[1], 2);  // Two values in bin 1
        assert_relative_eq!(bin_sums[0], 4.0);  // 1.0 + 3.0
        assert_relative_eq!(bin_sums[1], 6.0);  // 2.0 + 4.0
    }


    #[test]
    fn test_squared_diff_calculation() {
        let signal = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let bin_index = vec![0, 1, 0, 1];
        let bin_means = vec![3.0, 5.0];  // Mean for bin 0 = 3.0, bin 1 = 5.0
        let mean = 4.0;  // Overall mean
        
        let mut bin_squared_difference = vec![0.0; 2];
        let mut squared_difference = 0.0;
        
        squared_diff_calculation(
            &mut bin_squared_difference, 
            &mut squared_difference, 
            &bin_index, 
            &bin_means, 
            &signal.view(), 
            &mean
        );
        
        // Bin 0: (3-1)² + (3-5)² = 4 + 4 = 8
        assert_relative_eq!(bin_squared_difference[0], 8.0);
        
        // Bin 1: (5-3)² + (5-7)² = 4 + 4 = 8
        assert_relative_eq!(bin_squared_difference[1], 8.0);
        
        // Total: (4-1)² + (4-3)² + (4-5)² + (4-7)² = 9 + 1 + 1 + 9 = 20
        assert_relative_eq!(squared_difference, 20.0);
    }
}