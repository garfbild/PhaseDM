use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF};

pub mod error;
pub mod process;
pub mod timing;

/// A Python module implemented in Rust.
#[pymodule]
fn phasedm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "pdm")]
    #[pyo3(signature = (time, signal, min_freq, max_freq, n_freqs, sigma=None, n_bins=None, verbose=None))]
    fn pdm<'py>(
        py: Python<'py>,
        time: &Bound<'py, PyAny>,
        signal: PyReadonlyArray1<'py, f64>,
        min_freq: f64, //assumed units are in seconds
        max_freq: f64,
        n_freqs: u64,
        sigma: Option<PyReadonlyArray1<'py, f64>>,
        n_bins: Option<u64>,
        verbose: Option<u64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let n_bins = n_bins.unwrap_or(10);
        let verbose = verbose.unwrap_or(0);

        if verbose == 0 {
            timing::enable_timing(false);
        } else {
            timing::enable_timing(true);
        }
        let time = error::check_time_array(py, time)?;

        let time = time.as_array();
        let signal = signal.as_array();

        error::check_matching_length(time, signal, &sigma)?;

        error::check_min_less_max(min_freq, max_freq, n_freqs)?;

        let freqs = time_section!(
            "generate_freqs",
            process::generate_freqs(min_freq, max_freq, n_freqs)
        );

        let thetas: Vec<f64> = if let Some(sigma) = sigma {
            let sigma_view = sigma.as_array();
            freqs
                .par_iter()
                .map(|freq| process::compute_theta_sigma(time, signal, sigma_view, *freq, n_bins))
                .collect::<Result<Vec<f64>, _>>()?
        } else {
            freqs
                .par_iter()
                .map(|freq| process::compute_theta(time, signal, *freq, n_bins))
                .collect::<Result<Vec<f64>, _>>()?
        };

        if verbose != 0 {
            println!("{}", timing::get_timing_report());
        }

        Ok((freqs.into_pyarray(py), thetas.into_pyarray(py)))
    }

    #[pyfn(m)]
    #[pyo3(name = "beta_test")]
    #[pyo3(signature = (n,n_bins,p))]
    fn beta_test(n: u64, n_bins: u64, p: f64) -> PyResult<f64> {
        if p < 0.0 || p > 1.0 {
            return Err(PyValueError::new_err(format!(
                "Cannot find significance value for: {}",
                p
            )));
        }
        if p == 0.0 {
            return Ok(0.0);
        } else if p == 1.0 {
            return Ok(1.0);
        } else {
            let d0 = { n - 1 } as f64;
            let d1 = { n_bins - 1 } as f64;
            let d2 = { n - n_bins } as f64;
            let n = Beta::new(d2 / 2.0, d1 / 2.0).map_err(|e| {
                PyValueError::new_err(format!("Failed to create Beta distribution: {}", e))
            })?;

            let result = n.inverse_cdf(p * d2 / d0);

            Ok(result)
        }
    }

    Ok(())
}
