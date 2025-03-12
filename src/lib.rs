use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1,PyReadonlyArray1,};
use rayon::prelude::*;

mod error;
mod process;
pub mod timing;

/// A Python module implemented in Rust.
#[pymodule]
fn phasedm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "pdm")]
    fn pdm<'py>(
        py: Python<'py>,
        time: &Bound<'py, PyAny>,
        signal: PyReadonlyArray1<'py, f64>,
        min_freq: f64, //assumed units are in seconds
        max_freq: f64,
        n_freqs: u64,
        n_bins: u64,
        verbose: u64
    ) -> PyResult<(Bound<'py, PyArray1<f64>>,Bound<'py, PyArray1<f64>>)> {
        if verbose == 0{
            timing::enable_timing(false);
        } else {
            timing::enable_timing(true);
        }
        // kind is the type of the time array, floating point array or a datetime array
        let time = error::check_time_array(py, time)?;

        // if the time is date time "M" then the units are nanoseconds
        // its easier to multiply the frequency bounds
        // let min_freq = if kind == "M" { min_freq * 1e9 } else { min_freq };
        // let max_freq = if kind == "M" { max_freq * 1e9 } else { max_freq };

        let time = time.as_array();
        let signal = signal.as_array();

        error::check_matching_length(time, signal)?;

        error::check_min_less_max(min_freq, max_freq, n_freqs)?;
        
        let freqs = process::generate_freqs(min_freq, max_freq, n_freqs);

        let thetas: Vec<f64> = freqs.par_iter()
        .map(|freq| process::compute_theta_st(time, signal, *freq, n_bins))
        .collect::<Result<Vec<f64>, _>>()?;
        
        if verbose != 0{
            println!("{}", timing::get_timing_report());
        }

        Ok((freqs.into_pyarray(py),thetas.into_pyarray(py)))
    }
    Ok(())
}
