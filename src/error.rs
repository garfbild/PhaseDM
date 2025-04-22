use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::{
    exceptions::PyTypeError, exceptions::PyValueError, types::PyType, PyAny, PyResult, Python,
};

pub fn check_matching_length(
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    sigma: &Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<()> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err(format!(
            "Array length mismatch: first array length {}, second array length {}",
            x.len(),
            y.len()
        )));
    };
    if let Some(sigma) = sigma {
        let std_view = sigma.as_array();
        if std_view.len() != x.len() {
            return Err(PyValueError::new_err(format!(
                "Array length mismatch: data array length {}, std array length {}",
                x.len(),
                std_view.len()
            )));
        }
    };
    Ok(())
}

pub fn check_min_less_max(min_freq: f64, max_freq: f64, n_freqs: u64) -> PyResult<()> {
    if min_freq > max_freq {
        return Err(PyValueError::new_err(format!(
            "frequency bound value mismatch: min_freq {}, max_freq {}",
            min_freq, max_freq
        )));
    } else if min_freq == max_freq && n_freqs != 1 {
        return Err(PyValueError::new_err(format!(
            "frequency value mismatch: if you wish to test a single frequency then min_freq = max_freq and n=1"
        )));
    } else if min_freq < 0_f64 || max_freq < 0_f64 {
        return Err(PyValueError::new_err(format!(
            "frequency value issue: cannot interpret a negative frequncy {} or {}",
            min_freq, max_freq
        )));
    } else {
        Ok(())
    }
}
pub fn check_sigma_array<'py>(
    py: Python<'py>,
    sigma: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let np = py.import("numpy")?;
    let ndarray_attr = np.getattr("ndarray")?;
    let ndarray_type = ndarray_attr.downcast::<PyType>()?;
    if sigma.is_exact_instance(ndarray_type) {
        let array_bound = sigma.downcast::<PyArray1<f64>>()?.readonly();
        return Ok(array_bound);
    }
    let astropy_quantity = py.import("astropy.units")?.getattr("Quantity")?;
    let float64_attr = np.getattr("float64")?;
    if sigma.is_instance(&astropy_quantity)? {
        let value = sigma.getattr("value")?;
        let float_array = np.call_method1("array", (value, float64_attr))?;
        let array_bound = float_array.downcast::<PyArray1<f64>>()?.readonly();
        return Ok(array_bound);
    }
    return Err(PyTypeError::new_err(
        "sigma must be either a numpy array or astropy Quantity",
    ));
}
pub fn check_signal_array<'py>(
    py: Python<'py>,
    signal: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let np = py.import("numpy")?;
    let ndarray_attr = np.getattr("ndarray")?;
    let ndarray_type = ndarray_attr.downcast::<PyType>()?;
    // be careful as Quanitity inherits from ndarry so will pass with is_instance
    if signal.is_exact_instance(&ndarray_type) {
        let array_bound = signal.downcast::<PyArray1<f64>>()?.readonly();
        return Ok(array_bound);
    }
    let astropy_quantity = py.import("astropy.units")?.getattr("Quantity")?;
    let float64_attr = np.getattr("float64")?;
    if signal.is_instance(&astropy_quantity)? {
        let value = signal.getattr("value")?;
        let float_array = np.call_method1("array", (value, float64_attr))?;
        let array_bound = float_array.downcast::<PyArray1<f64>>()?.readonly();
        return Ok(array_bound);
    }
    return Err(PyTypeError::new_err(
        "signal must be either a numpy array or astropy Quantity",
    ));
}

pub fn check_time_array<'py>(
    py: Python<'py>,
    time: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    // Import modules once at the beginning
    let np = py.import("numpy")?;
    let float64_attr = np.getattr("float64")?;

    // Check if time is a numpy array
    let ndarray_type = np.getattr("ndarray")?;
    if time.is_exact_instance(&ndarray_type) {
        let dtype = time.getattr("dtype")?;
        let kind = dtype.getattr("kind")?.extract::<String>()?;

        match kind.as_str() {
            "f" => {
                // Float array - check if it's float64
                let dtype_name = dtype.str()?.to_string();
                if dtype_name.contains("float64") {
                    // It's already float64
                    return Ok(time.downcast::<PyArray1<f64>>()?.readonly());
                } else {
                    // Convert to float64
                    let float_array = np.call_method1("array", (time, float64_attr))?;
                    return Ok(float_array.downcast::<PyArray1<f64>>()?.readonly());
                }
            }
            "M" => {
                // Datetime64 array - convert and normalize
                let float_array = np.call_method1("array", (time, float64_attr))?;
                return normalize_datetime_array(py, &float_array);
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "time must be a numpy array with dtype float64 or datetime64",
                ))
            }
        }
    }
    // Check if time is an astropy Time array
    let astropy_time = py.import("astropy.time")?.getattr("Time")?;
    if time.is_instance(&astropy_time)? {
        // Convert Time object to float64 array
        let datetime64 = time.getattr("datetime64")?;
        let float_array = datetime64.call_method1("astype", (float64_attr,))?;
        return normalize_datetime_array(py, &float_array);
    }

    // Invalid input type
    Err(PyTypeError::new_err(
        "time must be a numpy array or astropy Time object",
    ))
}

#[inline]
fn normalize_datetime_array<'py>(
    py: Python<'py>,
    float_array: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let array_bound = float_array.downcast::<PyArray1<f64>>()?.readonly();
    let array_slice = array_bound.as_slice()?;

    if array_slice.is_empty() {
        return Err(PyValueError::new_err("Empty time array"));
    }

    let min_time = *array_slice.get(0).unwrap();

    // Preallocate the output array for better performance
    let mut array_vec = Vec::with_capacity(array_slice.len());

    // Process in chunks for better cache locality
    const CHUNK_SIZE: usize = 1024;
    for chunk in array_slice.chunks(CHUNK_SIZE) {
        for &value in chunk {
            array_vec.push((value - min_time) / 1e9);
        }
    }

    Ok(array_vec.into_pyarray(py).readonly())
}
