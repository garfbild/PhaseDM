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

pub fn check_signal_array<'py>(
    py: Python<'py>,
    signal: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let np = py.import("numpy")?;
    let float64_attr = np.getattr("float64")?;
    let ndarray_attr = np.getattr("ndarray")?;
    let ndarray_type = ndarray_attr.downcast::<PyType>()?;
    let astropy_quantity = py.import("astropy.units")?.getattr("Quantity")?;

    if signal.is_instance(&astropy_quantity)? {
        let value = signal.getattr("value")?;
        let float_array = np.call_method1("array", (value, float64_attr))?;

        let array_bound = float_array.downcast::<PyArray1<f64>>()?.readonly();
        return Ok(array_bound);
    } else if signal.is_instance(ndarray_type)? {
        let array_bound = signal.downcast::<PyArray1<f64>>()?.readonly();
        return Ok(array_bound);
    } else {
        return Err(PyTypeError::new_err(
            "signal must be either a numpy array or astropy Quantity",
        ));
    }
}

pub fn check_time_array<'py>(
    py: Python<'py>,
    time: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    // Check if time is an astropy Time array
    let np = py.import("numpy")?;
    let ndarray_attr = np.getattr("ndarray")?;
    let ndarray_type = ndarray_attr.downcast::<PyType>()?;
    let astropy_time = py.import("astropy.time")?.getattr("Time")?;

    if time.is_instance(&astropy_time)? {
        let float64_attr = np.getattr("float64")?;

        //We can convert the Time object to a datetime representation which is the most consistent
        let datetime64 = time.getattr("datetime64")?;

        let float_array = datetime64.call_method1("astype", (float64_attr,))?;
        let array_bound = float_array.downcast::<PyArray1<f64>>()?.readonly();

        //This is actually super important! small overhead from converting but speeds up the phase calculation by a lot
        let min_time = array_bound.get(0).unwrap();
        let array_vec: Vec<f64> = {
            let array_slice = array_bound.as_slice()?;
            array_slice.iter().map(|&x| (x - min_time) / 1e9).collect()
        };

        return Ok(array_vec.into_pyarray(py).readonly());
    } else if time.is_instance(ndarray_type)? {
        let dtype = time.getattr("dtype")?;
        let kind = dtype.getattr("kind")?.extract::<String>()?;

        // Check dtype - 'f' for float, 'M' for datetime64
        if kind == "f" {
            // It's a float array (any kind of float)
            // Check if it's specifically float64
            let dtype_name = dtype.str()?.to_string();
            if !dtype_name.contains("float64") {
                // Convert to float64 if it's not already
                let float64_attr = np.getattr("float64")?;
                let float_array = np.call_method1("array", (time, float64_attr))?;
                let array_bound = float_array.downcast::<PyArray1<f64>>()?;
                return Ok(array_bound.readonly());
            }
            // It's already float64
            let array_bound = time.downcast::<PyArray1<f64>>()?;
            return Ok(array_bound.readonly());
        } else if kind == "M" {
            // It's a datetime64 array, convert to float64
            // This part depends on how you want to interpret datetime values
            let float64_attr = np.getattr("float64")?;
            let float_array = np.call_method1("array", (time, float64_attr))?;
            let array_bound = float_array.downcast::<PyArray1<f64>>()?.readonly();

            //This is actually super important! small overhead from converting but speeds up the phase calculation by a lot
            let min_time = array_bound.get(0).unwrap();
            let array_vec: Vec<f64> = {
                let array_slice = array_bound.as_slice()?;
                array_slice.iter().map(|&x| (x - min_time) / 1e9).collect()
            };

            return Ok(array_vec.into_pyarray(py).readonly());
        } else {
            return Err(PyTypeError::new_err(
                "time must be a numpy array with dtype float64 or datetime64",
            ));
        }
    } else {
        return Err(PyTypeError::new_err(
            "time must be a numpy array or astropy Time object",
        ));
    }
}
