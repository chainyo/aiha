pub mod scan;

use pyo3::prelude::*;

#[pymodule]
fn aiha(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<scan::Scan>()?;
    Ok(())
}

