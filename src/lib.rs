pub mod scan;
pub mod config;

use pyo3::prelude::*;

#[pymodule]
fn aiha(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<scan::Scan>()?;
    m.add_class::<config::ModelConfig>()?;
    Ok(())
}

