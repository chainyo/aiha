use pyo3::prelude::*;

mod config;
mod error;
mod scan;
mod utils;


#[pymodule]
fn aiha(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<config::ConfigUrl>()?;
    m.add_class::<config::ModelConfig>()?;
    m.add_class::<scan::Scan>()?;
    Ok(())
}

