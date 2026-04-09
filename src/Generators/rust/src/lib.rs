mod stage3;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyReadonlyArray2, PyReadwriteArray2, PyArray2};

#[pyfunction]
fn populate<'py>(
    py: Python<'py>,
    tilemap: PyReadwriteArray2<'py, u8>,
    theme_map: PyReadonlyArray2<'py, u8>,
    rng_seed: u64,
    room_size: u8,
    //themes: _,
    //tiles: _,
) {

}

#[pymodule]
fn dungeon_rust(_py: Python, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(populate, module)?)?;
    Ok(())
}