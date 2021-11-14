pub mod error;
pub mod filters;

use std::path::Path;

use image::io::Reader;
use image::{GrayImage, Rgb};
// use ndarray::prelude::*;
use pyo3::{exceptions, prelude::*};

use crate::error::ScoreError;

// pub fn process(filepath: &str) {
//     // let filepath = Box::new(Path::new(filepath));
//     let img = read_image(filepath).unwrap();
// }

#[pymodule]
fn score(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binarize, m)?)?;
    m.add_function(wrap_pyfunction!(detect_lines, m)?)?;
    Ok(())
}

fn read_image(filename: impl AsRef<Path>) -> Result<GrayImage, ScoreError> {
    let image = Reader::open(filename)?
        .decode()
        .map_err(|_| ScoreError::ImageDecodeError)?;
    Ok(image.into_luma8())
}

#[pyfunction]
fn binarize(filepath: &str) -> PyResult<()> {
    let img =
        read_image(filepath).map_err(|_| exceptions::PyIOError::new_err("Cannot open image."))?;
    let filtered = filters::binarize(img, 127);
    filtered.save("processed/binary_00.png").map_err(|_| exceptions::PyIOError::new_err("Cannot save image."))?;
    Ok(())
}

#[pyfunction]
fn detect_lines(filepath: &str) -> PyResult<()> {
    let img = image::open(filepath).map_err(|_| exceptions::PyIOError::new_err("Cannot open image."))?;
    let modified = filters::detect_lines(&img, Rgb::<u8>([255, 0, 0]));
    modified.save("processed/lines_00.png").map_err(|_| exceptions::PyIOError::new_err("Cannot save image."))
}
