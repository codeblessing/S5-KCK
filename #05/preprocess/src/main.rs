mod filters;

use image::{open, Rgb};
use rayon::prelude::*;

use crate::filters::ExtractLines;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Not enough arguments were given.");

    let img = open(path).expect("Cannot open image.");
    let input = img.to_luma8();

    let out = filters::binarize(&input).expect("cannot create binarized image");

    use imageproc::hough::draw_polar_lines;
    let lines = filters::find_lines(&out, 1);
    let horizontal = draw_polar_lines(
        &img.to_rgb8(),
        &lines.get_horizontal(1),
        Rgb::<u8>([255, 0, 0]),
    );

    out.save("processed/output.png")
        .expect("Cannot save image.");
    horizontal
        .save("processed/horizontal.png")
        .expect("Cannot save image.");
}

fn normalize(matrix: &mut [i16]) {
    if let Some(&min) = matrix.iter().min() {
        if let Some(&max) = matrix.iter().max() {
            let span = (max - min) as f64;
            matrix
                .par_iter_mut()
                .for_each(|val| *val = (((*val - min) as f64 / span) * 255.0) as i16)
        }
    }
}
