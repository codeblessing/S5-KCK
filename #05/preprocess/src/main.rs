mod filters;

use image::{open, GrayImage};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Not enough arguments were given.");
    // let operation = std::env::args()
    //     .nth(2)
    //     .expect("Not enough arguments were given.");

    let start = std::time::Instant::now();
    let img = open(path).expect("Cannot open image.").into_luma8();
    let end = std::time::Instant::now() - start;
    println!("Time to load image: {}ms", end.as_millis());

    // if operation == "-b" {
    let start = std::time::Instant::now();
    let out = filters::binarize(&img).expect("cannot create binarized image");
    let end = std::time::Instant::now() - start;
    println!("Time to process (binarize) image: {}ms", end.as_millis());

    let start = std::time::Instant::now();
    let mut grad = filters::gradient_vertical(&out);
    let end = std::time::Instant::now() - start;
    println!("Time to process (gradient) image: {}ms", end.as_millis());
    normalize(&mut grad);
    let grad = grad.par_iter().map(|&val| val as u8).collect::<Vec<u8>>();

    let grad = GrayImage::from_vec(out.width(), out.height(), grad).expect("cannot convert image");

    let start = std::time::Instant::now();
    out.save("processed/output.png")
        .expect("Cannot save image.");
    grad.save("processed/gradient.png")
        .expect("Cannot save image.");
    let end = std::time::Instant::now() - start;
    println!("Time to save image: {}ms", end.as_millis());
    // }
}

fn normalize(matrix: &mut [i16]) {
    if let Some(&min) = matrix.iter().min() {
        matrix.par_iter_mut().for_each(|val| {
            *val = (*val - min).clamp(0, 255);
        })
    }
}
