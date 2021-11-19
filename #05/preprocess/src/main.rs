mod filters;

use image::{buffer::ConvertBuffer, open, Rgb};
use imageproc::hough::{draw_polar_lines_mut, PolarLine};
use rayon::prelude::*;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Not enough arguments were given.");

    let img = open(path).expect("Cannot open image.");
    let img = imageproc::filter::gaussian_blur_f32(&img.to_rgb8(), 2.0);
    let input = img.convert();

    // let blurred = imageproc::filter::gaussian_blur_f32(&input, 3.5);
    // blurred.save("processed/blur.png").expect("cannot save blur image");
    // let out = filters::binarize(&blurred, 127).expect("cannot create binarized image");
    let out = imageproc::contrast::adaptive_threshold(&input, 4);
    let out = imageproc::filter::median_filter(&out, 4, 4);

    use imageproc::hough::draw_polar_lines;
    let mut staff = filters::find_staff_lines(out, 5);
    staff.sort_by(|x, y| x.r.partial_cmp(&y.r).unwrap_or(std::cmp::Ordering::Less));
    let first: Vec<PolarLine> = staff.iter().step_by(5).cloned().collect();
    let second: Vec<PolarLine> = staff.iter().skip(1).step_by(5).cloned().collect();
    let third: Vec<PolarLine> = staff.iter().skip(2).step_by(5).cloned().collect();
    let fourth: Vec<PolarLine> = staff.iter().skip(3).step_by(5).cloned().collect();
    let fifth: Vec<PolarLine> = staff.iter().skip(4).step_by(5).cloned().collect();

    let mut horizontal = draw_polar_lines(&img.convert(), &first, Rgb::<u8>([255, 0, 0]));

    draw_polar_lines_mut(&mut horizontal, &second, Rgb::<u8>([255, 255, 0]));
    draw_polar_lines_mut(&mut horizontal, &third, Rgb::<u8>([0, 255, 0]));
    draw_polar_lines_mut(&mut horizontal, &fourth, Rgb::<u8>([0, 255, 255]));
    draw_polar_lines_mut(&mut horizontal, &fifth, Rgb::<u8>([0, 0, 255]));

    // out.save("processed/output.png")
    //     .expect("Cannot save image.");
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
