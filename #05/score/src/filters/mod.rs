use image::{DynamicImage, GrayImage, Pixel, Rgb};
use imageproc::hough;
use imageproc::{definitions::Image, hough::LineDetectionOptions};
use rayon::prelude::*;

// pub fn gradient(image: GrayImage){}

pub fn binarize(mut image: GrayImage, threshold: u8) -> GrayImage {
    image.as_mut().par_iter_mut().for_each(|pixel| {
        if *pixel > threshold {
            *pixel = 255
        } else {
            *pixel = 0
        }
    });
    image
}

pub fn detect_lines(image: &DynamicImage, color: Rgb<u8>) -> Image<Rgb<u8>> {
    let options = LineDetectionOptions {
        vote_threshold: 30,
        suppression_radius: 8,
    };

    let lines = hough::detect_lines(&image.to_luma8(), options);

    let mut image = image.to_rgb8();

    hough::draw_polar_lines_mut(&mut image, &lines, color);

    image
}
