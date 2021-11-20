use image::{GrayImage, Luma};
use imageproc::integral_image::row_running_sum;
use rayon::prelude::*;

pub fn remove_staff(image: &GrayImage) -> GrayImage {
    // threshold is value that needs to be not exceeded in order to be recognized as a line.
    // this value means that max half of line can be white.
    let threshold = image.width() * 128;
    let rows: Vec<u32> = (0..image.height())
    .into_par_iter()
    .filter(|&row| {
            let mut row_buffer = vec![0; image.width() as usize];
            row_running_sum(&image, row, &mut row_buffer, 0);
            row_buffer.last().unwrap() > &threshold
        })
        .collect();

    let buffer: Vec<u8> = image
        .enumerate_rows()
        .filter(|(index, _)| rows.contains(index))
        .map(|(_, row)| row)
        .flatten()
        .map(|(_, _, Luma(val))| val[0] )
        .collect();

    let width = image.width();
    let height = buffer.len() as u32 / width;

    GrayImage::from_raw(width, height, buffer).unwrap()
}
