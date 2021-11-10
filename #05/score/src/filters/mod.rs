use image::GrayImage;

// pub fn gradient(image: GrayImage){}

pub fn binarize(mut image: GrayImage, threshold: u8) -> GrayImage {
    image.as_mut().into_iter().for_each(|pixel| if *pixel > threshold { *pixel = 255 } else { *pixel = 0 });
    image
}