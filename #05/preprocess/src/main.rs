mod draw;
mod filters;

use image::Luma;
use image::{buffer::ConvertBuffer, open, Rgb, RgbImage};
use imageproc::distance_transform::Norm;
use imageproc::hough::{draw_polar_lines_mut, PolarLine};
use imageproc::morphology::{close_mut, dilate_mut};

use crate::draw::remove_staff;

struct Profiler {
    start_time: std::time::Instant,
    span: std::time::Duration,
}

impl Profiler {
    fn start() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            span: std::time::Duration::from_micros(0),
        }
    }

    fn end(&mut self) {
        self.span = std::time::Instant::now() - self.start_time;
    }

    fn print(&self, msg: &str) {
        println!("{}: {}ms", msg, self.span.as_millis());
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Not enough arguments were given.");

    let mut profiler = Profiler::start();
    let img = open(path).expect("Cannot open image.");
    profiler.end();
    profiler.print("Image loading took");
    let img = imageproc::filter::gaussian_blur_f32(&img.to_rgb8(), 2.0);
    let input = img.convert();

    profiler = Profiler::start();
    let out = imageproc::contrast::adaptive_threshold(&input, 7);
    let out = imageproc::filter::median_filter(&out, 4, 4);
    profiler.end();
    profiler.print("Thresholding took");

    profiler = Profiler::start();
    let no_staff = remove_staff(&out);
    profiler.end();
    profiler.print("Removing staff took");

    profiler = Profiler::start();
    let mut staff = filters::find_staff_lines(out.clone(), 2);
    profiler.end();
    profiler.print("Line detection took");

    staff.sort_by(|x, y| x.r.partial_cmp(&y.r).unwrap_or(std::cmp::Ordering::Less));
    let first: Vec<PolarLine> = staff.iter().step_by(5).cloned().collect();
    let second: Vec<PolarLine> = staff.iter().skip(1).step_by(5).cloned().collect();
    let third: Vec<PolarLine> = staff.iter().skip(2).step_by(5).cloned().collect();
    let fourth: Vec<PolarLine> = staff.iter().skip(3).step_by(5).cloned().collect();
    let fifth: Vec<PolarLine> = staff.iter().skip(4).step_by(5).cloned().collect();

    let mut staff_img = RgbImage::from_raw(
        img.width(),
        img.height(),
        vec![0; img.width() as usize * img.height() as usize * 3],
    )
    .expect("Cannot create image buffer for staff lines.");

    // let mut horizontal = draw_polar_lines(&img.convert(), &first, Rgb::<u8>([255, 0, 0]));

    draw_polar_lines_mut(&mut staff_img, &first, Rgb::<u8>([255, 0, 0]));
    draw_polar_lines_mut(&mut staff_img, &second, Rgb::<u8>([255, 255, 0]));
    draw_polar_lines_mut(&mut staff_img, &third, Rgb::<u8>([0, 255, 0]));
    draw_polar_lines_mut(&mut staff_img, &fourth, Rgb::<u8>([0, 255, 255]));
    draw_polar_lines_mut(&mut staff_img, &fifth, Rgb::<u8>([0, 0, 255]));

    let mut overlay = out.clone();
    draw_polar_lines_mut(&mut overlay, &staff, Luma::<u8>([255]));
    // close_mut(&mut overlay, Norm::LInf, 2);
    dilate_mut(&mut overlay, Norm::LInf, 1);
    // draw_polar_lines_mut(&mut overlay, &first, Rgb::<u8>([255, 0, 0]));
    // draw_polar_lines_mut(&mut overlay, &second, Rgb::<u8>([255, 255, 0]));
    // draw_polar_lines_mut(&mut overlay, &third, Rgb::<u8>([0, 255, 0]));
    // draw_polar_lines_mut(&mut overlay, &fourth, Rgb::<u8>([0, 255, 255]));
    // draw_polar_lines_mut(&mut overlay, &fifth, Rgb::<u8>([0, 0, 255]));

    out.save("processed/output.png")
        .expect("Cannot save image.");
    staff_img
        .save("processed/staff.png")
        .expect("Cannot save image.");
    overlay
        .save("processed/overlay.png")
        .expect("Cannot save image.");
    no_staff
        .save("processed/no_staff.png")
        .expect("Cannot save image.");
}
