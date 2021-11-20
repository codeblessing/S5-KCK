use image::GrayImage;
use imageproc::hough::{LineDetectionOptions, PolarLine};
use rayon::prelude::*;

pub fn find_staff_lines(mut img: GrayImage, tolerance: u32) -> Vec<PolarLine> {
    //! Return vector of horizontal lines with given `tolerance`
    //! where `tolerance` is maximum allowed deviation (in angles) from horizontal line.
    //!
    //! NOTE: `tolerance` is required to be from range [0, 90]. If given `tolerance` is from outside of this range
    //! it is clipped appropriately.
    //!
    //! `img` is expected to be projection corrected and binarized, with notes & staff being black.
    //!
    //! Additional informations:
    //!     - points are detected as lines if the formed line occupies at least half of image width.
    //!     - currently there's no maxima suppression used.
    use image::imageops;
    use imageproc::hough;

    let tolerance = tolerance.clamp(0, 90);

    imageops::invert(&mut img);
    let options = LineDetectionOptions {
        vote_threshold: img.width() / 2,
        suppression_radius: 0,
    };

    //****************************************************************************************************************//
    // Lines that differ only by angle are combined into one, by taking shared distance (r) and average tangent (degree).
    let mut lines: Vec<PolarLine> = hough::detect_lines(&img, options)
        .drain(..)
        .filter(|line| {
            line.angle_in_degrees > 90 - tolerance && line.angle_in_degrees < 90 + tolerance
        })
        .collect();

    // We need to order lines by distance
    lines.par_sort_by(|first, second| {
        first
            .r
            .partial_cmp(&second.r)
            .unwrap_or(std::cmp::Ordering::Less)
    });

    let mut output: Vec<PolarLine> = Vec::with_capacity(15);

    loop {
        if let Some(current) = lines.first().cloned() {
            // Because `Vec::drain_filter` is unstable we need to use workaround,
            // by manually removing elements from Vec.
            let drain_max: usize = lines
                .iter()
                .take_while(|line| (line.r - current.r).abs() < 10.0)
                .map(|_| 1)
                .sum();

            let mut chunk: Vec<PolarLine> = lines.drain(..drain_max).collect();
            let count = chunk.len() as u32;
            let (theta, r): (Vec<_>, Vec<_>) = chunk
                .drain(..)
                .map(|line| (line.angle_in_degrees, line.r))
                .unzip();

            let theta = theta.iter().sum::<u32>() / count;
            let r = r.iter().sum::<f32>() / count as f32;

            output.push(PolarLine {
                r,
                angle_in_degrees: theta,
            });
        } else {
            break;
        }
    }

    //****************************************************************************************************************//

    output
}
