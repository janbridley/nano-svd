use std::ops::Div;

use num_traits::{Float, Signed};

/**Computes the singular value decomposition of a 2-by-2 triangular matrix.


This method implements the same function signature as LAPACK. The [`SVDTri2`] function
provides a more intuitive interface for the same decomposition.
*/
pub fn lasv2<T: Clone + PartialOrd + Float>(
    f: &T,
    g: &T,
    h: &T,
    ssmin: &mut T,
    ssmax: &mut T,
    snr: &mut T,
    csr: &mut T,
    snl: &mut T,
    csl: &mut T,
) {
    let two = T::one() + T::one();
    let four = two + two;
    let mut ft = *f;
    let mut ht = *h;
    let mut fa = ft.abs();
    let mut ha = h.abs();

    // pmax indexes the largest element in the matrix, in order [f, g, [0], h]
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        // let (ft, ht) = (ht, ft);
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    // Now fa >= ha

    let gt = *g;
    let ga = gt.abs();

    let (mut clt, mut srt, mut slt, mut crt) = (T::one(), T::one(), T::one(), T::one());

    if ga == T::zero() {
        // Diagonal matrix
        *ssmin = ha;
        *ssmax = fa;
        (clt, crt) = (T::one(), T::one());
        (slt, srt) = (T::zero(), T::zero());
    } else {
        let mut ga_small = true;
        if ga > fa {
            pmax = 2;
            if (fa / ga) < T::epsilon() {
                // Absolute value of g is very large.
                ga_small = false;
                *ssmax = ga;
                match ha > T::one() {
                    true => *ssmin = fa / (ga / ha),
                    false => *ssmin = (fa / ga) * ha,
                };
                (clt, srt) = (T::one(), T::one());
                slt = ht / gt;
                crt = ft / gt;
            }
        }
        if ga_small {
            // Normal case
            let d = fa - ha;
            let mut l = match d == fa {
                true => T::one(), // Handle infinite F or H
                false => d / fa,
            }; // Note that 0.0 <= l <= 1.0
            let m = gt / ft; // Note that abs(m) <= 1/ε
            let mm = m * m;

            let mut t = two - l; // T >= 1.0

            let s = (mm + t * t).sqrt(); // 1 <= s <= 1 + 1/ε
            let r = match l == T::zero() {
                true => m.abs(),
                false => (mm + l * l).sqrt(),
            }; // 0 <= r <= 1 + 1/ε

            let a = (s + r) / two; // 1 <= a <= 1 + abs(m);

            *ssmin = ha / a;
            *ssmax = fa * a;

            t = match (mm == T::zero(), l == T::zero()) {
                // TODO: is this correct?
                (true, true) => two.copysign(ft) * gt.signum(),
                (true, false) => gt / d.copysign(ft) + m / t,
                (false, _) => (m / (s + t) + m / (r + l)) * (T::one() + a),
            };
            l = (t * t + four).sqrt();
            crt = two / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    if swap {
        (*csl, *csr) = (srt, slt);
        (*snl, *snr) = (crt, clt);
    } else {
        (*csl, *csr) = (clt, crt);
        (*snl, *snr) = (slt, srt);
    }
    let tsign = match pmax {
        1 => csr.signum() * csl.signum() * f.signum(),
        2 => snr.signum() * csl.signum() * g.signum(),
        3 => snr.signum() * snl.signum() * h.signum(),
        _ => unreachable!(),
    };
    *ssmax = ssmax.copysign(tsign);
    *ssmin = ssmin.copysign(tsign * f.signum() * h.signum());
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};
    use rstest::rstest;

    const SMALL_F64: f64 = 1e-15;
    const LARGE_F64: f64 = 1e34;

    #[rstest]
    #[case::eye((1.0, 0.0, 1.0))]
    #[case::eye_neg((-1.0, 0.0, -1.0))]
    #[case::diagonal((3.0, 0.0, 4.0))]
    #[case::standard((-2.0, 5.0, -3.0))]
    #[case::zero((0.0, 0.0, 0.0))]
    #[case::degenerate((LARGE_F64, 1e8, SMALL_F64))]
    #[case::nilpotent_g((0.0, 5.0, 0.0))]
    #[case::nilpotent_f((5.0, 0.0, 0.0))]
    #[case::nilpotent_h((0.0, 0.0, 5.0))]
    #[case::all_tiny_positive((SMALL_F64, SMALL_F64, SMALL_F64))]
    #[case::all_tiny_negative((-SMALL_F64, -SMALL_F64, -SMALL_F64))]
    #[case::large_mixed_signs((LARGE_F64, -LARGE_F64, LARGE_F64))]
    #[case::large_opposite_signs((-LARGE_F64, LARGE_F64, -LARGE_F64))]
    fn test_lasv2_param(#[case] (f, g, h): (f64, f64, f64)) {
        let (mut ssmin, mut ssmax, mut snr, mut csr, mut snl, mut csl) =
            (0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0);

        lasv2(
            &f, &g, &h, &mut ssmin, &mut ssmax, &mut snr, &mut csr, &mut snl, &mut csl,
        );
        assert!(ssmax.abs() >= ssmin);
        assert!((ssmin <= 0.0 && ssmax <= 0.0) || ssmin >= 0.0);
        assert_ulps_eq!(snr * snr + csr * csr, 1.0, max_ulps = 4);
        assert_ulps_eq!(snl * snl + csl * csl, 1.0, max_ulps = 4);

        // Validate rotation matrices approximately
        let matrix = faer::mat![[f, g], [0.0, h]];

        // Compare against faer SVD
        let svd = matrix.svd().unwrap();
        let (faeru, faers, faerv) = (svd.U().to_owned(), svd.S(), svd.V().to_owned());
        assert_ulps_eq!(ssmin.abs(), faers[1], max_ulps = 10);
        assert_ulps_eq!(ssmax.abs(), faers[0], max_ulps = 10);

        let u = faer::mat![[csl, snl], [-snl, csl]];
        let v = faer::mat![[csr, snr], [-snr, csr]];

        // println!("{u:?}\n{faeru:?}");
        // assert_ulps_eq!(u[(0, 0)], faeru[(0, 0)], max_ulps = 10);

        let reconstructed = u * faer::mat![[ssmax, 0.0], [0.0, ssmin]] * v.transpose();
        assert_ulps_eq!(reconstructed[(0, 0)], f, max_ulps = 10);
        assert_ulps_eq!(reconstructed[(0, 1)], g, max_ulps = 10);
        assert_ulps_eq!(
            reconstructed[(1, 0)].abs(),
            0.0,
            epsilon = f64::EPSILON * 4.0
        );
        assert_ulps_eq!(reconstructed[(1, 1)], h, max_ulps = 10);
    }
}
