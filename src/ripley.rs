//! Ripley's K-function and L-function for spatial point pattern analysis.

use crate::poisson::Point2D;
use crate::poisson::Window;

/// Ripley's K-function estimator.
#[derive(Debug, Clone)]
pub struct RipleysK {
    pub points: Vec<Point2D>,
    pub window: Window,
}

impl RipleysK {
    pub fn new(points: Vec<Point2D>, window: Window) -> Self {
        Self { points, window }
    }

    /// Compute K(r) using the border-corrected estimator.
    /// K(r) = (1/λ) * (1/n) * Σᵢ Σⱼ≠ᵢ I(d(i,j) ≤ r) / w(i,j)
    /// where w(i,j) is the edge correction weight.
    pub fn compute(&self, r: f64) -> f64 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }

        let lambda = n as f64 / self.window.area();
        let mut sum = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = self.points[i].distance_to(&self.points[j]);
                if d <= r {
                    // Simple edge correction: fraction of circumference within window
                    let weight = self.edge_correction(&self.points[i], r);
                    sum += 1.0 / weight;
                }
            }
        }

        sum / (lambda * n as f64)
    }

    /// Compute K(r) for multiple radii.
    pub fn compute_curve(&self, radii: &[f64]) -> Vec<(f64, f64)> {
        radii.iter().map(|&r| (r, self.compute(r))).collect()
    }

    /// Edge correction weight: fraction of circumference at distance r from point p
    /// that falls within the window.
    fn edge_correction(&self, p: &Point2D, r: f64) -> f64 {
        match &self.window {
            Window::Rect { xmin, xmax, ymin, ymax } => {
                // For rectangular window, approximate edge correction
                let mut weight = 1.0;
                if p.x - r < *xmin { weight *= 1.0 - (*xmin - (p.x - r)) / (2.0 * r).max(1e-10); }
                if p.x + r > *xmax { weight *= 1.0 - ((p.x + r) - *xmax) / (2.0 * r).max(1e-10); }
                if p.y - r < *ymin { weight *= 1.0 - (*ymin - (p.y - r)) / (2.0 * r).max(1e-10); }
                if p.y + r > *ymax { weight *= 1.0 - ((p.y + r) - *ymax) / (2.0 * r).max(1e-10); }
                weight.max(0.01)
            }
            Window::Disk { cx, cy, radius } => {
                let dist_to_center = ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt();
                if dist_to_center + r <= *radius {
                    1.0
                } else {
                    // Approximate
                    let max_dist = dist_to_center + r;
                    let overlap = (2.0 * radius - max_dist + r).min(2.0 * r) / (2.0 * r);
                    overlap.max(0.01)
                }
            }
        }
    }

    /// Theoretical K(r) for CSR (Complete Spatial Randomness) in 2D: K(r) = π r².
    pub fn csr_k(r: f64) -> f64 {
        std::f64::consts::PI * r * r
    }
}

/// Ripley's L-function: L(r) = √(K(r) / π) - r.
/// Under CSR, L(r) = 0. Clustering gives L(r) > 0, regularity gives L(r) < 0.
#[derive(Debug, Clone)]
pub struct RipleysL {
    pub k: RipleysK,
}

impl RipleysL {
    pub fn new(points: Vec<Point2D>, window: Window) -> Self {
        Self { k: RipleysK::new(points, window) }
    }

    /// Compute L(r).
    pub fn compute(&self, r: f64) -> f64 {
        let k_r = self.k.compute(r);
        if k_r < 0.0 {
            return -r;
        }
        (k_r / std::f64::consts::PI).sqrt() - r
    }

    /// Compute L(r) for multiple radii.
    pub fn compute_curve(&self, radii: &[f64]) -> Vec<(f64, f64)> {
        radii.iter().map(|&r| (r, self.compute(r))).collect()
    }

    /// Under CSR, L(r) = 0 for all r.
    pub fn csr_l(_r: f64) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poisson::{HomogeneousPoisson, Window};
    use rand::{thread_rng, Rng};

    fn rect_window() -> Window {
        Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 }
    }

    #[test]
    fn test_k_function_csr_near_theoretical() {
        // For CSR, K(r) ≈ π r²
        let mut rng = thread_rng();
        let ppp = HomogeneousPoisson::new(2.0, rect_window());
        let points = ppp.sample(&mut rng);
        let k = RipleysK::new(points, rect_window());

        let r = 1.0;
        let k_val = k.compute(r);
        let theoretical = std::f64::consts::PI * r * r;
        // Allow generous tolerance due to sampling
        assert!(
            (k_val - theoretical).abs() < 5.0,
            "K({r}) = {k_val}, expected ~{theoretical}"
        );
    }

    #[test]
    fn test_k_function_zero_points() {
        let k = RipleysK::new(vec![], rect_window());
        assert!((k.compute(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_k_function_one_point() {
        let k = RipleysK::new(vec![Point2D::new(5.0, 5.0)], rect_window());
        assert!((k.compute(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_k_function_increases_with_r() {
        let mut rng = thread_rng();
        let ppp = HomogeneousPoisson::new(1.0, rect_window());
        let points = ppp.sample(&mut rng);
        let k = RipleysK::new(points, rect_window());

        let k1 = k.compute(0.5);
        let k2 = k.compute(1.0);
        let k3 = k.compute(2.0);
        assert!(k1 <= k2, "K(0.5)={k1} should be ≤ K(1.0)={k2}");
        assert!(k2 <= k3, "K(1.0)={k2} should be ≤ K(2.0)={k3}");
    }

    #[test]
    fn test_l_function_near_zero_for_csr() {
        let mut rng = thread_rng();
        let ppp = HomogeneousPoisson::new(2.0, rect_window());
        let points = ppp.sample(&mut rng);
        let l = RipleysL::new(points, rect_window());

        let l_val = l.compute(1.0);
        // Under CSR, L(r) ≈ 0
        assert!(
            l_val.abs() < 2.0,
            "L(1.0) = {l_val}, expected near 0"
        );
    }

    #[test]
    fn test_l_function_csr_constant() {
        assert!((RipleysL::csr_l(1.0)).abs() < 1e-10);
        assert!((RipleysL::csr_l(5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_csr_k_formula() {
        let r = 2.0;
        assert!(
            (RipleysK::csr_k(r) - std::f64::consts::PI * 4.0).abs() < 1e-10
        );
    }

    #[test]
    fn test_compute_curve() {
        let k = RipleysK::new(
            vec![Point2D::new(1.0, 1.0), Point2D::new(2.0, 2.0)],
            rect_window(),
        );
        let curve = k.compute_curve(&[0.5, 1.0, 2.0]);
        assert_eq!(curve.len(), 3);
        assert!((curve[0].0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_l_function_curve() {
        let l = RipleysL::new(
            vec![Point2D::new(1.0, 1.0), Point2D::new(2.0, 2.0)],
            rect_window(),
        );
        let curve = l.compute_curve(&[0.5, 1.0]);
        assert_eq!(curve.len(), 2);
    }

    #[test]
    fn test_k_function_clustered_pattern() {
        // Create a clustered pattern: points near origin
        let points: Vec<Point2D> = (0..20)
            .map(|_| Point2D::new(
                rand::thread_rng().gen_range(-0.5..0.5),
                rand::thread_rng().gen_range(-0.5..0.5),
            ))
            .collect();
        let w = Window::Rect { xmin: -5.0, xmax: 5.0, ymin: -5.0, ymax: 5.0 };
        let k = RipleysK::new(points, w);
        let k_val = k.compute(1.0);
        let theoretical = std::f64::consts::PI;
        // Clustered → K(r) > π r²
        assert!(k_val > theoretical, "Clustered K(1)={k_val} should be > π={theoretical}");
    }
}
