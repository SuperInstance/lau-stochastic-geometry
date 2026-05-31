//! Poisson point processes — homogeneous and inhomogeneous.

use rand::Rng;
use serde::{Serialize, Deserialize};

/// A 2D point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn norm(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

/// Bounding window for point processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Window {
    Rect { xmin: f64, xmax: f64, ymin: f64, ymax: f64 },
    Disk { cx: f64, cy: f64, radius: f64 },
}

impl Window {
    pub fn area(&self) -> f64 {
        match self {
            Window::Rect { xmin, xmax, ymin, ymax } => (xmax - xmin) * (ymax - ymin),
            Window::Disk { radius, .. } => std::f64::consts::PI * radius * radius,
        }
    }

    pub fn contains(&self, p: &Point2D) -> bool {
        match self {
            Window::Rect { xmin, xmax, ymin, ymax } => {
                p.x >= *xmin && p.x <= *xmax && p.y >= *ymin && p.y <= *ymax
            }
            Window::Disk { cx, cy, radius } => {
                ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt() <= *radius
            }
        }
    }

    pub fn width(&self) -> f64 {
        match self {
            Window::Rect { xmin, xmax, .. } => xmax - xmin,
            Window::Disk { radius, .. } => 2.0 * radius,
        }
    }

    pub fn height(&self) -> f64 {
        match self {
            Window::Rect { ymin, ymax, .. } => ymax - ymin,
            Window::Disk { radius, .. } => 2.0 * radius,
        }
    }

    pub fn rect_xmin(&self) -> f64 {
        match self {
            Window::Rect { xmin, .. } => *xmin,
            Window::Disk { cx, radius, .. } => cx - radius,
        }
    }

    pub fn rect_xmax(&self) -> f64 {
        match self {
            Window::Rect { xmax, .. } => *xmax,
            Window::Disk { cx, radius, .. } => cx + radius,
        }
    }

    pub fn rect_ymin(&self) -> f64 {
        match self {
            Window::Rect { ymin, .. } => *ymin,
            Window::Disk { cy, radius, .. } => cy - radius,
        }
    }

    pub fn rect_ymax(&self) -> f64 {
        match self {
            Window::Rect { ymax, .. } => *ymax,
            Window::Disk { cy, radius, .. } => cy + radius,
        }
    }
}

/// Homogeneous Poisson point process with constant intensity λ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomogeneousPoisson {
    /// Intensity (points per unit area).
    pub intensity: f64,
    /// Observation window.
    pub window: Window,
}

impl HomogeneousPoisson {
    pub fn new(intensity: f64, window: Window) -> Self {
        assert!(intensity >= 0.0, "Intensity must be non-negative");
        Self { intensity, window }
    }

    /// Generate a realization of the Poisson point process.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<Point2D> {
        if self.intensity == 0.0 {
            return Vec::new();
        }
        let area = self.window.area();
        let mean_count = self.intensity * area;

        // Number of points ~ Poisson(λ * area)
        let n = poisson_sample(rng, mean_count);

        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            points.push(self.sample_point_in_window(rng));
        }
        points
    }

    /// Expected number of points.
    pub fn expected_count(&self) -> f64 {
        self.intensity * self.window.area()
    }

    /// Variance of the count (equals mean for Poisson).
    pub fn count_variance(&self) -> f64 {
        self.expected_count()
    }

    fn sample_point_in_window<R: Rng + ?Sized>(&self, rng: &mut R) -> Point2D {
        match &self.window {
            Window::Rect { xmin, xmax, ymin, ymax } => {
                let x = rng.gen_range(*xmin..*xmax);
                let y = rng.gen_range(*ymin..*ymax);
                Point2D::new(x, y)
            }
            Window::Disk { cx, cy, radius } => {
                // Rejection sampling for uniform disk
                loop {
                    let x = rng.gen_range(cx - radius..cx + radius);
                    let y = rng.gen_range(cy - radius..cy + radius);
                    if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= *radius {
                        return Point2D::new(x, y);
                    }
                }
            }
        }
    }

    /// Probability of observing exactly n points.
    pub fn pmf_count(&self, n: u32) -> f64 {
        let mean = self.expected_count();
        poisson_pmf(n, mean)
    }

    /// Empty space function F(r): probability of finding at least one point within distance r
    /// of an arbitrary location. For homogeneous PPP: F(r) = 1 - exp(-λ * π * r²).
    pub fn empty_space_function(&self, r: f64) -> f64 {
        1.0 - (-self.intensity * std::f64::consts::PI * r * r).exp()
    }
}

/// Inhomogeneous Poisson point process with spatially varying intensity.
pub struct InhomogeneousPoisson {
    /// Intensity function: takes (x, y) and returns local intensity.
    pub intensity_fn: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
    /// Observation window.
    pub window: Window,
    /// Upper bound on intensity (for thinning).
    pub lambda_max: f64,
}

impl InhomogeneousPoisson {
    pub fn new<F>(intensity_fn: F, window: Window, lambda_max: f64) -> Self
    where
        F: Fn(f64, f64) -> f64 + Send + Sync + 'static,
    {
        assert!(lambda_max > 0.0, "lambda_max must be positive");
        Self {
            intensity_fn: Box::new(intensity_fn),
            window,
            lambda_max,
        }
    }

    /// Generate a realization via Lewis-Shedler thinning.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<Point2D> {
        let area = self.window.area();
        let mean_count = self.lambda_max * area;
        let n = poisson_sample(rng, mean_count);

        let mut points = Vec::new();
        for _ in 0..n {
            // Sample uniformly, then thin
            let p = self.sample_uniform(rng);
            let local_intensity = (self.intensity_fn)(p.x, p.y);
            let accept_prob = local_intensity / self.lambda_max;
            if rng.gen::<f64>() < accept_prob {
                points.push(p);
            }
        }
        points
    }

    /// Approximate expected count via numerical integration.
    pub fn expected_count(&self, grid_res: usize) -> f64 {
        let dx = self.window.width() / grid_res as f64;
        let dy = self.window.height() / grid_res as f64;
        let mut total = 0.0;
        for i in 0..grid_res {
            for j in 0..grid_res {
                let x = self.window.rect_xmin() + (i as f64 + 0.5) * dx;
                let y = self.window.rect_ymin() + (j as f64 + 0.5) * dy;
                let p = Point2D::new(x, y);
                if self.window.contains(&p) {
                    total += (self.intensity_fn)(x, y) * dx * dy;
                }
            }
        }
        total
    }

    fn sample_uniform<R: Rng + ?Sized>(&self, rng: &mut R) -> Point2D {
        match &self.window {
            Window::Rect { xmin, xmax, ymin, ymax } => {
                Point2D::new(rng.gen_range(*xmin..*xmax), rng.gen_range(*ymin..*ymax))
            }
            Window::Disk { cx, cy, radius } => {
                loop {
                    let x = rng.gen_range(cx - radius..cx + radius);
                    let y = rng.gen_range(cy - radius..cy + radius);
                    if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= *radius {
                        return Point2D::new(x, y);
                    }
                }
            }
        }
    }
}

/// Sample from Poisson(λ) using Knuth's algorithm for small λ,
/// and normal approximation for large λ.
pub fn poisson_sample<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> usize {
    if lambda < 30.0 {
        // Knuth's algorithm
        let l = (-lambda).exp();
        let mut k = 0usize;
        let mut p = 1.0;
        loop {
            p *= rng.gen::<f64>();
            if p <= l {
                break;
            }
            k += 1;
        }
        k
    } else {
        // Normal approximation (accurate enough for large λ)
        use rand_distr::Normal;
        let normal = Normal::new(lambda, lambda.sqrt()).unwrap();
        let v: f64 = rng.sample(normal);
        v.round().max(0.0) as usize
    }
}

/// Poisson PMF: P(X = k | λ).
pub fn poisson_pmf(k: u32, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    let log_pmf = k as f64 * lambda.ln() - lambda - ln_factorial(k);
    log_pmf.exp()
}

/// Natural log of factorial.
pub fn ln_factorial(n: u32) -> f64 {
    // Stirling's approximation for large n
    if n <= 20 {
        (1..=n).map(|i| (i as f64).ln()).sum()
    } else {
        let n_f = n as f64;
        n_f * n_f.ln() - n_f + 0.5 * (2.0 * std::f64::consts::PI * n_f).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn rect_window() -> Window {
        Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 }
    }

    #[test]
    fn test_homogeneous_expected_count() {
        let ppp = HomogeneousPoisson::new(2.0, rect_window());
        assert!((ppp.expected_count() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_homogeneous_variance_equals_mean() {
        let ppp = HomogeneousPoisson::new(5.0, rect_window());
        assert!((ppp.count_variance() - ppp.expected_count()).abs() < 1e-10);
    }

    #[test]
    fn test_homogeneous_sample_count_distribution() {
        let ppp = HomogeneousPoisson::new(1.0, rect_window());
        let mut rng = thread_rng();
        let n_samples = 1000;
        let mut counts: Vec<f64> = (0..n_samples)
            .map(|_| ppp.sample(&mut rng).len() as f64)
            .collect();
        let mean = counts.iter().sum::<f64>() / n_samples as f64;
        // Expected mean = 100, allow 10% tolerance
        assert!((mean - 100.0).abs() < 10.0, "Mean was {mean}");
    }

    #[test]
    fn test_homogeneous_zero_intensity() {
        let ppp = HomogeneousPoisson::new(0.0, rect_window());
        let mut rng = thread_rng();
        assert!(ppp.sample(&mut rng).is_empty());
    }

    #[test]
    fn test_homogeneous_points_in_window() {
        let ppp = HomogeneousPoisson::new(5.0, rect_window());
        let mut rng = thread_rng();
        let points = ppp.sample(&mut rng);
        for p in &points {
            assert!(ppp.window.contains(p));
        }
    }

    #[test]
    fn test_poisson_pmf_sums_near_one() {
        let lambda = 5.0;
        let sum: f64 = (0..50).map(|k| poisson_pmf(k, lambda)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_poisson_pmf_zero_lambda() {
        assert!((poisson_pmf(0, 0.0) - 1.0).abs() < 1e-10);
        assert!((poisson_pmf(1, 0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_empty_space_function() {
        let ppp = HomogeneousPoisson::new(1.0, rect_window());
        // F(0) = 0
        assert!((ppp.empty_space_function(0.0)).abs() < 1e-10);
        // F(∞) → 1
        assert!((ppp.empty_space_function(100.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_disk_window_area() {
        let w = Window::Disk { cx: 0.0, cy: 0.0, radius: 1.0 };
        assert!((w.area() - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_disk_window_contains() {
        let w = Window::Disk { cx: 0.0, cy: 0.0, radius: 1.0 };
        assert!(w.contains(&Point2D::new(0.0, 0.0)));
        assert!(w.contains(&Point2D::new(0.5, 0.5)));
        assert!(!w.contains(&Point2D::new(2.0, 0.0)));
    }

    #[test]
    fn test_point_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_inhomogeneous_sample() {
        let ppp = InhomogeneousPoisson::new(
            |x, _y| x / 10.0,
            rect_window(),
            1.0,
        );
        let mut rng = thread_rng();
        let points = ppp.sample(&mut rng);
        // All points should be in window
        for p in &points {
            assert!(ppp.window.contains(p));
        }
    }

    #[test]
    fn test_inhomogeneous_expected_count() {
        // Uniform intensity 1.0 → should give area
        let ppp = InhomogeneousPoisson::new(
            |_x, _y| 1.0,
            rect_window(),
            1.0,
        );
        let expected = ppp.expected_count(100);
        assert!((expected - 100.0).abs() < 5.0, "Expected ~100, got {expected}");
    }

    #[test]
    fn test_homogeneous_poisson_serialization() {
        let ppp = HomogeneousPoisson::new(2.0, rect_window());
        let json = serde_json::to_string(&ppp).unwrap();
        let deserialized: HomogeneousPoisson = serde_json::from_str(&json).unwrap();
        assert!((deserialized.intensity - 2.0).abs() < 1e-10);
    }
}
