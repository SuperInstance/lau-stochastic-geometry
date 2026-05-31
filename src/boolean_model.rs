//! Boolean model — random grains (disks) centered on a point process.

use crate::poisson::{HomogeneousPoisson, Point2D, Window};
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Distribution of grain radii.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrainDistribution {
    /// Fixed radius.
    Constant(f64),
    /// Uniform on [min, max].
    Uniform { min: f64, max: f64 },
    /// Exponential with given mean.
    Exponential { mean: f64 },
}

impl GrainDistribution {
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            GrainDistribution::Constant(r) => *r,
            GrainDistribution::Uniform { min, max } => rng.gen_range(*min..*max),
            GrainDistribution::Exponential { mean } => {
                use rand_distr::Exp;
                let dist = Exp::new(1.0 / mean).unwrap();
                rng.sample(dist)
            }
        }
    }

    pub fn mean(&self) -> f64 {
        match self {
            GrainDistribution::Constant(r) => *r,
            GrainDistribution::Uniform { min, max } => (min + max) / 2.0,
            GrainDistribution::Exponential { mean } => *mean,
        }
    }
}

/// Boolean model: union of random grains centered on a Poisson point process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BooleanModel {
    /// The underlying Poisson point process.
    pub ppp: HomogeneousPoisson,
    /// Distribution of grain radii.
    pub grain_distribution: GrainDistribution,
}

impl BooleanModel {
    pub fn new(ppp: HomogeneousPoisson, grain_distribution: GrainDistribution) -> Self {
        Self { ppp, grain_distribution }
    }

    /// Generate a realization: returns (center, radius) for each grain.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<(Point2D, f64)> {
        let centers = self.ppp.sample(rng);
        centers
            .into_iter()
            .map(|p| {
                let r = self.grain_distribution.sample(rng);
                (p, r)
            })
            .collect()
    }

    /// Theoretical coverage fraction: p = 1 - exp(-λ * E[π R²]).
    pub fn coverage_fraction(&self) -> f64 {
        let mean_area = std::f64::consts::PI * self.grain_distribution.mean().powi(2);
        1.0 - (-self.ppp.intensity * mean_area).exp()
    }

    /// Expected number of grains.
    pub fn expected_grain_count(&self) -> f64 {
        self.ppp.expected_count()
    }

    /// Mean area of a single grain.
    pub fn mean_grain_area(&self) -> f64 {
        std::f64::consts::PI * self.grain_distribution.mean().powi(2)
    }

    /// Mean perimeter of a single grain.
    pub fn mean_grain_perimeter(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.grain_distribution.mean()
    }

    /// Check if a query point is covered by any grain.
    pub fn is_covered(&self, query: &Point2D, grains: &[(Point2D, f64)]) -> bool {
        for (center, radius) in grains {
            if query.distance_to(center) <= *radius {
                return true;
            }
        }
        false
    }

    /// Estimate coverage fraction empirically by Monte Carlo on a grid.
    pub fn estimate_coverage(
        &self,
        grains: &[(Point2D, f64)],
        window: &Window,
        grid_res: usize,
    ) -> f64 {
        let dx = window.width() / grid_res as f64;
        let dy = window.height() / grid_res as f64;
        let mut covered = 0u64;
        let mut total = 0u64;

        for i in 0..grid_res {
            for j in 0..grid_res {
                let x = window.rect_xmin() + (i as f64 + 0.5) * dx;
                let y = window.rect_ymin() + (j as f64 + 0.5) * dy;
                let p = Point2D::new(x, y);
                if window.contains(&p) {
                    total += 1;
                    if self.is_covered(&p, grains) {
                        covered += 1;
                    }
                }
            }
        }

        if total == 0 {
            0.0
        } else {
            covered as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poisson::Window;
    use rand::thread_rng;

    fn make_model(intensity: f64, grain_r: f64) -> BooleanModel {
        let w = Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
        let ppp = HomogeneousPoisson::new(intensity, w);
        BooleanModel::new(ppp, GrainDistribution::Constant(grain_r))
    }

    #[test]
    fn test_coverage_fraction_zero_intensity() {
        let bm = make_model(0.0, 1.0);
        assert!((bm.coverage_fraction()).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_fraction_between_0_and_1() {
        let bm = make_model(0.1, 1.0);
        let cf = bm.coverage_fraction();
        assert!(cf > 0.0 && cf < 1.0);
    }

    #[test]
    fn test_coverage_fraction_increases_with_intensity() {
        let bm1 = make_model(0.1, 1.0);
        let bm2 = make_model(0.5, 1.0);
        assert!(bm1.coverage_fraction() < bm2.coverage_fraction());
    }

    #[test]
    fn test_expected_grain_count() {
        let bm = make_model(1.0, 0.5);
        assert!((bm.expected_grain_count() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_grain_area() {
        let bm = make_model(1.0, 1.0);
        let expected = std::f64::consts::PI;
        assert!((bm.mean_grain_area() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mean_grain_perimeter() {
        let bm = make_model(1.0, 1.0);
        let expected = 2.0 * std::f64::consts::PI;
        assert!((bm.mean_grain_perimeter() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sample_grains() {
        let bm = make_model(1.0, 0.5);
        let mut rng = thread_rng();
        let grains = bm.sample(&mut rng);
        for (p, r) in &grains {
            assert!(bm.ppp.window.contains(p));
            assert!(*r > 0.0);
        }
    }

    #[test]
    fn test_is_covered() {
        let bm = make_model(1.0, 1.0);
        let grains = vec![
            (Point2D::new(5.0, 5.0), 2.0),
        ];
        assert!(bm.is_covered(&Point2D::new(5.0, 5.0), &grains));
        assert!(bm.is_covered(&Point2D::new(6.0, 5.0), &grains));
        assert!(!bm.is_covered(&Point2D::new(8.0, 8.0), &grains));
    }

    #[test]
    fn test_estimate_coverage_matches_theory() {
        let bm = make_model(0.05, 1.0);
        let mut rng = thread_rng();
        let mut coverages = Vec::new();
        for _ in 0..20 {
            let grains = bm.sample(&mut rng);
            let est = bm.estimate_coverage(&grains, &bm.ppp.window, 50);
            coverages.push(est);
        }
        let mean_coverage: f64 = coverages.iter().sum::<f64>() / coverages.len() as f64;
        let theoretical = bm.coverage_fraction();
        // Allow generous tolerance for Monte Carlo
        assert!(
            (mean_coverage - theoretical).abs() < 0.15,
            "mean_coverage={mean_coverage}, theoretical={theoretical}"
        );
    }

    #[test]
    fn test_uniform_grain_distribution() {
        let w = Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
        let ppp = HomogeneousPoisson::new(1.0, w);
        let bm = BooleanModel::new(ppp, GrainDistribution::Uniform { min: 0.1, max: 2.0 });
        let mut rng = thread_rng();
        let grains = bm.sample(&mut rng);
        for (_, r) in &grains {
            assert!(*r >= 0.1 && *r <= 2.0);
        }
    }

    #[test]
    fn test_exponential_grain_distribution() {
        let w = Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
        let ppp = HomogeneousPoisson::new(1.0, w);
        let bm = BooleanModel::new(ppp, GrainDistribution::Exponential { mean: 1.0 });
        let mut rng = thread_rng();
        let grains = bm.sample(&mut rng);
        for (_, r) in &grains {
            assert!(*r >= 0.0);
        }
    }
}
