//! Agent uncertainty quantification using stochastic geometry.
//!
//! Models agent uncertainty as spatial point patterns and confidence regions as random sets.

use crate::boolean_model::{BooleanModel, GrainDistribution};
use crate::poisson::{HomogeneousPoisson, Point2D, Window};
use crate::ripley::RipleysL;
use crate::minkowski::MinkowskiFunctionals;
use crate::random_field::{GaussianRandomField, MaternParams};
use crate::voronoi::VoronoiTessellation;
use rand::Rng;
use serde::{Serialize, Deserialize};
use nalgebra::DMatrix;

/// An agent observation with uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentObservation {
    /// Estimated position.
    pub position: Point2D,
    /// Position uncertainty (standard deviation).
    pub sigma: f64,
    /// Confidence level (0 to 1).
    pub confidence: f64,
    /// Timestamp (optional).
    pub timestamp: Option<f64>,
}

impl AgentObservation {
    pub fn new(x: f64, y: f64, sigma: f64, confidence: f64) -> Self {
        Self {
            position: Point2D::new(x, y),
            sigma,
            confidence,
            timestamp: None,
        }
    }
}

/// Summary of uncertainty analysis for an agent's observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyReport {
    /// Number of observations.
    pub n_observations: usize,
    /// Estimated intensity of observations.
    pub intensity: f64,
    /// Ripley's L at several radii (r, L(r)).
    pub l_function: Vec<(f64, f64)>,
    /// Minkowski functionals of the confidence region.
    pub minkowski: MinkowskiFunctionals,
    /// Coverage fraction of the confidence region.
    pub coverage_fraction: f64,
    /// Whether the confidence region percolates.
    pub percolates: bool,
}

/// Agent uncertainty quantification tools.
pub struct AgentUncertainty {
    /// Observation window.
    pub window: Window,
}

impl AgentUncertainty {
    pub fn new(window: Window) -> Self {
        Self { window }
    }

    /// Model agent observations as a Poisson point process.
    /// Returns the fitted intensity.
    pub fn fit_poisson_intensity(&self, observations: &[AgentObservation]) -> f64 {
        observations.len() as f64 / self.window.area()
    }

    /// Generate a confidence region (Boolean model) from observations.
    /// Each observation becomes a grain with radius proportional to its uncertainty.
    pub fn confidence_region(
        &self,
        observations: &[AgentObservation],
        confidence_multiplier: f64,
    ) -> Vec<(Point2D, f64)> {
        observations
            .iter()
            .map(|obs| {
                // Confidence disk radius: sigma * z_{1-(1-confidence)/2}
                // For confidence=0.95, z≈1.96
                let z = if obs.confidence >= 0.99 {
                    2.576
                } else if obs.confidence >= 0.95 {
                    1.96
                } else if obs.confidence >= 0.90 {
                    1.645
                } else if obs.confidence >= 0.80 {
                    1.282
                } else {
                    1.0
                };
                let radius = obs.sigma * z * confidence_multiplier;
                (obs.position.clone(), radius)
            })
            .collect()
    }

    /// Perform spatial clustering analysis using Ripley's K/L functions.
    pub fn spatial_analysis(
        &self,
        observations: &[AgentObservation],
        radii: &[f64],
    ) -> Vec<(f64, f64, String)> {
        let points: Vec<Point2D> = observations.iter().map(|o| o.position.clone()).collect();
        let l = RipleysL::new(points, self.window.clone());

        radii
            .iter()
            .map(|&r| {
                let l_val = l.compute(r);
                let pattern = if l_val > 0.5 {
                    "clustered".to_string()
                } else if l_val < -0.5 {
                    "regular".to_string()
                } else {
                    "random".to_string()
                };
                (r, l_val, pattern)
            })
            .collect()
    }

    /// Full uncertainty report.
    pub fn analyze<R: Rng + ?Sized>(
        &self,
        observations: &[AgentObservation],
        _rng: &mut R,
        grid_res: usize,
    ) -> UncertaintyReport {
        let n = observations.len();
        let intensity = self.fit_poisson_intensity(observations);

        // L-function
        let radii: Vec<f64> = (1..=10).map(|i| i as f64 * 0.5).collect();
        let points: Vec<Point2D> = observations.iter().map(|o| o.position.clone()).collect();
        let l = RipleysL::new(points, self.window.clone());
        let l_function = l.compute_curve(&radii);

        // Confidence region as Boolean model grains
        let grains = self.confidence_region(observations, 1.0);
        let minkowski = MinkowskiFunctionals::from_grains(&grains, &self.window, grid_res);

        // Coverage
        let bm = BooleanModel::new(
            HomogeneousPoisson::new(intensity, self.window.clone()),
            GrainDistribution::Constant(
                observations
                    .iter()
                    .map(|o| o.sigma)
                    .sum::<f64>()
                    / n.max(1) as f64,
            ),
        );
        let coverage_fraction = bm.coverage_fraction();

        // Percolation check
        let detector = crate::percolation::PercolationDetector::new(grid_res);
        let perc_result = detector.check(&grains, &self.window);

        UncertaintyReport {
            n_observations: n,
            intensity,
            l_function,
            minkowski,
            coverage_fraction,
            percolates: perc_result.percolates,
        }
    }

    /// Simulate uncertain observations from a ground truth point pattern.
    pub fn simulate_uncertain_observations<R: Rng + ?Sized>(
        &self,
        true_points: &[Point2D],
        sigma: f64,
        detection_probability: f64,
        rng: &mut R,
    ) -> Vec<AgentObservation> {
        use rand_distr::Normal;
        let normal = Normal::new(0.0, sigma).unwrap();

        let mut obs = Vec::new();
        for p in true_points {
            if rng.gen::<f64>() < detection_probability {
                let dx: f64 = rng.sample(normal);
                let dy: f64 = rng.sample(normal);
                obs.push(AgentObservation::new(p.x + dx, p.y + dy, sigma, 0.95));
            }
        }
        obs
    }

    /// Create a Gaussian random field representing spatial uncertainty.
    pub fn uncertainty_field<R: Rng + ?Sized>(
        &self,
        observations: &[AgentObservation],
        rng: &mut R,
        grid_res: usize,
    ) -> DMatrix<f64> {
        let mean_sigma = observations.iter().map(|o| o.sigma).sum::<f64>()
            / observations.len().max(1) as f64;

        let params = MaternParams::new(mean_sigma * mean_sigma, mean_sigma, 1.5);
        let grf = GaussianRandomField::new(params, grid_res);
        grf.sample(&self.window, rng)
    }

    /// Tessellate the observation space and assign each cell to nearest observation.
    pub fn tessellate(&self, observations: &[AgentObservation]) -> VoronoiTessellation {
        let points: Vec<Point2D> = observations.iter().map(|o| o.position.clone()).collect();
        VoronoiTessellation::from_points(points, self.window.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn rect_window() -> Window {
        Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 }
    }

    fn make_observations() -> Vec<AgentObservation> {
        vec![
            AgentObservation::new(2.0, 2.0, 0.5, 0.95),
            AgentObservation::new(5.0, 5.0, 0.3, 0.95),
            AgentObservation::new(8.0, 8.0, 0.7, 0.90),
            AgentObservation::new(3.0, 7.0, 0.4, 0.95),
            AgentObservation::new(7.0, 3.0, 0.6, 0.90),
        ]
    }

    #[test]
    fn test_fit_poisson_intensity() {
        let au = AgentUncertainty::new(rect_window());
        let obs = make_observations();
        let intensity = au.fit_poisson_intensity(&obs);
        assert!((intensity - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_region() {
        let au = AgentUncertainty::new(rect_window());
        let obs = make_observations();
        let grains = au.confidence_region(&obs, 1.0);
        assert_eq!(grains.len(), obs.len());
        for (_, r) in &grains {
            assert!(r > &0.0);
        }
    }

    #[test]
    fn test_confidence_region_radii_proportional_to_sigma() {
        let au = AgentUncertainty::new(rect_window());
        let obs = vec![
            AgentObservation::new(5.0, 5.0, 0.5, 0.95),
            AgentObservation::new(5.0, 5.0, 1.0, 0.95),
        ];
        let grains = au.confidence_region(&obs, 1.0);
        assert!(grains[1].1 > grains[0].1);
    }

    #[test]
    fn test_spatial_analysis() {
        let au = AgentUncertainty::new(rect_window());
        let obs = make_observations();
        let result = au.spatial_analysis(&obs, &[0.5, 1.0, 2.0]);
        assert_eq!(result.len(), 3);
        for (r, l_val, pattern) in &result {
            assert!(l_val.is_finite());
            assert!(*r > 0.0);
            assert!(!pattern.is_empty());
        }
    }

    #[test]
    fn test_simulate_uncertain_observations() {
        let au = AgentUncertainty::new(rect_window());
        let true_points = vec![
            Point2D::new(3.0, 3.0),
            Point2D::new(7.0, 7.0),
        ];
        let mut rng = thread_rng();
        let obs = au.simulate_uncertain_observations(&true_points, 0.5, 1.0, &mut rng);
        // With p=1.0, should get all points
        assert_eq!(obs.len(), 2);
        for o in &obs {
            assert!((o.sigma - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simulate_with_detection_probability() {
        let au = AgentUncertainty::new(rect_window());
        let true_points: Vec<Point2D> = (0..100)
            .map(|i| Point2D::new(i as f64 % 10.0, i as f64 / 10.0))
            .collect();
        let mut rng = thread_rng();
        let obs = au.simulate_uncertain_observations(&true_points, 0.1, 0.5, &mut rng);
        // With p=0.5, should get roughly half
        assert!(obs.len() < 90 && obs.len() > 10);
    }

    #[test]
    fn test_uncertainty_field() {
        let au = AgentUncertainty::new(rect_window());
        let obs = make_observations();
        let mut rng = thread_rng();
        let field = au.uncertainty_field(&obs, &mut rng, 5);
        assert_eq!(field.nrows(), 5);
        assert_eq!(field.ncols(), 5);
    }

    #[test]
    fn test_tessellate() {
        let au = AgentUncertainty::new(rect_window());
        let obs = make_observations();
        let tess = au.tessellate(&obs);
        assert_eq!(tess.cells.len(), obs.len());
    }

    #[test]
    fn test_full_analysis() {
        let au = AgentUncertainty::new(rect_window());
        let obs = make_observations();
        let mut rng = thread_rng();
        let report = au.analyze(&obs, &mut rng, 50);
        assert_eq!(report.n_observations, 5);
        assert!((report.intensity - 0.05).abs() < 1e-10);
        assert_eq!(report.l_function.len(), 10);
        assert!(report.coverage_fraction >= 0.0);
    }

    #[test]
    fn test_observation_serialization() {
        let obs = AgentObservation::new(1.0, 2.0, 0.5, 0.95);
        let json = serde_json::to_string(&obs).unwrap();
        let deserialized: AgentObservation = serde_json::from_str(&json).unwrap();
        assert!((deserialized.position.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_observations() {
        let au = AgentUncertainty::new(rect_window());
        let intensity = au.fit_poisson_intensity(&[]);
        assert!((intensity).abs() < 1e-10);
    }
}
