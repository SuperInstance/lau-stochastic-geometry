//! Percolation threshold detection for random sets.

use crate::boolean_model::BooleanModel;
use crate::poisson::{Point2D, Window};
use rand::Rng;

/// Result of a percolation analysis.
#[derive(Debug, Clone)]
pub struct PercolationResult {
    pub parameter: f64,
    pub percolates: bool,
    pub largest_component_fraction: f64,
    pub num_components: usize,
}

/// Percolation detector for Boolean models.
#[derive(Debug, Clone)]
pub struct PercolationDetector {
    /// Grid resolution for discretization.
    pub grid_res: usize,
    /// Direction to check for percolation.
    pub direction: PercolationDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PercolationDirection {
    /// Check for a path from left to right.
    Horizontal,
    /// Check for a path from bottom to top.
    Vertical,
    /// Check both directions.
    Both,
}

impl PercolationDetector {
    pub fn new(grid_res: usize) -> Self {
        Self {
            grid_res,
            direction: PercolationDirection::Both,
        }
    }

    pub fn with_direction(mut self, dir: PercolationDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Check if a Boolean model realization percolates.
    pub fn check(
        &self,
        grains: &[(Point2D, f64)],
        window: &Window,
    ) -> PercolationResult {
        let grid = self.build_grid(grains, window);
        let (components, labels) = self.find_components(&grid);

        let max_component = components.iter().map(|c| c.len()).max().unwrap_or(0);
        let total_occupied: usize = grid.iter().flatten().filter(|&&v| v).count();
        let fraction = if total_occupied > 0 {
            max_component as f64 / total_occupied as f64
        } else {
            0.0
        };

        let percolates = self.check_percolation(&labels);

        // Find the "parameter" — approximate coverage fraction
        let parameter = total_occupied as f64 / (self.grid_res * self.grid_res) as f64;

        PercolationResult {
            parameter,
            percolates,
            largest_component_fraction: fraction,
            num_components: components.len(),
        }
    }

    /// Find the critical intensity for percolation using binary search.
    pub fn find_critical_intensity<R: Rng + ?Sized>(
        &self,
        grain_radius: f64,
        window: &Window,
        rng: &mut R,
        trials: usize,
        target_probability: f64,
    ) -> f64 {
        let mut lo = 0.0;
        let mut hi = 5.0; // Start with high upper bound

        // Increase upper bound if needed
        for _ in 0..5 {
            if self.estimate_percolation_probability(lo, hi, grain_radius, window, rng, trials)
                >= target_probability
            {
                break;
            }
            lo = hi;
            hi *= 2.0;
        }

        // Binary search
        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            let prob = self.estimate_percolation_probability(lo, mid, grain_radius, window, rng, trials);
            if prob < target_probability {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        (lo + hi) / 2.0
    }

    fn estimate_percolation_probability<R: Rng + ?Sized>(
        &self,
        _lo: f64,
        intensity: f64,
        grain_radius: f64,
        window: &Window,
        rng: &mut R,
        trials: usize,
    ) -> f64 {
        let ppp = crate::poisson::HomogeneousPoisson::new(intensity, window.clone());
        let bm = BooleanModel::new(ppp, crate::boolean_model::GrainDistribution::Constant(grain_radius));
        let mut percolation_count = 0;
        for _ in 0..trials {
            let grains = bm.sample(rng);
            let result = self.check(&grains, window);
            if result.percolates {
                percolation_count += 1;
            }
        }
        percolation_count as f64 / trials as f64
    }

    fn build_grid(&self, grains: &[(Point2D, f64)], window: &Window) -> Vec<Vec<bool>> {
        let n = self.grid_res;
        let dx = window.width() / n as f64;
        let dy = window.height() / n as f64;

        let mut grid = vec![vec![false; n]; n];

        for i in 0..n {
            for j in 0..n {
                let x = window.rect_xmin() + (i as f64 + 0.5) * dx;
                let y = window.rect_ymin() + (j as f64 + 0.5) * dy;
                let p = Point2D::new(x, y);
                for (center, radius) in grains {
                    if p.distance_to(center) <= *radius {
                        grid[i][j] = true;
                        break;
                    }
                }
            }
        }

        grid
    }

    /// Find connected components using flood fill.
    fn find_components(&self, grid: &Vec<Vec<bool>>) -> (Vec<Vec<(usize, usize)>>, Vec<Vec<usize>>) {
        let n = grid.len();
        let mut labels = vec![vec![0usize; n]; n];
        let mut components: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut current_label = 1usize;

        for i in 0..n {
            for j in 0..n {
                if grid[i][j] && labels[i][j] == 0 {
                    let mut component = Vec::new();
                    let mut stack = vec![(i, j)];
                    while let Some((ci, cj)) = stack.pop() {
                        if labels[ci][cj] != 0 || !grid[ci][cj] {
                            continue;
                        }
                        labels[ci][cj] = current_label;
                        component.push((ci, cj));

                        // 4-connectivity neighbors
                        if ci > 0 { stack.push((ci - 1, cj)); }
                        if ci + 1 < n { stack.push((ci + 1, cj)); }
                        if cj > 0 { stack.push((ci, cj - 1)); }
                        if cj + 1 < n { stack.push((ci, cj + 1)); }
                    }
                    components.push(component);
                    current_label += 1;
                }
            }
        }

        (components, labels)
    }

    /// Check if any component percolates from one side to the other.
    fn check_percolation(&self, labels: &Vec<Vec<usize>>) -> bool {
        let n = labels.len();
        if n == 0 {
            return false;
        }

        // Find labels that touch left edge and right edge (horizontal percolation)
        if self.direction == PercolationDirection::Horizontal || self.direction == PercolationDirection::Both {
            let mut left_labels = std::collections::HashSet::new();
            for j in 0..n {
                if labels[0][j] > 0 {
                    left_labels.insert(labels[0][j]);
                }
            }
            for j in 0..n {
                if labels[n - 1][j] > 0 && left_labels.contains(&labels[n - 1][j]) {
                    return true;
                }
            }
        }

        // Find labels that touch bottom edge and top edge (vertical percolation)
        if self.direction == PercolationDirection::Vertical || self.direction == PercolationDirection::Both {
            let mut bottom_labels = std::collections::HashSet::new();
            for i in 0..n {
                if labels[i][0] > 0 {
                    bottom_labels.insert(labels[i][0]);
                }
            }
            for i in 0..n {
                if labels[i][n - 1] > 0 && bottom_labels.contains(&labels[i][n - 1]) {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_model::{BooleanModel, GrainDistribution};
    use crate::poisson::HomogeneousPoisson;
    use rand::thread_rng;

    fn rect_window() -> Window {
        Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 }
    }

    #[test]
    fn test_no_percolation_sparse() {
        let det = PercolationDetector::new(100);
        let ppp = HomogeneousPoisson::new(0.001, rect_window());
        let bm = BooleanModel::new(ppp, GrainDistribution::Constant(0.1));
        let mut rng = thread_rng();

        let mut percolates = false;
        for _ in 0..10 {
            let grains = bm.sample(&mut rng);
            let result = det.check(&grains, &rect_window());
            if result.percolates {
                percolates = true;
            }
        }
        assert!(!percolates, "Sparse model should not percolate");
    }

    #[test]
    fn test_percolation_dense() {
        let det = PercolationDetector::new(50);
        let ppp = HomogeneousPoisson::new(2.0, rect_window());
        let bm = BooleanModel::new(ppp, GrainDistribution::Constant(1.0));
        let mut rng = thread_rng();

        let mut percolates = false;
        for _ in 0..5 {
            let grains = bm.sample(&mut rng);
            let result = det.check(&grains, &rect_window());
            if result.percolates {
                percolates = true;
            }
        }
        assert!(percolates, "Dense model should percolate");
    }

    #[test]
    fn test_percolation_empty() {
        let det = PercolationDetector::new(50);
        let grains: Vec<(Point2D, f64)> = vec![];
        let result = det.check(&grains, &rect_window());
        assert!(!result.percolates);
        assert_eq!(result.num_components, 0);
    }

    #[test]
    fn test_percolation_single_component() {
        let det = PercolationDetector::new(50);
        // One big grain spanning the window
        let grains = vec![(Point2D::new(5.0, 5.0), 15.0)];
        let result = det.check(&grains, &rect_window());
        assert!(result.percolates);
        assert_eq!(result.num_components, 1);
    }

    #[test]
    fn test_largest_component_fraction_range() {
        let det = PercolationDetector::new(50);
        let ppp = HomogeneousPoisson::new(0.1, rect_window());
        let bm = BooleanModel::new(ppp, GrainDistribution::Constant(1.0));
        let mut rng = thread_rng();
        let grains = bm.sample(&mut rng);
        let result = det.check(&grains, &rect_window());
        assert!(result.largest_component_fraction >= 0.0);
        assert!(result.largest_component_fraction <= 1.0);
    }

    #[test]
    fn test_percolation_horizontal_only() {
        let det = PercolationDetector::new(50)
            .with_direction(PercolationDirection::Horizontal);
        let grains = vec![(Point2D::new(5.0, 5.0), 15.0)];
        let result = det.check(&grains, &rect_window());
        assert!(result.percolates);
    }

    #[test]
    fn test_percolation_vertical_only() {
        let det = PercolationDetector::new(50)
            .with_direction(PercolationDirection::Vertical);
        let grains = vec![(Point2D::new(5.0, 5.0), 15.0)];
        let result = det.check(&grains, &rect_window());
        assert!(result.percolates);
    }

    #[test]
    fn test_percolation_threshold_order() {
        // Higher intensity should have higher percolation probability
        let det = PercolationDetector::new(30);
        let mut rng = thread_rng();
        let w = Window::Rect { xmin: 0.0, xmax: 5.0, ymin: 0.0, ymax: 5.0 };

        let mut count_low = 0;
        let mut count_high = 0;
        let trials = 10;

        for _ in 0..trials {
            let ppp_low = HomogeneousPoisson::new(0.1, w.clone());
            let bm_low = BooleanModel::new(ppp_low, GrainDistribution::Constant(0.5));
            let result = det.check(&bm_low.sample(&mut rng), &w);
            if result.percolates { count_low += 1; }

            let ppp_high = HomogeneousPoisson::new(1.0, w.clone());
            let bm_high = BooleanModel::new(ppp_high, GrainDistribution::Constant(0.5));
            let result = det.check(&bm_high.sample(&mut rng), &w);
            if result.percolates { count_high += 1; }
        }

        assert!(count_high >= count_low, "Higher intensity should percolate at least as often");
    }
}
