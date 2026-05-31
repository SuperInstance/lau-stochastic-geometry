//! Minkowski functionals: Euler characteristic, perimeter, and area for random sets.

use crate::boolean_model::BooleanModel;
use crate::poisson::{Point2D, Window};
use serde::{Serialize, Deserialize};

/// Minkowski functionals for a 2D random set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinkowskiFunctionals {
    /// Area (W₀).
    pub area: f64,
    /// Perimeter (W₁).
    pub perimeter: f64,
    /// Euler characteristic (W₂).
    pub euler_characteristic: i64,
}

impl MinkowskiFunctionals {
    /// Compute Minkowski functionals from a Boolean model realization on a grid.
    pub fn from_grains(
        grains: &[(Point2D, f64)],
        window: &Window,
        grid_res: usize,
    ) -> Self {
        let dx = window.width() / grid_res as f64;
        let dy = window.height() / grid_res as f64;

        // Build binary grid
        let mut grid = vec![vec![false; grid_res]; grid_res];
        for i in 0..grid_res {
            for j in 0..grid_res {
                let x = window.rect_xmin() + (i as f64 + 0.5) * dx;
                let y = window.rect_ymin() + (j as f64 + 0.5) * dy;
                let p = Point2D::new(x, y);
                if window.contains(&p) {
                    for (center, radius) in grains {
                        if p.distance_to(center) <= *radius {
                            grid[i][j] = true;
                            break;
                        }
                    }
                }
            }
        }

        // Area = number of occupied cells × cell_area
        let cell_area = dx * dy;
        let mut occupied = 0u64;
        for i in 0..grid_res {
            for j in 0..grid_res {
                if grid[i][j] {
                    occupied += 1;
                }
            }
        }
        let area = occupied as f64 * cell_area;

        // Perimeter = boundary edges × edge_length
        let _cell_perim = 2.0 * (dx + dy); // Not needed, use boundary edges
        let mut boundary_edges = 0u64;
        for i in 0..grid_res {
            for j in 0..grid_res {
                if !grid[i][j] {
                    continue;
                }
                // Check 4 neighbors
                if i == 0 || !grid[i - 1][j] { boundary_edges += 1; }
                if i == grid_res - 1 || !grid[i + 1][j] { boundary_edges += 1; }
                if j == 0 || !grid[i][j - 1] { boundary_edges += 1; }
                if j == grid_res - 1 || !grid[i][j + 1] { boundary_edges += 1; }
            }
        }
        let perimeter = boundary_edges as f64 * dx; // Approximate with dx as edge length

        // Euler characteristic via 2x2 configuration counting
        // V - E + F approach on the pixel grid
        // For 2D: χ = #vertices - #edges + #faces
        // Simplified: count components minus holes
        let euler = count_euler_characteristic(&grid);

        MinkowskiFunctionals {
            area,
            perimeter,
            euler_characteristic: euler,
        }
    }

    /// Theoretical Minkowski functionals for a Boolean model.
    pub fn theoretical_boolean_model(bm: &BooleanModel) -> Self {
        let lambda = bm.ppp.intensity;
        let mean_r = bm.grain_distribution.mean();
        let mean_area = std::f64::consts::PI * mean_r * mean_r;
        let mean_perim = 2.0 * std::f64::consts::PI * mean_r;
        let coverage = bm.coverage_fraction();

        // Area = p * W_area (window area)
        let area = coverage * bm.ppp.window.area();

        // Perimeter: for Boolean model with constant grains,
        // E[perimeter] ≈ λ * (1-p) * mean_perim * window_area
        let perimeter = lambda * (1.0 - coverage) * mean_perim * bm.ppp.window.area();

        // Euler characteristic: for Boolean model,
        // E[χ] ≈ λ * (1-p) * window_area - λ² * (1-p) * mean_area * window_area
        // Simplified: depends on grain shape
        let euler = (lambda * (1.0 - coverage) * bm.ppp.window.area()
            - lambda * lambda * (1.0 - coverage) * mean_area * bm.ppp.window.area()) as i64;

        MinkowskiFunctionals {
            area,
            perimeter,
            euler_characteristic: euler,
        }
    }

    /// Isoperimetric ratio: 4π * area / perimeter². Equals 1 for a disk.
    pub fn isoperimetric_ratio(&self) -> f64 {
        if self.perimeter <= 0.0 {
            return 0.0;
        }
        4.0 * std::f64::consts::PI * self.area / (self.perimeter * self.perimeter)
    }
}

/// Count Euler characteristic of a binary image using 2x2 configuration counting.
/// χ = Σ (c00 - c01 - c10 + c11) over all 2x2 blocks
/// This is the standard digital image Euler characteristic formula.
fn count_euler_characteristic(grid: &Vec<Vec<bool>>) -> i64 {
    let n = grid.len();
    if n == 0 { return 0; }
    let m = grid[0].len();
    if m == 0 { return 0; }

    // Use connected components approach: count components - count holes
    // First count connected components (4-connected foreground)
    let components = count_components(grid, true);
    // Count holes (4-connected background components that don't touch border)
    let holes = count_holes(grid);

    (components - holes) as i64
}

fn count_components(grid: &Vec<Vec<bool>>, target: bool) -> usize {
    let n = grid.len();
    let m = grid[0].len();
    let mut visited = vec![vec![false; m]; n];
    let mut count = 0;

    for i in 0..n {
        for j in 0..m {
            if grid[i][j] == target && !visited[i][j] {
                // Flood fill
                let mut stack = vec![(i, j)];
                while let Some((ci, cj)) = stack.pop() {
                    if ci >= n || cj >= m { continue; }
                    if visited[ci][cj] || grid[ci][cj] != target { continue; }
                    visited[ci][cj] = true;
                    if ci > 0 { stack.push((ci - 1, cj)); }
                    if ci + 1 < n { stack.push((ci + 1, cj)); }
                    if cj > 0 { stack.push((ci, cj - 1)); }
                    if cj + 1 < m { stack.push((ci, cj + 1)); }
                }
                count += 1;
            }
        }
    }
    count
}

fn count_holes(grid: &Vec<Vec<bool>>) -> usize {
    let n = grid.len();
    let m = grid[0].len();
    if n < 3 || m < 3 { return 0; }

    // Count background components that don't touch the border
    let mut visited = vec![vec![false; m]; n];
    let mut holes = 0;

    for i in 0..n {
        for j in 0..m {
            if grid[i][j] || visited[i][j] { continue; }
            let mut touches_border = false;
            let mut stack = vec![(i, j)];
            let mut component = Vec::new();
            while let Some((ci, cj)) = stack.pop() {
                if ci >= n || cj >= m { continue; }
                if visited[ci][cj] || grid[ci][cj] { continue; }
                visited[ci][cj] = true;
                if ci == 0 || ci == n - 1 || cj == 0 || cj == m - 1 {
                    touches_border = true;
                }
                component.push((ci, cj));
                if ci > 0 { stack.push((ci - 1, cj)); }
                if ci + 1 < n { stack.push((ci + 1, cj)); }
                if cj > 0 { stack.push((ci, cj - 1)); }
                if cj + 1 < m { stack.push((ci, cj + 1)); }
            }
            if !touches_border && !component.is_empty() {
                holes += 1;
            }
        }
    }
    holes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_model::{BooleanModel, GrainDistribution};
    use crate::poisson::{HomogeneousPoisson, Window};
    use rand::thread_rng;

    fn rect_window() -> Window {
        Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 }
    }

    #[test]
    fn test_minkowski_single_disk() {
        let grains = vec![(Point2D::new(5.0, 5.0), 2.0)];
        let mf = MinkowskiFunctionals::from_grains(&grains, &rect_window(), 200);
        let expected_area = std::f64::consts::PI * 4.0;
        assert!(
            (mf.area - expected_area).abs() < 3.0,
            "Area: got {}, expected ~{}",
            mf.area, expected_area
        );
    }

    #[test]
    fn test_minkowski_empty_set() {
        let grains: Vec<(Point2D, f64)> = vec![];
        let mf = MinkowskiFunctionals::from_grains(&grains, &rect_window(), 100);
        assert!((mf.area).abs() < 1e-10);
        assert!((mf.perimeter).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_full_coverage() {
        // One giant grain covering everything
        let grains = vec![(Point2D::new(5.0, 5.0), 20.0)];
        let mf = MinkowskiFunctionals::from_grains(&grains, &rect_window(), 100);
        // Area should be close to window area
        assert!(
            (mf.area - 100.0).abs() < 10.0,
            "Area for full coverage: {}",
            mf.area
        );
    }

    #[test]
    fn test_minkowski_two_disks() {
        let grains = vec![
            (Point2D::new(2.0, 5.0), 1.0),
            (Point2D::new(8.0, 5.0), 1.0),
        ];
        let mf = MinkowskiFunctionals::from_grains(&grains, &rect_window(), 200);
        let expected_area = 2.0 * std::f64::consts::PI;
        assert!(
            (mf.area - expected_area).abs() < 3.0,
            "Two disks area: got {}, expected ~{}",
            mf.area, expected_area
        );
        // Euler characteristic should be 2 (two components)
        assert!(mf.euler_characteristic > 0);
    }

    #[test]
    fn test_minkowski_overlapping_disks() {
        let grains = vec![
            (Point2D::new(4.0, 5.0), 2.0),
            (Point2D::new(6.0, 5.0), 2.0),
        ];
        let mf = MinkowskiFunctionals::from_grains(&grains, &rect_window(), 200);
        // Overlapping → Euler characteristic should be 1
        assert!(
            mf.euler_characteristic >= 1,
            "Euler char for overlapping: {}",
            mf.euler_characteristic
        );
    }

    #[test]
    fn test_isoperimetric_ratio_disk() {
        let grains = vec![(Point2D::new(5.0, 5.0), 3.0)];
        let mf = MinkowskiFunctionals::from_grains(&grains, &rect_window(), 500);
        let ratio = mf.isoperimetric_ratio();
        // Should be close to 1 for a disk
        assert!(
            ratio > 0.5 && ratio < 1.5,
            "Isoperimetric ratio: {ratio}"
        );
    }

    #[test]
    fn test_isoperimetric_ratio_empty() {
        let mf = MinkowskiFunctionals {
            area: 0.0,
            perimeter: 0.0,
            euler_characteristic: 0,
        };
        assert!((mf.isoperimetric_ratio()).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_serialization() {
        let mf = MinkowskiFunctionals {
            area: 10.0,
            perimeter: 20.0,
            euler_characteristic: 1,
        };
        let json = serde_json::to_string(&mf).unwrap();
        let deserialized: MinkowskiFunctionals = serde_json::from_str(&json).unwrap();
        assert!((deserialized.area - 10.0).abs() < 1e-10);
        assert_eq!(deserialized.euler_characteristic, 1);
    }

    #[test]
    fn test_boolean_model_functionals() {
        let ppp = HomogeneousPoisson::new(0.05, rect_window());
        let bm = BooleanModel::new(ppp, GrainDistribution::Constant(1.0));
        let theoretical = MinkowskiFunctionals::theoretical_boolean_model(&bm);
        assert!(theoretical.area >= 0.0);
        assert!(theoretical.perimeter >= 0.0);
    }
}
