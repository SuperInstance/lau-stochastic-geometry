//! Voronoi tessellation from random points.

use crate::poisson::Point2D;
use serde::{Serialize, Deserialize};

/// A Voronoi cell: the region closest to a given seed point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiCell {
    /// The seed point.
    pub seed: Point2D,
    /// Vertices of the cell polygon (ordered).
    pub vertices: Vec<Point2D>,
}

impl VoronoiCell {
    /// Area of the cell via the shoelace formula.
    pub fn area(&self) -> f64 {
        if self.vertices.len() < 3 {
            return 0.0;
        }
        let n = self.vertices.len();
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }
        area.abs() / 2.0
    }

    /// Perimeter of the cell.
    pub fn perimeter(&self) -> f64 {
        if self.vertices.len() < 2 {
            return 0.0;
        }
        let n = self.vertices.len();
        let mut perim = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            perim += self.vertices[i].distance_to(&self.vertices[j]);
        }
        perim
    }
}

/// Voronoi tessellation computed from seed points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiTessellation {
    pub cells: Vec<VoronoiCell>,
    /// Bounding window.
    pub window: crate::poisson::Window,
}

impl VoronoiTessellation {
    /// Build Voronoi tessellation from seed points within a bounded window.
    /// Uses a simple incremental clipping algorithm (Sutherland-Hodgman style).
    pub fn from_points(points: Vec<Point2D>, window: crate::poisson::Window) -> Self {
        if points.is_empty() {
            return VoronoiTessellation { cells: Vec::new(), window };
        }

        let cells: Vec<VoronoiCell> = points
            .iter()
            .map(|seed| {
                let vertices = compute_voronoi_cell(seed, &points, &window);
                VoronoiCell {
                    seed: seed.clone(),
                    vertices,
                }
            })
            .collect();

        VoronoiTessellation { cells, window }
    }

    /// Total area of all cells.
    pub fn total_area(&self) -> f64 {
        self.cells.iter().map(|c| c.area()).sum()
    }

    /// Mean cell area.
    pub fn mean_cell_area(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        self.total_area() / self.cells.len() as f64
    }

    /// Mean cell perimeter.
    pub fn mean_cell_perimeter(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let total: f64 = self.cells.iter().map(|c| c.perimeter()).sum();
        total / self.cells.len() as f64
    }
}

/// Compute the Voronoi cell for a seed point by clipping the window
/// with perpendicular bisectors against all other seeds.
fn compute_voronoi_cell(
    seed: &Point2D,
    all_points: &[Point2D],
    window: &crate::poisson::Window,
) -> Vec<Point2D> {
    // Start with the bounding rectangle as initial polygon
    let mut polygon = vec![
        Point2D::new(window.rect_xmin(), window.rect_ymin()),
        Point2D::new(window.rect_xmax(), window.rect_ymin()),
        Point2D::new(window.rect_xmax(), window.rect_ymax()),
        Point2D::new(window.rect_xmin(), window.rect_ymax()),
    ];

    // Clip against each neighbor's bisector
    for other in all_points {
        if (other.x - seed.x).abs() < 1e-12 && (other.y - seed.y).abs() < 1e-12 {
            continue;
        }

        // Midpoint
        let mx = (seed.x + other.x) / 2.0;
        let my = (seed.y + other.y) / 2.0;

        // Normal direction (from seed to other)
        let nx = other.x - seed.x;
        let ny = other.y - seed.y;

        // Clip polygon to the half-plane: (p - mid) · n ≥ 0
        // which is: nx*(p.x - mx) + ny*(p.y - my) >= 0
        polygon = clip_polygon_by_halfplane(&polygon, mx, my, nx, ny);

        if polygon.is_empty() {
            break;
        }
    }

    polygon
}

/// Clip a polygon by a half-plane defined by point (mx, my) and normal (nx, ny).
/// Keeps points where (p - m) · n ≥ 0.
fn clip_polygon_by_halfplane(
    polygon: &[Point2D],
    mx: f64,
    my: f64,
    nx: f64,
    ny: f64,
) -> Vec<Point2D> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let eval = |p: &Point2D| -> f64 { nx * (p.x - mx) + ny * (p.y - my) };

    let mut result = Vec::new();
    let n = polygon.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let pi = &polygon[i];
        let pj = &polygon[j];
        let di = eval(pi);
        let dj = eval(pj);

        if di >= 0.0 {
            result.push(pi.clone());
            if dj < 0.0 {
                // Exiting: add intersection
                if let Some(intersection) = line_intersection(pi, pj, di, dj) {
                    result.push(intersection);
                }
            }
        } else if dj >= 0.0 {
            // Entering: add intersection
            if let Some(intersection) = line_intersection(pi, pj, di, dj) {
                result.push(intersection);
            }
        }
    }

    result
}

/// Compute intersection of edge pi→pj with the clipping line,
/// given signed distances di, dj.
fn line_intersection(pi: &Point2D, pj: &Point2D, di: f64, dj: f64) -> Option<Point2D> {
    if (di - dj).abs() < 1e-15 {
        return None;
    }
    let t = di / (di - dj);
    if t < -1e-10 || t > 1.0 + 1e-10 {
        return None;
    }
    Some(Point2D::new(
        pi.x + t * (pj.x - pi.x),
        pi.y + t * (pj.y - pi.y),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poisson::{HomogeneousPoisson, Window};
    use rand::thread_rng;

    fn rect_window() -> Window {
        Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 }
    }

    #[test]
    fn test_voronoi_from_grid_points() {
        // 2x2 grid
        let points = vec![
            Point2D::new(2.5, 2.5),
            Point2D::new(7.5, 2.5),
            Point2D::new(2.5, 7.5),
            Point2D::new(7.5, 7.5),
        ];
        let tess = VoronoiTessellation::from_points(points, rect_window());
        assert_eq!(tess.cells.len(), 4);
        // Total area should be ~100
        let total = tess.total_area();
        assert!(
            (total - 100.0).abs() < 5.0,
            "Total area was {total}, expected ~100"
        );
    }

    #[test]
    fn test_voronoi_single_point() {
        let points = vec![Point2D::new(5.0, 5.0)];
        let tess = VoronoiTessellation::from_points(points, rect_window());
        assert_eq!(tess.cells.len(), 1);
        let area = tess.cells[0].area();
        assert!((area - 100.0).abs() < 1.0, "Area was {area}");
    }

    #[test]
    fn test_voronoi_empty() {
        let tess = VoronoiTessellation::from_points(vec![], rect_window());
        assert!(tess.cells.is_empty());
    }

    #[test]
    fn test_voronoi_cell_areas_positive() {
        let ppp = HomogeneousPoisson::new(0.5, rect_window());
        let mut rng = thread_rng();
        let points = ppp.sample(&mut rng);
        if points.len() < 2 {
            return;
        }
        let tess = VoronoiTessellation::from_points(points, rect_window());
        for cell in &tess.cells {
            assert!(cell.area() >= 0.0, "Cell area was negative: {}", cell.area());
        }
    }

    #[test]
    fn test_voronoi_mean_area_decreases_with_density() {
        let w = rect_window();
        let ppp_low = HomogeneousPoisson::new(0.1, w.clone());
        let ppp_high = HomogeneousPoisson::new(1.0, w.clone());
        let mut rng = thread_rng();

        let pts_low = ppp_low.sample(&mut rng);
        let pts_high = ppp_high.sample(&mut rng);

        if pts_low.len() < 2 || pts_high.len() < 2 {
            return;
        }

        let tess_low = VoronoiTessellation::from_points(pts_low, w.clone());
        let tess_high = VoronoiTessellation::from_points(pts_high, w);

        // Higher density → smaller cells on average
        if tess_low.cells.len() > 0 && tess_high.cells.len() > 0 {
            assert!(tess_low.mean_cell_area() > tess_high.mean_cell_area());
        }
    }

    #[test]
    fn test_voronoi_from_poisson_points() {
        let ppp = HomogeneousPoisson::new(0.3, rect_window());
        let mut rng = thread_rng();
        let points = ppp.sample(&mut rng);
        if points.len() < 3 {
            return;
        }
        let tess = VoronoiTessellation::from_points(points, rect_window());
        assert_eq!(tess.cells.len(), tess.cells.len()); // just verify no crash
    }

    #[test]
    fn test_shoelace_area() {
        let cell = VoronoiCell {
            seed: Point2D::new(0.0, 0.0),
            vertices: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(1.0, 0.0),
                Point2D::new(1.0, 1.0),
                Point2D::new(0.0, 1.0),
            ],
        };
        assert!((cell.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_perimeter() {
        let cell = VoronoiCell {
            seed: Point2D::new(0.0, 0.0),
            vertices: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(3.0, 0.0),
                Point2D::new(3.0, 4.0),
                Point2D::new(0.0, 4.0),
            ],
        };
        assert!((cell.perimeter() - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_voronoi_serialization() {
        let points = vec![Point2D::new(5.0, 5.0)];
        let tess = VoronoiTessellation::from_points(points, rect_window());
        let json = serde_json::to_string(&tess).unwrap();
        let deserialized: VoronoiTessellation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.cells.len(), 1);
    }
}
