//! Delaunay triangulation from random points.

use crate::poisson::Point2D;
use serde::{Serialize, Deserialize};

/// A triangle defined by three vertex indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triangle {
    pub i: usize,
    pub j: usize,
    pub k: usize,
}

impl Triangle {
    /// Compute the circumcircle of the triangle defined by three points.
    pub fn circumcircle(&self, points: &[Point2D]) -> (Point2D, f64) {
        let a = &points[self.i];
        let b = &points[self.j];
        let c = &points[self.k];

        let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
        if d.abs() < 1e-12 {
            // Degenerate
            let cx = (a.x + b.x + c.x) / 3.0;
            let cy = (a.y + b.y + c.y) / 3.0;
            return (Point2D::new(cx, cy), 0.0);
        }

        let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
            + (b.x * b.x + b.y * b.y) * (c.y - a.y)
            + (c.x * c.x + c.y * c.y) * (a.y - b.y))
            / d;
        let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
            + (b.x * b.x + b.y * b.y) * (a.x - c.x)
            + (c.x * c.x + c.y * c.y) * (b.x - a.x))
            / d;

        let center = Point2D::new(ux, uy);
        let radius = center.distance_to(a);
        (center, radius)
    }

    /// Area of the triangle.
    pub fn area(&self, points: &[Point2D]) -> f64 {
        let a = &points[self.i];
        let b = &points[self.j];
        let c = &points[self.k];
        ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs() / 2.0
    }

    /// Perimeter of the triangle.
    pub fn perimeter(&self, points: &[Point2D]) -> f64 {
        let a = &points[self.i];
        let b = &points[self.j];
        let c = &points[self.k];
        a.distance_to(b) + b.distance_to(c) + c.distance_to(a)
    }

    /// Check if a point is inside the circumcircle.
    pub fn in_circumcircle(&self, points: &[Point2D], p: &Point2D) -> bool {
        let (center, radius) = self.circumcircle(points);
        center.distance_to(p) < radius
    }
}

/// Delaunay triangulation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelaunayTriangulation {
    /// Input points.
    pub points: Vec<Point2D>,
    /// Triangles.
    pub triangles: Vec<Triangle>,
}

impl DelaunayTriangulation {
    /// Compute Delaunay triangulation using the Bowyer-Watson algorithm.
    pub fn from_points(points: Vec<Point2D>) -> Self {
        if points.len() < 3 {
            return DelaunayTriangulation {
                points,
                triangles: Vec::new(),
            };
        }

        let _n = points.len();
        let mut pts = points;

        // Create super-triangle that contains all points
        let (xmin, xmax, ymin, ymax) = pts.iter().fold(
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
            |(xn, xx, yn, yx), p| {
                (xn.min(p.x), xx.max(p.x), yn.min(p.y), yx.max(p.y))
            },
        );
        let dx = xmax - xmin;
        let dy = ymax - ymin;
        let dmax = dx.max(dy);
        let cx = (xmin + xmax) / 2.0;
        let cy = (ymin + ymax) / 2.0;

        // Super-triangle vertices
        let p_super0 = Point2D::new(cx - 20.0 * dmax, cy - dmax);
        let p_super1 = Point2D::new(cx, cy + 20.0 * dmax);
        let p_super2 = Point2D::new(cx + 20.0 * dmax, cy - dmax);

        let n_orig = pts.len();
        pts.push(p_super0);
        pts.push(p_super1);
        pts.push(p_super2);

        let s0 = n_orig;
        let s1 = n_orig + 1;
        let s2 = n_orig + 2;

        // Start with super-triangle
        let mut triangles = vec![Triangle { i: s0, j: s1, k: s2 }];

        // Add points one by one
        for pi in 0..n_orig {
            let p = &pts[pi];

            // Find bad triangles (point inside circumcircle)
            let mut bad = Vec::new();
            for (ti, tri) in triangles.iter().enumerate() {
                if tri.in_circumcircle(&pts, p) {
                    bad.push(ti);
                }
            }

            // Find boundary polygon of bad triangles
            let mut polygon = Vec::new();
            for (idx, &ti) in bad.iter().enumerate() {
                let tri = &triangles[ti];
                let edges = [(tri.i, tri.j), (tri.j, tri.k), (tri.k, tri.i)];
                for (ea, eb) in edges {
                    let mut shared = false;
                    for (idx2, &tj) in bad.iter().enumerate() {
                        if idx == idx2 {
                            continue;
                        }
                        let tri2 = &triangles[tj];
                        let edges2 = [(tri2.i, tri2.j), (tri2.j, tri2.k), (tri2.k, tri2.i)];
                        for (e2a, e2b) in edges2 {
                            if (ea == e2a && eb == e2b) || (ea == e2b && eb == e2a) {
                                shared = true;
                                break;
                            }
                        }
                        if shared {
                            break;
                        }
                    }
                    if !shared {
                        polygon.push((ea, eb));
                    }
                }
            }

            // Remove bad triangles (reverse order)
            let mut bad_sorted = bad;
            bad_sorted.sort_unstable();
            for &ti in bad_sorted.iter().rev() {
                triangles.remove(ti);
            }

            // Create new triangles
            for (ea, eb) in polygon {
                triangles.push(Triangle { i: ea, j: eb, k: pi });
            }
        }

        // Remove triangles that share vertices with super-triangle
        triangles.retain(|t| {
            t.i < n_orig && t.j < n_orig && t.k < n_orig
        });

        // Remove super-triangle points
        pts.truncate(n_orig);

        DelaunayTriangulation { points: pts, triangles }
    }

    /// Total area of all triangles.
    pub fn total_area(&self) -> f64 {
        self.triangles.iter().map(|t| t.area(&self.points)).sum()
    }

    /// Mean triangle area.
    pub fn mean_triangle_area(&self) -> f64 {
        if self.triangles.is_empty() {
            return 0.0;
        }
        self.total_area() / self.triangles.len() as f64
    }

    /// Mean edge length.
    pub fn mean_edge_length(&self) -> f64 {
        let mut edges = std::collections::HashSet::new();
        let mut total = 0.0;
        let mut count = 0usize;

        for tri in &self.triangles {
            for (a, b) in [(tri.i, tri.j), (tri.j, tri.k), (tri.k, tri.i)] {
                let key = if a < b { (a, b) } else { (b, a) };
                if edges.insert(key) {
                    total += self.points[a].distance_to(&self.points[b]);
                    count += 1;
                }
            }
        }

        if count == 0 { 0.0 } else { total / count as f64 }
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        let mut edges = std::collections::HashSet::new();
        for tri in &self.triangles {
            for (a, b) in [(tri.i, tri.j), (tri.j, tri.k), (tri.k, tri.i)] {
                let key = if a < b { (a, b) } else { (b, a) };
                edges.insert(key);
            }
        }
        edges.len()
    }

    /// Check if the triangulation is valid (no point inside any circumcircle).
    pub fn is_delaunay(&self) -> bool {
        for tri in &self.triangles {
            let (center, radius) = tri.circumcircle(&self.points);
            for (pi, p) in self.points.iter().enumerate() {
                if pi == tri.i || pi == tri.j || pi == tri.k {
                    continue;
                }
                if center.distance_to(p) < radius - 1e-8 {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poisson::{HomogeneousPoisson, Window};
    use rand::thread_rng;

    #[test]
    fn test_delaunay_from_grid() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];
        let dt = DelaunayTriangulation::from_points(points);
        // A quad should triangulate to exactly 2 triangles
        assert_eq!(dt.triangles.len(), 2);
    }

    #[test]
    fn test_delaunay_triangle_area() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(0.0, 3.0),
        ];
        let dt = DelaunayTriangulation::from_points(points);
        assert_eq!(dt.triangles.len(), 1);
        let area = dt.triangles[0].area(&dt.points);
        assert!((area - 6.0).abs() < 1e-10, "Area was {area}");
    }

    #[test]
    fn test_delaunay_is_valid() {
        let w = Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
        let ppp = HomogeneousPoisson::new(0.3, w);
        let mut rng = thread_rng();
        let points = ppp.sample(&mut rng);
        if points.len() < 3 {
            return;
        }
        let dt = DelaunayTriangulation::from_points(points);
        assert!(dt.is_delaunay(), "Triangulation is not Delaunay valid");
    }

    #[test]
    fn test_delaunay_empty() {
        let dt = DelaunayTriangulation::from_points(vec![]);
        assert!(dt.triangles.is_empty());
    }

    #[test]
    fn test_delaunay_two_points() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0)];
        let dt = DelaunayTriangulation::from_points(points);
        assert!(dt.triangles.is_empty());
    }

    #[test]
    fn test_delaunay_from_poisson() {
        let w = Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
        let ppp = HomogeneousPoisson::new(0.5, w);
        let mut rng = thread_rng();
        let points = ppp.sample(&mut rng);
        if points.len() < 3 {
            return;
        }
        let dt = DelaunayTriangulation::from_points(points);
        // Euler formula: T ≤ 2n - 5 for convex hull
        let n = dt.points.len();
        assert!(dt.triangles.len() <= 2 * n - 2);
    }

    #[test]
    fn test_mean_edge_length_positive() {
        let w = Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
        let ppp = HomogeneousPoisson::new(1.0, w);
        let mut rng = thread_rng();
        let points = ppp.sample(&mut rng);
        if points.len() < 3 {
            return;
        }
        let dt = DelaunayTriangulation::from_points(points);
        if !dt.triangles.is_empty() {
            assert!(dt.mean_edge_length() > 0.0);
        }
    }

    #[test]
    fn test_circumcircle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(1.0, 1.0),
        ];
        let dt = DelaunayTriangulation::from_points(points);
        let (center, radius) = dt.triangles[0].circumcircle(&dt.points);
        assert!((center.x - 1.0).abs() < 1e-8, "cx={}", center.x);
        assert!(radius > 0.0);
    }

    #[test]
    fn test_triangle_perimeter() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(0.0, 4.0),
        ];
        let dt = DelaunayTriangulation::from_points(points);
        let perim = dt.triangles[0].perimeter(&dt.points);
        assert!((perim - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_delaunay_serialization() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let dt = DelaunayTriangulation::from_points(points);
        let json = serde_json::to_string(&dt).unwrap();
        let deserialized: DelaunayTriangulation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.triangles.len(), 1);
    }
}
