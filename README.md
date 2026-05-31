# lau-stochastic-geometry

Spatial randomness modeled rigorously: Poisson point processes, Voronoi tessellations, Delaunay triangulations, percolation thresholds, and Gaussian random fields with Matérn covariance.

Drop points randomly in the ocean, draw Voronoi cells, and you've mapped every hermit crab's territory.

## The math in 60 seconds

A **Poisson point process** places N points in region A with N ~ Poisson(λ|A|), each uniformly distributed. Build on this:

- **Boolean model:** grow random grains around each point — coverage fraction, connectivity
- **Voronoi tessellation:** partition space into cells nearest to each point
- **Delaunay triangulation:** dual of Voronoi — connect points whose cells share an edge
- **Ripley's K-function:** K(r) = λ⁻¹E[# points within r of a typical point] — detects clustering
- **Minkowski functionals:** area, perimeter, Euler characteristic of random sets
- **Percolation:** does an infinite connected cluster form? Critical intensity λ_c
- **Matérn fields:** Gaussian random fields with covariance C(r) = σ²2^(1-ν)/Γ(ν)(r/ℓ)^νK_ν(r/ℓ)

References: Chiu, Stoyan, Kendall & Mecke, *Stochastic Geometry and its Applications* (2013)

## Quick start

```rust
use lau_stochastic_geometry::{
    PoissonProcess, VoronoiTessellation, RipleysK, GaussianRandomField
};

// Homogeneous Poisson process in [0,1]² with intensity 50
let pp = PoissonProcess::homogeneous(50.0, 2);
let points = pp.sample();

// Voronoi tessellation
let voronoi = VoronoiTessellation::from_points(&points);
let cells = voronoi.cells();
let avg_area = cells.iter().map(|c| c.area()).sum::<f64>() / cells.len() as f64;

// Ripley's K-function (border-corrected)
let k = RipleysK::compute(&points, 0.5, 1.0);
// Under CSR: K(r) = πr²

// Gaussian random field with Matérn covariance
let field = GaussianRandomField::matern(1.0, 0.3, 0.5); // σ², ν, ℓ
let realization = field.sample_grid(32, 32);
```

## Key types

| Type | What it is |
|------|-----------|
| `PoissonProcess` | Homogeneous or inhomogeneous PPP with Lewis-Shedler thinning |
| `VoronoiTessellation` | Cells clipped to bounded domain, with area/perimeter |
| `DelaunayTriangulation` | Bowyer-Watson algorithm with circumcircle validation |
| `RipleysK` | Border-corrected spatial statistics K(r) and L(r) |
| `MinkowskiFunctionals` | Area, perimeter, Euler characteristic of random sets |
| `Percolation` | Connected component analysis, critical intensity search |
| `GaussianRandomField` | Matérn covariance with Cholesky simulation |

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-stochastic-geometry/issues) or PR. Natural extensions:

- Gibbs point processes (pairwise interaction)
- Shot-noise Cox processes
- Applications to sensor networks and spatial ML
