# lau-stochastic-geometry

Stochastic geometry for agent uncertainty — Poisson processes, random tessellations, and spatial statistics.

## Features

- **Poisson point processes**: Homogeneous and inhomogeneous (with Lewis-Shedler thinning)
- **Boolean model**: Random grains (disks) on a Poisson process, with configurable grain distributions
- **Voronoi tessellation**: Sutherland-Hodgman clipping algorithm
- **Delaunay triangulation**: Bowyer-Watson incremental insertion
- **Ripley's K-function and L-function**: Spatial summary statistics with edge correction
- **Minkowski functionals**: Area, perimeter, and Euler characteristic for random sets
- **Percolation detection**: Connected component analysis and threshold detection
- **Gaussian random fields**: Matérn covariance (with Bessel function computation)
- **Agent uncertainty**: Model observations as spatial point patterns, confidence regions as random sets

## Usage

```rust
use lau_stochastic_geometry::*;

// Create a homogeneous Poisson point process
let window = poisson::Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
let ppp = HomogeneousPoisson::new(1.0, window.clone());
let mut rng = rand::thread_rng();
let points = ppp.sample(&mut rng);

// Compute Ripley's K-function
let k = RipleysK::new(points.clone(), window.clone());
let k_val = k.compute(1.0); // K(1.0)

// Build a Boolean model
let bm = BooleanModel::new(ppp, boolean_model::GrainDistribution::Constant(0.5));
let grains = bm.sample(&mut rng);
let coverage = bm.coverage_fraction();

// Agent uncertainty analysis
let au = AgentUncertainty::new(window);
let report = au.analyze(&observations, &mut rng, 50);
```

## License

MIT
