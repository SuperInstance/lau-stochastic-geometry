# lau-stochastic-geometry

**Stochastic geometry for agent uncertainty quantification — Poisson processes, random tessellations, Boolean models, spatial statistics, and percolation.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

93 tests · 3,125 lines of Rust · 9 modules

---

## What This Does

When agents operate in physical or abstract space, their positions, observations, and uncertainties form **spatial point patterns** and **random sets**. This crate applies the tools of stochastic geometry to model, analyze, and reason about these patterns.

You get:
- **Poisson point processes** (homogeneous and inhomogeneous) for modeling agent locations
- **Boolean models** — random grains on Poisson processes for coverage and confidence regions
- **Voronoi tessellation** — partition space into agent territories
- **Delaunay triangulation** — dual graph connecting nearby agents
- **Ripley's K-function and L-function** — test for clustering vs. regularity in agent distributions
- **Minkowski functionals** — area, perimeter, and Euler characteristic of random sets
- **Percolation detection** — does the agent network form a connected cluster across the domain?
- **Gaussian random fields** with Matérn covariance — spatially correlated uncertainty
- **Agent uncertainty quantification** — tie everything together for real agent observation data

---

## Key Idea

Stochastic geometry provides the probabilistic foundation for reasoning about random spatial structures. In the agent context:

| Stochastic Geometry | Agent Application |
|---|---|
| Poisson point process | Agent positions / observations |
| Boolean model | Confidence regions / coverage areas |
| Voronoi cell | Agent territory / responsibility zone |
| Delaunay edge | Communication link / neighbor relation |
| Ripley's K-function | Clustered vs. dispersed agents |
| Minkowski functionals | Shape analysis of agent coverage |
| Percolation | Network connectivity |
| Gaussian random field | Spatially correlated uncertainty |
| Coverage fraction | Fraction of space monitored |

The crate implements the full pipeline: generate point patterns → build tessellations → compute summary statistics → detect connectivity → quantify uncertainty.

---

## Install

```toml
[dependencies]
lau-stochastic-geometry = "0.1"
```

Requires Rust 2021 edition. Dependencies: `nalgebra`, `rand`, `rand_distr`, `serde`, `statrs`.

---

## Quick Start

```rust
use lau_stochastic_geometry::*;
use rand::thread_rng;

// 1. Create a homogeneous Poisson point process
let window = poisson::Window::Rect { xmin: 0.0, xmax: 10.0, ymin: 0.0, ymax: 10.0 };
let ppp = HomogeneousPoisson::new(2.0, window.clone()); // intensity λ = 2
let points = ppp.sample(&mut thread_rng());
println!("Generated {} agent positions", points.len()); // ~200 points

// 2. Build Voronoi tessellation (agent territories)
let voronoi = VoronoiTessellation::from_points(points.clone(), window.clone());
for cell in &voronoi.cells {
    println!("Cell area: {:.2}", cell.area());
}

// 3. Build Delaunay triangulation (neighbor graph)
let delaunay = DelaunayTriangulation::from_points(points.clone());
println!("{} triangles, {} edges", delaunay.triangles.len(), delaunay.edges().len());

// 4. Boolean model (confidence regions)
let boolean = BooleanModel::new(ppp.clone(), boolean_model::GrainDistribution::Constant(0.5));
let grains = boolean.sample(&mut thread_rng());
let coverage = boolean.coverage_fraction();

// 5. Ripley's K-function (spatial pattern analysis)
let ripleys = ripley::RipleysK::new(points.clone(), window.clone());
let k_curve = ripleys.compute_curve(&[0.5, 1.0, 2.0, 3.0]);

// 6. Percolation check (network connectivity)
let perc = percolation::PercolationDetector::new(100);
let result = perc.check(&grains, &window);
println!("Percolates: {}, coverage: {:.1}%", result.percolates, result.parameter * 100.0);
```

---

## API Reference

### `HomogeneousPoisson` / `InhomogeneousPoisson`
Poisson point processes on 2D windows.

```rust
// Homogeneous: constant intensity λ
let ppp = HomogeneousPoisson::new(lambda, window);
let points = ppp.sample(&mut rng);
let n_expected = ppp.expected_count(); // λ × area

// Inhomogeneous: intensity function λ(x,y)
let ipp = InhomogeneousPoisson::new(|x, y| 2.0 + (x * 0.1).sin(), window);
let points = ipp.sample(&mut rng);
```

Supports rectangular and circular windows.

### `BooleanModel`
Random grains (disks) centered on a Poisson process.

```rust
let bm = BooleanModel::new(ppp, GrainDistribution::Uniform { min: 0.2, max: 1.0 });
let grains = bm.sample(&mut rng);      // Vec<(Point2D, radius)>
let coverage = bm.coverage_fraction();  // 1 - exp(-λ · E[πR²])
let is_covered = bm.contains(&grains, &point);
```

Grain distributions: `Constant(r)`, `Uniform { min, max }`, `Exponential { mean }`.

### `VoronoiTessellation`
Partitions space into cells closest to each seed point.

```rust
let vt = VoronoiTessellation::from_points(points, window);
for cell in &vt.cells {
    let area = cell.area();           // Shoelace formula
    let perim = cell.perimeter();     // Sum of edge lengths
    let n_verts = cell.vertices.len();
}
```

Uses Sutherland-Hodgman clipping against bisector half-planes.

### `DelaunayTriangulation`
Dual of the Voronoi — connects nearby points into triangles.

```rust
let dt = DelaunayTriangulation::from_points(points);
for tri in &dt.triangles {
    let (center, radius) = tri.circumcircle(&points);
    let area = tri.area(&points);
    let inside = tri.in_circumcircle(&points, &test_point);
}
let edges = dt.edges();              // Unique edges
let neighbors = dt.neighbors(0);     // Nodes connected to node 0
```

### `RipleysK` / `RipleysL`
Spatial summary statistics for point pattern analysis.

```rust
let k = RipleysK::new(points.clone(), window);
let k_val = k.compute(2.0);                    // K(2.0)
let k_curve = k.compute_curve(&[0.5, 1.0, 2.0]); // K(r) for multiple r

let l = RipleysL::new(points, window);
let l_val = l.compute(2.0);                    // L(2.0) = √(K(2.0)/π) - 2.0
```

- **K(r) > πr²**: agents are **clustered** at scale r
- **K(r) = πr²**: agents are **random** (CSR — complete spatial randomness)
- **K(r) < πr²**: agents are **regular/dispersed**

### `MinkowskiFunctionals`
Shape descriptors for random sets: area (W₀), perimeter (W₁), Euler characteristic (W₂).

```rust
let mf = MinkowskiFunctionals::from_grains(&grains, &window, 200);
println!("Area: {}, Perimeter: {}, χ: {}", mf.area, mf.perimeter, mf.euler_characteristic);
```

The Euler characteristic counts connected components minus holes.

### `PercolationDetector`
Checks whether a random set forms a spanning cluster.

```rust
let detector = PercolationDetector::new(100)
    .with_direction(percolation::PercolationDirection::Both);
let result = detector.check(&grains, &window);
// result.percolates: does it span the domain?
// result.largest_component_fraction: size of biggest cluster
// result.num_components: total connected components
```

Uses flood-fill (BFS) on a discretized grid. Supports horizontal, vertical, or both directions.

### `GaussianRandomField`
Spatially correlated random fields with Matérn covariance.

```rust
let params = MaternParams::new(1.0, 2.0, 1.5); // σ²=1, range=2, ν=1.5
let grf = GaussianRandomField::new(params, 50);
let field = grf.sample(&mut rng);         // 50×50 grid of correlated values
let cov = params.covariance(1.5);          // C(1.5) with Matérn kernel
```

The **Matérn kernel** interpolates between:
- **ν = 0.5**: Exponential covariance C(h) = σ²·exp(−h/ρ) — rough fields
- **ν = 1.5, 2.5**: Medium smoothness — most practical
- **ν → ∞**: Squared exponential C(h) = σ²·exp(−h²/2ρ²) — infinitely smooth

### `AgentUncertainty`
Ties everything together for agent observation data.

```rust
let au = AgentUncertainty::new(window);
let observations = vec![
    AgentObservation::new(1.0, 2.0, 0.5, 0.95),
    AgentObservation::new(3.0, 4.0, 0.3, 0.99),
    // ... more observations
];

// Fit Poisson intensity
let intensity = au.fit_poisson_intensity(&observations);

// Build confidence region
let report = au.analyze(&observations, &mut rng);
// report.intensity, report.l_function, report.minkowski, report.coverage_fraction, report.percolates
```

---

## How It Works

The crate is structured as a layered pipeline:

```
Layer 1: Point Process        HomogeneousPoisson, InhomogeneousPoisson
              │
              ▼
Layer 2: Random Structures     BooleanModel, VoronoiTessellation, DelaunayTriangulation
              │
              ▼
Layer 3: Summary Statistics    RipleysK, RipleysL, MinkowskiFunctionals
              │
              ▼
Layer 4: Connectivity          PercolationDetector
              │
              ▼
Layer 5: Spatial Correlation   GaussianRandomField (Matérn covariance)
              │
              ▼
Layer 6: Agent Application     AgentUncertainty
```

**Layer 1** generates random point patterns. A Poisson process with intensity λ produces N ~ Poisson(λ·|W|) points uniformly in window W.

**Layer 2** builds spatial structures from points: Boolean models (grain coverage), Voronoi diagrams (territory partition), and Delaunay triangulations (neighbor graphs).

**Layer 3** computes summary statistics that characterize the spatial pattern. Ripley's K-function detects clustering. Minkowski functionals quantify shape.

**Layer 4** checks percolation — whether the occupied set spans the domain. This determines if the agent network is connected.

**Layer 5** models spatially correlated fields. The Matérn covariance family provides flexible smoothness control.

**Layer 6** applies all tools to agent observation data, producing a comprehensive uncertainty report.

---

## The Math

### Poisson Point Processes

A **homogeneous Poisson point process** with intensity λ on window W:
1. Total count: N ~ Poisson(λ|W|)
2. Given N = n, points are i.i.d. uniform on W
3. Independent scattering: counts in disjoint regions are independent

An **inhomogeneous** Poisson process replaces constant λ with a function λ(x,y). The expected count is ∫_W λ(x,y) dA.

### Boolean Model

The **Boolean model** Ξ = ⋃ᵢ B(xᵢ, Rᵢ) where {xᵢ} is a Poisson process and {Rᵢ} are i.i.d. radii.

Key properties:
- **Coverage fraction**: p = 1 − exp(−λ · E[πR²])
- **Covariance**: C(r) = P(0 ∈ Ξ and r ∈ Ξ) = (2p − p²) + p²·exp(−λ · E[area(B₀ ∩ B_r)])
- **Contact distribution**: distance from a fixed point to Ξ

### Voronoi and Delaunay

The **Voronoi diagram** partitions ℝ² into cells Vᵢ = {x : d(x, pᵢ) ≤ d(x, pⱼ) ∀j}. Each agent "owns" the region closest to it.

The **Delaunay triangulation** is the dual: two points are connected iff their Voronoi cells share an edge. It satisfies the **empty circumcircle property**: no point lies inside the circumcircle of any triangle.

### Ripley's K-function

**Ripley's K-function** K(r) = (1/λ) · E[number of points within distance r of a typical point].

For complete spatial randomness (CSR): K_CSR(r) = πr².

Deviations reveal spatial structure:
- K(r) > πr² → **clustering** (points are closer than expected)
- K(r) < πr² → **regularity** (points are more spread than expected)

The **L-function** L(r) = √(K(r)/π) − r linearizes this: L(r) > 0 = clustering, L(r) < 0 = regularity.

### Minkowski Functionals

In 2D, the four **intrinsic volumes** (additive functionals) are:

| Functional | Formula | Meaning |
|---|---|---|
| W₀ = Area | ∫ dA | Total covered area |
| W₁ = Perimeter | ∫ dL | Boundary length |
| W₂ = Euler characteristic | χ | Components minus holes |
| W₃ = Connectivity | (not in 2D) | — |

These satisfy the **Hadwiger theorem**: any motion-invariant, additive functional on 2D sets is a linear combination of these three.

### Matérn Covariance

The **Matérn covariance** family:
```
C(h) = σ² · 2^(1−ν) / Γ(ν) · (√(2ν)·h/ρ)^ν · K_ν(√(2ν)·h/ρ)
```

where K_ν is the modified Bessel function of the second kind.

- **σ²**: variance (sill)
- **ρ**: range parameter (correlation length)
- **ν**: smoothness (controls differentiability of sample paths)

Sample paths are ⌊ν⌋ times differentiable. This makes ν the key parameter: ν = 0.5 gives rough (exponential) fields, ν → ∞ gives smooth (Gaussian) fields.

### Percolation

A Boolean model **percolates** when it contains a connected component that spans the entire domain. The **critical coverage fraction** p_c ≈ 0.676 for 2D Boolean models with fixed-radius grains (continuous percolation).

Below p_c: disconnected clusters. Above p_c: a giant component connects the domain. This is the geometric analog of the Erdős–Rényi phase transition.

---

## Module Overview

| Module | Tests | Key Types | Purpose |
|--------|-------|-----------|---------|
| `poisson` | 14 | `HomogeneousPoisson`, `InhomogeneousPoisson`, `Point2D`, `Window` | Point process generation |
| `boolean_model` | 11 | `BooleanModel`, `GrainDistribution` | Random grain coverage |
| `voronoi` | 9 | `VoronoiTessellation`, `VoronoiCell` | Territory partition |
| `delaunay` | 10 | `DelaunayTriangulation`, `Triangle` | Neighbor graph |
| `ripley` | 10 | `RipleysK`, `RipleysL` | Spatial pattern analysis |
| `minkowski` | 9 | `MinkowskiFunctionals` | Shape descriptors |
| `percolation` | 8 | `PercolationDetector`, `PercolationResult` | Connectivity |
| `random_field` | 11 | `GaussianRandomField`, `MaternParams` | Spatial correlation |
| `agent_uncertainty` | 11 | `AgentUncertainty`, `AgentObservation`, `UncertaintyReport` | Agent application |

---

## References

- **Stochastic Geometry**: Chiu, Stoyan, Kendall & Mecke, *Stochastic Geometry and its Applications* (2013)
- **Boolean Models**: Schneider & Weil, *Stochastic and Integral Geometry* (2008)
- **Ripley's K**: Ripley, *Spatial Statistics* (1981)
- **Matérn Covariance**: Matérn, *Spatial Variation* (1960)
- **Percolation**: Meester & Roy, *Continuum Percolation* (1996)

---

## License

MIT
