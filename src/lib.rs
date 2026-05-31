#![deny(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::ptr_arg,
    clippy::excessive_precision,
)]
//! # lau-stochastic-geometry
//!
//! Stochastic geometry for agent uncertainty quantification.
//!
//! Provides tools for:
//! - Poisson point processes (homogeneous and inhomogeneous)
//! - Boolean model (random grains on point processes)
//! - Voronoi tessellation and Delaunay triangulation
//! - Ripley's K-function and L-function
//! - Minkowski functionals (Euler characteristic, perimeter, area)
//! - Percolation threshold detection
//! - Gaussian random fields with Matérn covariance
//! - Agent uncertainty as spatial point patterns and random sets

pub mod poisson;
pub mod boolean_model;
pub mod voronoi;
pub mod delaunay;
pub mod ripley;
pub mod minkowski;
pub mod percolation;
pub mod random_field;
pub mod agent_uncertainty;

pub use poisson::{HomogeneousPoisson, InhomogeneousPoisson};
pub use boolean_model::BooleanModel;
pub use voronoi::VoronoiTessellation;
pub use delaunay::DelaunayTriangulation;
pub use ripley::{RipleysK, RipleysL};
pub use minkowski::MinkowskiFunctionals;
pub use percolation::PercolationDetector;
pub use random_field::GaussianRandomField;
pub use agent_uncertainty::AgentUncertainty;
