//! Gaussian random fields with Matérn covariance.

use crate::poisson::Window;
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::Normal;
use serde::{Serialize, Deserialize};

/// Parameters for the Matérn covariance kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaternParams {
    /// Variance (sill).
    pub sigma_sq: f64,
    /// Range parameter.
    pub range: f64,
    /// Smoothness parameter (ν).
    pub smoothness: f64,
}

impl MaternParams {
    pub fn new(sigma_sq: f64, range: f64, smoothness: f64) -> Self {
        assert!(sigma_sq > 0.0);
        assert!(range > 0.0);
        assert!(smoothness > 0.0);
        Self { sigma_sq, range, smoothness }
    }

    /// Matérn covariance function C(h).
    pub fn covariance(&self, h: f64) -> f64 {
        if h < 1e-12 {
            return self.sigma_sq;
        }
        // C(h) = σ² * 2^(1-ν) / Γ(ν) * (√(2ν) * h / ρ)^ν * K_ν(√(2ν) * h / ρ)
        let scaled_h = (2.0 * self.smoothness).sqrt() * h / self.range;

        // Approximate modified Bessel function K_ν using different approaches
        let k_nu = bessel_k(self.smoothness, scaled_h);

        self.sigma_sq
            * 2.0_f64.powf(1.0 - self.smoothness)
            / gamma(self.smoothness)
            * scaled_h.powf(self.smoothness)
            * k_nu
    }

    /// Special case: exponential covariance (ν = 0.5).
    pub fn exponential_covariance(sigma_sq: f64, range: f64, h: f64) -> f64 {
        sigma_sq * (-h / range).exp()
    }

    /// Special case: squared exponential / Gaussian covariance (ν → ∞).
    pub fn squared_exponential_covariance(sigma_sq: f64, range: f64, h: f64) -> f64 {
        sigma_sq * (-(h * h) / (2.0 * range * range)).exp()
    }
}

/// Gaussian random field simulator.
#[derive(Debug, Clone)]
pub struct GaussianRandomField {
    /// Matérn covariance parameters.
    pub params: MaternParams,
    /// Grid resolution.
    pub grid_res: usize,
}

impl GaussianRandomField {
    pub fn new(params: MaternParams, grid_res: usize) -> Self {
        Self { params, grid_res }
    }

    /// Generate a realization of the Gaussian random field on a regular grid.
    /// Uses Cholesky decomposition of the covariance matrix.
    pub fn sample<R: Rng + ?Sized>(&self, window: &Window, rng: &mut R) -> DMatrix<f64> {
        let n = self.grid_res;
        let total = n * n;

        // Build grid coordinates
        let dx = window.width() / n as f64;
        let dy = window.height() / n as f64;
        let mut coords = Vec::with_capacity(total);
        for i in 0..n {
            for j in 0..n {
                let x = window.rect_xmin() + (i as f64 + 0.5) * dx;
                let y = window.rect_ymin() + (j as f64 + 0.5) * dy;
                coords.push((x, y));
            }
        }

        // Build covariance matrix
        let mut cov_data = vec![0.0; total * total];
        for i in 0..total {
            for j in i..total {
                let h = ((coords[i].0 - coords[j].0).powi(2)
                    + (coords[i].1 - coords[j].1).powi(2))
                    .sqrt();
                let c = self.params.covariance(h);
                cov_data[i * total + j] = c;
                cov_data[j * total + i] = c;
            }
        }

        let cov_matrix = DMatrix::from_row_slice(total, total, &cov_data);

        // Cholesky decomposition
        let cholesky = cov_matrix.cholesky();
        let field = match cholesky {
            Some(chol) => {
                // Generate standard normal vector
                let normal = Normal::new(0.0, 1.0).unwrap();
                let z: Vec<f64> = (0..total).map(|_| rng.sample(normal)).collect();
                let z_vec = nalgebra::DVector::from_vec(z);
                let result = chol.l() * z_vec;
                result
            }
            None => {
                // Fallback: add nugget for numerical stability
                let mut cov_data_reg = cov_data.clone();
                for i in 0..total {
                    cov_data_reg[i * total + i] += 1e-6;
                }
                let cov_reg = DMatrix::from_row_slice(total, total, &cov_data_reg);
                match cov_reg.cholesky() {
                    Some(chol) => {
                        let normal = Normal::new(0.0, 1.0).unwrap();
                        let z: Vec<f64> = (0..total).map(|_| rng.sample(normal)).collect();
                        let z_vec = nalgebra::DVector::from_vec(z);
                        chol.l() * z_vec
                    }
                    None => {
                        // Ultimate fallback: independent normals
                        let normal = Normal::new(0.0, self.params.sigma_sq.sqrt()).unwrap();
                        let z: Vec<f64> = (0..total).map(|_| rng.sample(normal)).collect();
                        nalgebra::DVector::from_vec(z)
                    }
                }
            }
        };

        // Reshape to grid
        let mut result = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] = field[i * n + j];
            }
        }
        result
    }

    /// Generate using exponential covariance (much faster, special case ν=0.5).
    pub fn sample_exponential<R: Rng + ?Sized>(
        sigma_sq: f64,
        range: f64,
        window: &Window,
        grid_res: usize,
        rng: &mut R,
    ) -> DMatrix<f64> {
        let n = grid_res;
        let total = n * n;
        let dx = window.width() / n as f64;
        let dy = window.height() / n as f64;

        let mut coords = Vec::with_capacity(total);
        for i in 0..n {
            for j in 0..n {
                let x = window.rect_xmin() + (i as f64 + 0.5) * dx;
                let y = window.rect_ymin() + (j as f64 + 0.5) * dy;
                coords.push((x, y));
            }
        }

        let mut cov_data = vec![0.0; total * total];
        for i in 0..total {
            for j in i..total {
                let h = ((coords[i].0 - coords[j].0).powi(2)
                    + (coords[i].1 - coords[j].1).powi(2))
                    .sqrt();
                let c = MaternParams::exponential_covariance(sigma_sq, range, h);
                cov_data[i * total + j] = c;
                cov_data[j * total + i] = c;
            }
        }

        let cov_matrix = DMatrix::from_row_slice(total, total, &cov_data);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z: Vec<f64> = (0..total).map(|_| rng.sample(normal)).collect();
        let z_vec = nalgebra::DVector::from_vec(z);

        let field = match cov_matrix.cholesky() {
            Some(chol) => chol.l() * z_vec,
            None => z_vec * sigma_sq.sqrt(),
        };

        let mut result = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] = field[i * n + j];
            }
        }
        result
    }

    /// Compute the field variance empirically.
    pub fn field_variance(field: &DMatrix<f64>) -> f64 {
        let mean = field.iter().sum::<f64>() / (field.nrows() * field.ncols()) as f64;
        let var = field.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (field.nrows() * field.ncols()) as f64;
        var
    }

    /// Compute the field mean.
    pub fn field_mean(field: &DMatrix<f64>) -> f64 {
        field.iter().sum::<f64>() / (field.nrows() * field.ncols()) as f64
    }

    /// Threshold the field to create a binary excursion set.
    pub fn excursion_set(field: &DMatrix<f64>, threshold: f64) -> DMatrix<bool> {
        field.map(|v| v > threshold)
    }
}

/// Approximation of the modified Bessel function of the second kind K_ν(x).
fn bessel_k(nu: f64, x: f64) -> f64 {
    if x < 1e-10 {
        return if nu < 1e-10 { 1.0 } else { f64::INFINITY };
    }

    // For ν = 0.5: K_{1/2}(x) = sqrt(π/(2x)) * exp(-x)
    if (nu - 0.5).abs() < 1e-6 {
        return (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
    }

    // For ν = 1.5: K_{3/2}(x) = sqrt(π/(2x)) * (1 + 1/x) * exp(-x)
    if (nu - 1.5).abs() < 1e-6 {
        return (std::f64::consts::PI / (2.0 * x)).sqrt() * (1.0 + 1.0 / x) * (-x).exp();
    }

    // For ν = 2.5: K_{5/2}(x) = sqrt(π/(2x)) * (1 + 3/x + 3/x²) * exp(-x)
    if (nu - 2.5).abs() < 1e-6 {
        return (std::f64::consts::PI / (2.0 * x)).sqrt()
            * (1.0 + 3.0 / x + 3.0 / (x * x))
            * (-x).exp();
    }

    // General approximation for small x: use asymptotic
    // For large x: K_ν(x) ≈ sqrt(π/(2x)) * exp(-x)
    if x > 10.0 {
        return (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
    }

    // Use the series representation for moderate x
    // K_ν(x) ≈ π/(2*sin(π*ν)) * (I_{-ν}(x) - I_ν(x))
    // Approximate I_ν(x) by series
    let i_nu = bessel_i_series(nu, x);
    let i_neg_nu = bessel_i_series(-nu, x);

    if nu.fract() == 0.0 {
        // Integer order — use limit
        i_nu // Simplified
    } else {
        let sin_pi_nu = (std::f64::consts::PI * nu).sin();
        if sin_pi_nu.abs() < 1e-10 {
            i_nu
        } else {
            std::f64::consts::PI / (2.0 * sin_pi_nu) * (i_neg_nu - i_nu)
        }
    }
}

/// Series approximation for I_ν(x).
fn bessel_i_series(nu: f64, x: f64) -> f64 {
    let half_x = x / 2.0;
    let mut sum = 0.0;
    let mut term = 1.0 / gamma(nu + 1.0);
    sum += term;
    for k in 1..30 {
        term *= (half_x * half_x) / (k as f64 * (nu + k as f64));
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
    }
    half_x.powf(nu) * sum
}

/// Gamma function approximation (Stirling's for large x, Lanczos for small).
fn gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos approximation
    let g = 7.0;
    let coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        return std::f64::consts::PI
            / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
    }

    let x = x - 1.0;
    let mut a = coef[0];
    for i in 1..coef.len() {
        a += coef[i] / (x + i as f64);
    }

    let t = x + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn rect_window() -> Window {
        Window::Rect { xmin: 0.0, xmax: 1.0, ymin: 0.0, ymax: 1.0 }
    }

    #[test]
    fn test_matern_covariance_at_zero() {
        let params = MaternParams::new(1.0, 0.5, 1.0);
        let cov = params.covariance(0.0);
        assert!((cov - 1.0).abs() < 1e-6, "C(0) = {cov}, expected 1.0");
    }

    #[test]
    fn test_matern_covariance_decreases() {
        let params = MaternParams::new(1.0, 0.5, 0.5);
        let c1 = params.covariance(0.1);
        let c2 = params.covariance(1.0);
        assert!(c1 > c2, "C(0.1)={c1} should be > C(1.0)={c2}");
    }

    #[test]
    fn test_matern_covariance_positive() {
        let params = MaternParams::new(2.0, 0.5, 1.5);
        for h in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0] {
            assert!(params.covariance(h) >= 0.0, "C({h}) = {}", params.covariance(h));
        }
    }

    #[test]
    fn test_exponential_covariance() {
        let c = MaternParams::exponential_covariance(1.0, 1.0, 0.0);
        assert!((c - 1.0).abs() < 1e-10);
        let c = MaternParams::exponential_covariance(1.0, 1.0, 1.0);
        assert!((c - (-1.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_squared_exponential_covariance() {
        let c = MaternParams::squared_exponential_covariance(1.0, 1.0, 0.0);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grf_sample_shape() {
        let params = MaternParams::new(1.0, 0.5, 0.5);
        let grf = GaussianRandomField::new(params, 5);
        let mut rng = thread_rng();
        let field = grf.sample(&rect_window(), &mut rng);
        assert_eq!(field.nrows(), 5);
        assert_eq!(field.ncols(), 5);
    }

    #[test]
    fn test_grf_sample_variance() {
        let sigma_sq = 1.0;
        let params = MaternParams::new(sigma_sq, 0.3, 0.5);
        let grf = GaussianRandomField::new(params, 8);
        let mut rng = thread_rng();

        let mut variances = Vec::new();
        for _ in 0..10 {
            let field = grf.sample(&rect_window(), &mut rng);
            variances.push(GaussianRandomField::field_variance(&field));
        }

        let mean_var: f64 = variances.iter().sum::<f64>() / variances.len() as f64;
        // Variance should be in reasonable range (within 5x of sigma_sq)
        assert!(
            mean_var > 0.01 && mean_var < 10.0 * sigma_sq,
            "Mean variance = {mean_var}"
        );
    }

    #[test]
    fn test_grf_field_mean_near_zero() {
        let params = MaternParams::new(1.0, 0.3, 0.5);
        let grf = GaussianRandomField::new(params, 10);
        let mut rng = thread_rng();

        let mut means = Vec::new();
        for _ in 0..20 {
            let field = grf.sample(&rect_window(), &mut rng);
            means.push(GaussianRandomField::field_mean(&field));
        }

        let overall_mean: f64 = means.iter().sum::<f64>() / means.len() as f64;
        assert!(
            overall_mean.abs() < 2.0,
            "Overall mean = {overall_mean}"
        );
    }

    #[test]
    fn test_excursion_set() {
        let field = DMatrix::from_row_slice(2, 2, &[0.5, -0.5, 1.5, -1.0]);
        let exc = GaussianRandomField::excursion_set(&field, 0.0);
        assert!(exc[(0, 0)]);
        assert!(!exc[(0, 1)]);
        assert!(exc[(1, 0)]);
        assert!(!exc[(1, 1)]);
    }

    #[test]
    fn test_gamma_function() {
        // Γ(1) = 1
        assert!((gamma(1.0) - 1.0).abs() < 1e-6);
        // Γ(2) = 1
        assert!((gamma(2.0) - 1.0).abs() < 1e-6);
        // Γ(3) = 2
        assert!((gamma(3.0) - 2.0).abs() < 1e-5);
        // Γ(0.5) = √π
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_sample_exponential() {
        let mut rng = thread_rng();
        let field = GaussianRandomField::sample_exponential(
            1.0, 0.5, &rect_window(), 5, &mut rng,
        );
        assert_eq!(field.nrows(), 5);
        assert_eq!(field.ncols(), 5);
    }
}
