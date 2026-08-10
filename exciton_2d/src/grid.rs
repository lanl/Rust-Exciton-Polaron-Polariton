//! Grid module for spatial discretization
//!
//! This module provides a 2D grid implementation for spatial calculations,
//! with support for Cartesian and polar coordinates. The grid is centered at
//! the origin and provides methods to access coordinates and derived quantities
//! like radial distance and angle.
//!
//! The implementation is optimized for memory efficiency, with large grids
//! processed in blocks to reduce peak memory usage.

use log::info;
use ndarray::{Array1, Array2};
use std::sync::{Arc, RwLock};

/// A 2D spatial grid with support for Cartesian and polar coordinates
///
/// This struct represents a 2D spatial grid centered at the origin. It provides
/// methods to access the grid points in Cartesian coordinates (x, y) and derived
/// quantities in polar coordinates (r, θ).
///
/// For memory efficiency, some arrays are computed on-demand for large grids,
/// and the computation is done in blocks to reduce peak memory usage.
#[derive(Clone)]
pub struct Grid {
    pub grid_size: usize,
    pub box_size: f64,
    pub verbose: bool,
    pub dx: f64,
    pub x: Array1<f64>,
    pub y: Array1<f64>,
    x_grid: Arc<RwLock<Option<Array2<f64>>>>,
    y_grid: Arc<RwLock<Option<Array2<f64>>>>,
    r: Arc<RwLock<Option<Array2<f64>>>>,
    theta: Arc<RwLock<Option<Array2<f64>>>>,
}

impl Grid {
    /// Creates a new grid with the specified size and dimensions
    ///
    /// # Arguments
    /// * `grid_size` - Number of points in each dimension
    /// * `box_size` - Physical size of the box in each dimension
    /// * `verbose` - Whether to print information during grid setup
    pub fn new(grid_size: usize, box_size: f64, verbose: bool) -> Self {
        let mut grid = Self {
            grid_size,
            box_size,
            verbose,
            dx: box_size / grid_size as f64,
            x: Array1::zeros(grid_size),
            y: Array1::zeros(grid_size),
            x_grid: Arc::new(RwLock::new(None)),
            y_grid: Arc::new(RwLock::new(None)),
            r: Arc::new(RwLock::new(None)),
            theta: Arc::new(RwLock::new(None)),
        };

        grid.setup_grid();
        grid
    }

    /// Sets up the grid points and precomputes coordinate arrays
    ///
    /// For large grids, some arrays are computed on-demand to save memory.
    fn setup_grid(&mut self) {
        // Grid spacing already calculated in constructor

        // Grid points (shifted to be centered at zero)
        let half_box = self.box_size / 2.0;
        let half_dx = self.dx / 2.0;

        for i in 0..self.grid_size {
            let pos = -half_box + half_dx + i as f64 * self.dx;
            self.x[i] = pos;
            self.y[i] = pos;
        }

        // For very large grids, avoid creating the full meshgrid in memory
        // Only precompute meshgrids for smaller grids to save memory
        if self.grid_size <= 128 {
            let mut x_grid = Array2::zeros((self.grid_size, self.grid_size));
            let mut y_grid = Array2::zeros((self.grid_size, self.grid_size));

            for i in 0..self.grid_size {
                for j in 0..self.grid_size {
                    x_grid[[i, j]] = self.x[j];
                    y_grid[[i, j]] = self.y[i];
                }
            }

            *self.x_grid.write().unwrap() = Some(x_grid);
            *self.y_grid.write().unwrap() = Some(y_grid);

            // Calculate R
            let x_grid = self.x_grid.read().unwrap();
            let y_grid = self.y_grid.read().unwrap();

            if let (Some(x_grid), Some(y_grid)) = (x_grid.as_ref(), y_grid.as_ref()) {
                let mut r = Array2::zeros((self.grid_size, self.grid_size));

                for i in 0..self.grid_size {
                    for j in 0..self.grid_size {
                        r[[i, j]] = (x_grid[[i, j]].powi(2) + y_grid[[i, j]].powi(2)).sqrt();
                    }
                }

                *self.r.write().unwrap() = Some(r);
            }
        }

        if self.verbose {
            info!(
                "Grid setup complete: {}x{} points, box size = {}",
                self.grid_size, self.grid_size, self.box_size
            );
        }
    }

    /// Gets the radial distance matrix (r = √(x² + y²))
    ///
    /// This method returns a 2D array containing the radial distance from the origin
    /// to each point on the grid. For large grids, this is computed on-demand in blocks
    /// to save memory.
    ///
    /// # Returns
    /// A 2D array of radial distances
    pub fn get_r(&self) -> Array2<f64> {
        if let Some(r) = self.r.read().unwrap().as_ref() {
            return r.clone();
        }

        // Compute R on demand for large grids
        if self.verbose {
            info!("Computing radial distance matrix in blocks to save memory...");
        }

        // Process in blocks to reduce peak memory usage
        let n = self.grid_size;
        let block_size = std::cmp::min(128, n);
        let mut r = Array2::zeros((n, n));

        for i in (0..n).step_by(block_size) {
            let i_end = std::cmp::min(i + block_size, n);
            for j in (0..n).step_by(block_size) {
                let j_end = std::cmp::min(j + block_size, n);

                // Create a small block of the meshgrid
                let mut x_block = Array2::zeros((i_end - i, j_end - j));
                let mut y_block = Array2::zeros((i_end - i, j_end - j));

                for ii in i..i_end {
                    for jj in j..j_end {
                        x_block[[ii - i, jj - j]] = self.x[jj];
                        y_block[[ii - i, jj - j]] = self.y[ii];
                    }
                }

                // Compute R for this block
                for ii in i..i_end {
                    for jj in j..j_end {
                        let x = x_block[[ii - i, jj - j]];
                        let y = y_block[[ii - i, jj - j]];
                        r[[ii, jj]] = (x * x + y * y).sqrt();
                    }
                }
            }
        }

        *self.r.write().unwrap() = Some(r.clone());
        r
    }

    /// Gets the angular coordinate matrix (θ = atan2(y, x))
    ///
    /// This method returns a 2D array containing the angular coordinate (polar angle)
    /// of each point on the grid. For large grids, this is computed on-demand in blocks
    /// to save memory.
    ///
    /// # Returns
    /// A 2D array of angular coordinates in radians
    pub fn get_theta(&self) -> Array2<f64> {
        if let Some(theta) = self.theta.read().unwrap().as_ref() {
            return theta.clone();
        }

        if let (Some(x_grid), Some(y_grid)) = (
            self.x_grid.read().unwrap().as_ref(),
            self.y_grid.read().unwrap().as_ref(),
        ) {
            let mut theta = Array2::zeros((self.grid_size, self.grid_size));

            for i in 0..self.grid_size {
                for j in 0..self.grid_size {
                    theta[[i, j]] = y_grid[[i, j]].atan2(x_grid[[i, j]]);
                }
            }

            *self.theta.write().unwrap() = Some(theta.clone());
            return theta;
        }

        // Create Theta in blocks to save memory
        if self.verbose {
            info!("Computing angular coordinate matrix in blocks to save memory...");
        }

        let n = self.grid_size;
        let mut theta = Array2::zeros((n, n));
        let block_size = std::cmp::min(128, n);

        for i in (0..n).step_by(block_size) {
            let i_end = std::cmp::min(i + block_size, n);
            for j in (0..n).step_by(block_size) {
                let j_end = std::cmp::min(j + block_size, n);

                // Create a small block of the meshgrid
                let mut x_block = Array2::zeros((i_end - i, j_end - j));
                let mut y_block = Array2::zeros((i_end - i, j_end - j));

                for ii in i..i_end {
                    for jj in j..j_end {
                        x_block[[ii - i, jj - j]] = self.x[jj];
                        y_block[[ii - i, jj - j]] = self.y[ii];
                    }
                }

                // Compute Theta for this block
                for ii in i..i_end {
                    for jj in j..j_end {
                        let x = x_block[[ii - i, jj - j]];
                        let y = y_block[[ii - i, jj - j]];
                        theta[[ii, jj]] = y.atan2(x);
                    }
                }
            }
        }

        *self.theta.write().unwrap() = Some(theta.clone());
        theta
    }
}
