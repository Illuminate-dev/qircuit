use core::f64;
use std::f64::consts::SQRT_2;

use ndarray::{arr2, linalg::kron, Array2};

const X_MATRIX: [[f64; 2]; 2] = [[0.0, 1.0], [1.0, 0.0]];
const H_MATRIX: [[f64; 2]; 2] = [[1.0 / SQRT_2, 1.0 / SQRT_2], [1.0 / SQRT_2, -1.0 / SQRT_2]];
const I_MATRIX: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

pub enum Gate {
    X(usize),
    H(usize),
}

impl Gate {
    pub fn to_matrix(&self, n: usize) -> Array2<f64> {
        match self {
            Gate::X(i) => Self::build_matrix(*i, n, &X_MATRIX),
            Gate::H(i) => Self::build_matrix(*i, n, &H_MATRIX),
        }
    }

    fn build_matrix(i: usize, n: usize, gate: &[[f64; 2]; 2]) -> Array2<f64> {
        let mut m = if i == 0 {
            // X matrix
            arr2(gate)
        } else {
            arr2(&I_MATRIX)
        };
        for j in 1..n {
            if i == j {
                m = kron(&m, &arr2(gate));
            } else {
                m = kron(&m, &arr2(&I_MATRIX));
            }
        }
        m
    }
}

#[cfg(test)]
pub mod tests {

    use ndarray::arr1;
    use num::{complex::Complex64, Float};

    use crate::qstate::QState;

    use super::*;

    #[test]
    fn test_create_x() {
        assert_eq!(
            Gate::X(0).to_matrix(2),
            arr2(&[
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])
        );
    }

    #[test]
    fn test_apply_x() {
        // 0 to 1
        let mut state =
            QState::with_state(arr1(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]));
        let gate = Gate::X(0);
        state.apply(gate);
        assert_eq!(
            state.state,
            arr1(&[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)])
        );

        // 1/sqrt(2) * |01> + 1/sqrt(2) * |10> to 1/sqrt(2) * |00> + 1/sqrt(2) * |11>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::X(1));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(1.0 / 2.0.sqrt(), 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            ])
        );
    }

    #[test]
    fn test_create_hadamard() {
        assert_eq!(
            Gate::H(0).to_matrix(2),
            arr2(&[
                [1.0 / SQRT_2, 0.0, 1.0 / SQRT_2, 0.0],
                [0.0, 1.0 / SQRT_2, 0.0, 1.0 / SQRT_2],
                [1.0 / SQRT_2, 0.0, -1.0 / SQRT_2, 0.0],
                [0.0, 1.0 / SQRT_2, 0.0, -1.0 / SQRT_2]
            ])
        );
    }

    #[test]
    fn test_apply_hadamard() {
        // 0 to 1 / sqrt(2) * (|0> + |1>)
        let mut state =
            QState::with_state(arr1(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]));
        let gate = Gate::H(0);
        state.apply(gate);
        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(1.0 / SQRT_2, 0.0),
                Complex64::new(1.0 / SQRT_2, 0.0)
            ])
        );

        // 1/sqrt(2) * |01> + 1/sqrt(2) * |10> to 1/sqrt(2) * |00> + 1/sqrt(2) * |11>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::H(1));
        state
            .state
            .iter_mut()
            .for_each(|x| x.re = (x.re * 100.0).round() / 100.0);

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(1.0 / 2.0, 0.0),
                Complex64::new(-1.0 / 2.0, 0.0),
                Complex64::new(1.0 / 2.0, 0.0),
                Complex64::new(1.0 / 2.0, 0.0),
            ])
        );
    }
}
