use core::f64;
use std::f64::consts::SQRT_2;

use ndarray::{arr2, linalg::kron, Array2};
use num::complex::Complex64;

const ONE_SQRT_2: f64 = 1.0 / SQRT_2;

const X_MATRIX: [[f64; 2]; 2] = [[0.0, 1.0], [1.0, 0.0]];
const Y_MATRIX: [[Complex64; 2]; 2] = [
    [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
    [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
];
const Z_MATRIX: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, -1.0]];

const H_MATRIX: [[f64; 2]; 2] = [[ONE_SQRT_2, ONE_SQRT_2], [ONE_SQRT_2, -ONE_SQRT_2]];
const I_MATRIX: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
const PROJECTION_1_MATRIX: [[f64; 2]; 2] = [[0.0, 0.0], [0.0, 1.0]];

/// A quantum gate
pub enum Gate {
    /// Pauli X gate on the ith qubit
    X(usize),
    /// Pauli Y gate on the ith qubit
    Y(usize),
    /// Pauli Z gate on the ith qubit
    Z(usize),
    /// H gate on the ith qubit
    H(usize),
    /// CNOT gate with a vec of controls and a target qubit.
    CNOT(Vec<usize>, usize),
    /// I gate is same anywhere
    I,
    /// Classical or gate on quibit .0 and .1 with target qubit .2
    OR(usize, usize, usize),
    /// Custom gate with the following matrix
    Custom(Array2<Complex64>),
}

impl Gate {
    pub fn to_matrix(&self, n: usize) -> Array2<Complex64> {
        match self {
            Gate::X(i) => Self::build_matrix(*i, n, &X_MATRIX),
            Gate::Y(i) => Self::build_matrix_complex(*i, n, &Y_MATRIX),
            Gate::Z(i) => Self::build_matrix(*i, n, &Z_MATRIX),
            Gate::H(i) => Self::build_matrix(*i, n, &H_MATRIX),
            Gate::CNOT(controls, target) => {
                let proj_1_all = Self::build_matrix_many(controls, n, &PROJECTION_1_MATRIX);
                let i_all = Gate::I.to_matrix(n);
                let proj_1_inverse = &i_all - &proj_1_all;
                let x_all = Gate::X(*target).to_matrix(n);

                &proj_1_all.dot(&x_all) + &proj_1_inverse.dot(&i_all)
            }
            Gate::I => Self::build_matrix(0, n, &I_MATRIX),
            Gate::OR(a, b, i) => {
                match Self::combined_gate(
                    vec![
                        Gate::X(*a),
                        Gate::X(*b),
                        Gate::CNOT(vec![*a, *b], *i),
                        Gate::X(*i),
                        Gate::X(*a),
                        Gate::X(*b),
                    ],
                    n,
                ) {
                    Gate::Custom(m) => m,
                    _ => unreachable!(),
                }
            }
            Gate::Custom(m) => m.clone(),
        }
    }

    fn build_matrix(i: usize, n: usize, gate: &[[f64; 2]; 2]) -> Array2<Complex64> {
        let mut m = if i == 0 {
            // X matrix
            arr2(gate).map(Complex64::from)
        } else {
            arr2(&I_MATRIX).map(Complex64::from)
        };
        for j in 1..n {
            if i == j {
                m = kron(&m, &arr2(gate).map(Complex64::from));
            } else {
                m = kron(&m, &arr2(&I_MATRIX).map(Complex64::from));
            }
        }
        m
    }

    fn build_matrix_complex(i: usize, n: usize, gate: &[[Complex64; 2]; 2]) -> Array2<Complex64> {
        let mut m = if i == 0 {
            // X matrix
            arr2(gate)
        } else {
            arr2(&I_MATRIX).map(Complex64::from)
        };
        for j in 1..n {
            if i == j {
                m = kron(&m, &arr2(gate));
            } else {
                m = kron(&m, &arr2(&I_MATRIX).map(Complex64::from));
            }
        }
        m
    }

    fn build_matrix_many(i: &[usize], n: usize, gate: &[[f64; 2]; 2]) -> Array2<Complex64> {
        let mut m = if i.contains(&0) {
            // X matrix
            arr2(gate).map(Complex64::from)
        } else {
            arr2(&I_MATRIX).map(Complex64::from)
        };
        for j in 1..n {
            if i.contains(&j) {
                m = kron(&m, &arr2(gate).map(Complex64::from));
            } else {
                m = kron(&m, &arr2(&I_MATRIX).map(Complex64::from));
            }
        }
        m
    }

    /// contructs a combined gate from a list of gates, applying matrix multiplication in order
    /// from left to right. This means that the first gate in the list is the first to be applied
    pub fn combined_gate(gates: Vec<Gate>, n: usize) -> Gate {
        if gates.is_empty() {
            return Gate::I;
        }

        // apply the gates in reverse order
        let mut m = gates[gates.len() - 1].to_matrix(n);
        for gate in gates.iter().rev().skip(1) {
            m = m.dot(&gate.to_matrix(n));
        }
        Self::Custom(m)
    }
}

#[cfg(test)]
pub mod tests {

    use crate::round_state;
    use ndarray::arr1;
    use num::{complex::Complex64, Float};

    use crate::qstate::QState;

    use super::*;

    #[test]
    fn test_create_y() {
        assert_eq!(
            Gate::Y(0).to_matrix(2),
            arr2(&[
                [
                    Complex64::from(0.0),
                    Complex64::from(0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::from(0.0)
                ],
                [
                    Complex64::from(0.0),
                    Complex64::from(0.0),
                    Complex64::from(0.0),
                    Complex64::new(0.0, -1.0)
                ],
                [
                    Complex64::new(0.0, 1.0),
                    Complex64::from(0.0),
                    Complex64::from(0.0),
                    Complex64::from(0.0)
                ],
                [
                    Complex64::from(0.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::from(0.0),
                    Complex64::from(0.0)
                ]
            ])
        );
    }

    #[test]
    fn test_apply_y() {
        // |0> to i|1>
        let mut state =
            QState::with_state(arr1(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]));
        let gate = Gate::Y(0);
        state.apply(gate);
        assert_eq!(
            state.state,
            arr1(&[Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)])
        );

        //  |1> to -i|0>
        let mut state =
            QState::with_state(arr1(&[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]));
        let gate = Gate::Y(0);
        state.apply(gate);
        assert_eq!(
            state.state,
            arr1(&[Complex64::new(0.0, -1.0), Complex64::new(0.0, 0.0)])
        );

        // 1/sqrt(2) * |01> + 1/sqrt(2) * |10> to -i/sqrt(2) * |00> + i/sqrt(2) * |11>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::Y(1));
        round_state(&mut state);
        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, -1.0 / 2.0.sqrt()),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0 / 2.0.sqrt()),
            ])
        );
    }

    #[test]
    fn test_create_z() {
        assert_eq!(
            Gate::Z(0).to_matrix(2),
            arr2(&[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0]
            ])
            .map(Complex64::from)
        );
    }

    #[test]
    fn test_apply_z() {
        // |0> to |0>
        let mut state =
            QState::with_state(arr1(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]));
        let gate = Gate::Z(0);
        state.apply(gate);
        assert_eq!(
            state.state,
            arr1(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
        );

        // |1> to -|1>
        let mut state =
            QState::with_state(arr1(&[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]));
        let gate = Gate::Z(0);
        state.apply(gate);
        assert_eq!(
            state.state,
            arr1(&[Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)])
        );

        // 1/sqrt(2) * |01> + 1/sqrt(2) * |10> to -1/sqrt(2) * |01> + 1/sqrt(2) * |10>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::Z(1));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0 / 2.0.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0.sqrt(), 0.0),
                Complex64::new(0.0, 0.0),
            ])
        );
    }

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
            .map(Complex64::from)
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
                [ONE_SQRT_2, 0.0, ONE_SQRT_2, 0.0],
                [0.0, ONE_SQRT_2, 0.0, ONE_SQRT_2],
                [ONE_SQRT_2, 0.0, -ONE_SQRT_2, 0.0],
                [0.0, ONE_SQRT_2, 0.0, -ONE_SQRT_2]
            ])
            .map(Complex64::from)
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
                Complex64::new(ONE_SQRT_2, 0.0),
                Complex64::new(ONE_SQRT_2, 0.0)
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
        round_state(&mut state);

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

    #[test]
    fn test_apply_cnot() {
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]));

        state.apply(Gate::CNOT(vec![0], 1));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        );

        state.apply(Gate::CNOT(vec![1], 0));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        );

        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::CNOT(vec![0, 1], 2));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ])
        );

        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::CNOT(vec![0, 1], 2));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        );
    }
    #[test]
    fn test_cnot_creation() {
        let m = Gate::CNOT(vec![0], 1).to_matrix(2);
        assert_eq!(
            m,
            arr2(&[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0]
            ])
            .map(Complex64::from)
        );
    }

    #[test]
    fn test_combined_creation() {
        let m = Gate::combined_gate(vec![Gate::X(0), Gate::H(1)], 2).to_matrix(2);
        assert_eq!(
            m,
            arr2(&[
                [0.0, 0.0, ONE_SQRT_2, ONE_SQRT_2],
                [0.0, 0.0, ONE_SQRT_2, -ONE_SQRT_2],
                [ONE_SQRT_2, ONE_SQRT_2, 0.0, 0.0],
                [ONE_SQRT_2, -ONE_SQRT_2, 0.0, 0.0]
            ])
            .map(Complex64::from)
        );

        let m = Gate::combined_gate(vec![Gate::H(0), Gate::CNOT(vec![0], 1)], 2).to_matrix(2);

        assert_eq!(
            m,
            arr2(&[
                [ONE_SQRT_2, 0.0, ONE_SQRT_2, 0.0],
                [0.0, ONE_SQRT_2, 0.0, ONE_SQRT_2],
                [0.0, ONE_SQRT_2, 0.0, -ONE_SQRT_2],
                [ONE_SQRT_2, 0.0, -ONE_SQRT_2, 0.0]
            ])
            .map(Complex64::from)
        );
    }

    #[test]
    fn test_combined() {
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]));

        let m = Gate::combined_gate(vec![Gate::X(0), Gate::H(1)], 2);

        state.apply(m);

        assert_eq!(
            state.state,
            &arr1(&[
                Complex64::new(ONE_SQRT_2, 0.0),
                Complex64::new(-ONE_SQRT_2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ])
        );

        let mut state = QState::with_state(arr1(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        let m = Gate::combined_gate(vec![Gate::H(0), Gate::CNOT(vec![0], 1)], 2);

        state.apply(m);

        assert_eq!(
            state.state,
            &arr1(&[
                Complex64::new(ONE_SQRT_2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(ONE_SQRT_2, 0.0)
            ])
        );

        let mut state = QState::with_state(arr1(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::combined_gate(
            vec![Gate::H(0), Gate::X(1), Gate::H(1), Gate::CNOT(vec![0], 1)],
            2,
        ));

        round_state(&mut state);

        assert_eq!(
            state.state,
            &arr1(&[
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0)
            ])
        );
    }

    #[test]
    fn test_apply_or() {
        // |000> -> |000>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::OR(0, 1, 2));
        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        );

        // |010> -> |011>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::OR(0, 1, 2));
        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        );
        // |100> -> |101>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::OR(0, 1, 2));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        );
        // |110> -> |111>
        let mut state = QState::with_state(arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        state.apply(Gate::OR(0, 1, 2));

        assert_eq!(
            state.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ])
        );
    }

    #[test]
    fn test_or_creation() {
        let m = Gate::OR(0, 1, 2).to_matrix(3);
        assert_eq!(
            m,
            arr2(&[
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ])
            .map(Complex64::from)
        );
    }
}
