use ndarray::{linalg::general_mat_vec_mul, Array1};
use num::complex::Complex64;
use rand::Rng;

use crate::gates::Gate;

/// Struct representing the quantum state of a system
pub struct QState {
    /// state of the qubits with length 2^n
    pub state: Array1<Complex64>,
    pub n: usize,
}

impl QState {
    pub fn new(n: usize) -> Self {
        let mut state = Array1::zeros(1 << n);
        state[0] = Complex64::new(1.0, 0.0);
        Self { state, n }
    }

    pub fn with_state(state: Array1<Complex64>) -> Self {
        assert!(state.len().is_power_of_two());
        let n = (state.len() as f64).log2() as usize;
        Self { state, n }
    }

    pub fn apply(&mut self, gate: Gate) {
        let mut out = Array1::zeros(self.state.len());
        general_mat_vec_mul(
            1.0.into(),
            &gate.to_matrix(self.n),
            &self.state,
            1.0.into(),
            &mut out,
        );
        self.state = out;
    }

    pub fn measure_all(&mut self) -> u8 {
        let mut rng = rand::thread_rng();
        let mut sum = 0.0;
        let r = rng.gen_range(0.0..1.0);
        for i in 0..self.state.len() {
            sum += self.state[i].norm_sqr();
            if r < sum {
                return i as u8;
            }
        }
        unreachable!();
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    pub fn test_qstate_new() {
        let qstate = QState::new(2);
        assert_eq!(qstate.state.len(), 4);
        let qstate = QState::new(4);
        assert_eq!(qstate.state.len(), 16);
    }

    #[test]
    pub fn test_measure_all() {
        // Test case 1: Initial state |00> should always measure 0
        let mut qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        let measurement = qstate.measure_all();
        assert_eq!(measurement, 0);

        // Test case 2: Initial state |11> should always measure 3
        let mut qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]));
        qstate.state[3] = Complex64::new(1.0, 0.0);
        let measurement = qstate.measure_all();
        assert_eq!(measurement, 3);

        // Test case 3: Superposition state |00> + |01> should measure 0 or 1 with 50% probability each
        let mut qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        let mut counts = [0, 0, 0, 0];
        for _ in 0..1000 {
            let measurement = qstate.measure_all();
            counts[measurement as usize] += 1;
        }
        assert!(counts[0] > 400);
        assert!(counts[0] < 600);
        assert!(counts[1] > 400);
        assert!(counts[1] < 600);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[3], 0);

        // Test case 4: Superposition state |10> + |11> should measure 2 or 3 with 50% probability each
        let mut qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]));
        let mut counts = [0, 0, 0, 0];
        for _ in 0..1000 {
            let measurement = qstate.measure_all();
            counts[measurement as usize] += 1;
        }
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 0);
        assert!(counts[2] > 400);
        assert!(counts[2] < 600);
        assert!(counts[3] > 400);
        assert!(counts[3] < 600);

        // Test case 5: All states equal prob
        let mut qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ]));
        let mut counts = [0, 0, 0, 0];
        for _ in 0..1000 {
            let measurement = qstate.measure_all();
            counts[measurement as usize] += 1;
        }

        for count in counts.iter() {
            assert!(*count > 150);
            assert!(*count < 350);
        }
    }
}
