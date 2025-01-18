use ndarray::{linalg::general_mat_vec_mul, Array1};
use num::complex::Complex64;

use crate::gates::Gate;

/// Struct representing the quantum state of a system
pub struct QState {
    /// state of the qubits with length 2^n
    pub state: Array1<Complex64>,
    pub n: usize,
}

impl QState {
    pub fn new(n: usize) -> Self {
        let state = Array1::zeros(1 << n);
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
}
