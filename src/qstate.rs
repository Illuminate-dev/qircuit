use ndarray::{linalg::general_mat_vec_mul, Array1};
use num::complex::Complex64;
use rand::Rng;

use crate::gates::Gate;

/// Struct representing the quantum state of a system
/// most significant bit is lefmost qubit - big endian
#[derive(Debug, Clone)]
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

    pub fn from_classical(state: u8, n: usize) -> Self {
        let mut qstate = Array1::zeros(1 << n);
        qstate[state as usize] = Complex64::new(1.0, 0.0);
        Self { state: qstate, n }
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

    pub fn measure(&mut self, i: usize) -> bool {
        let mut rng = rand::thread_rng();
        let mut prob_1 = 0.0;

        for j in 0..self.state.len() {
            if j % (1 << (self.n - i)) >= 1 << (self.n - i - 1) {
                prob_1 += self.state[j].norm_sqr();
            }
        }

        let out = rng.gen_bool(prob_1);
        let out_prob = if out { prob_1 } else { 1.0 - prob_1 };

        for (j, amp) in self.state.iter_mut().enumerate() {
            if j % (1 << (self.n - i)) >= 1 << (self.n - i - 1) {
                if out {
                    *amp /= out_prob.sqrt();
                } else {
                    *amp = Complex64::new(0.0, 0.0);
                }
            } else if !out {
                *amp /= out_prob.sqrt();
            } else {
                *amp = Complex64::new(0.0, 0.0);
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {

    use ndarray::arr1;

    use super::*;
    use crate::round_state;

    #[test]
    pub fn test_qstate_new() {
        let qstate = QState::new(2);
        assert_eq!(qstate.state.len(), 4);
        let qstate = QState::new(4);
        assert_eq!(qstate.state.len(), 16);
    }

    #[test]
    fn test_from_classical() {
        let qstate = QState::from_classical(0, 2);
        assert_eq!(
            qstate.state,
            arr1(&[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ])
        );

        let qstate = QState::from_classical(1, 2);
        assert_eq!(
            qstate.state,
            arr1(&[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0)
            ])
        );
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

        let mut qstate = QState::from_classical(0, 5);
        let measurement = qstate.measure_all();
        assert_eq!(measurement, 0);
    }

    #[test]
    fn test_measure() {
        // Test case 1: Measure qubit 0 in state |00> should always return false and the state remains |00>
        let mut qstate = QState::new(2);
        let result = qstate.measure(0);
        assert!(!result);
        assert_eq!(qstate.state[0], Complex64::new(1.0, 0.0));
        assert_eq!(qstate.state[1], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[2], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[3], Complex64::new(0.0, 0.0));

        // Test case 2: Measure qubit 1 in state |00> should always return false and the state remains |00>
        let mut qstate = QState::new(2);
        let result = qstate.measure(1);
        assert!(!result);
        assert_eq!(qstate.state[0], Complex64::new(1.0, 0.0));
        assert_eq!(qstate.state[1], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[2], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[3], Complex64::new(0.0, 0.0));

        // Test case 3: Measure qubit 0 in state |10> should always return true and the state becomes |10>
        let mut qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        let result = qstate.measure(0);
        assert!(result);
        assert_eq!(qstate.state[0], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[1], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[2], Complex64::new(1.0, 0.0));
        assert_eq!(qstate.state[3], Complex64::new(0.0, 0.0));

        // Test case 4: Measure qubit 1 in state |10> should always return false and the state remains |10>
        let mut qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        let result = qstate.measure(1);
        assert!(!result);
        assert_eq!(qstate.state[0], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[1], Complex64::new(0.0, 0.0));
        assert_eq!(qstate.state[2], Complex64::new(1.0, 0.0));
        assert_eq!(qstate.state[3], Complex64::new(0.0, 0.0));

        // Test case 5: Measure qubit 0 in superposition (|00> + |10>)/sqrt(2). Should return false with probability 0.5 and state becomes |00>, and true with 0.5 with state |10>
        let qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        let mut count_true = 0;
        let mut count_false = 0;
        for _ in 0..1000 {
            let mut qstate_copy = QState::with_state(qstate.state.clone());
            let result = qstate_copy.measure(0);
            round_state(&mut qstate_copy);
            if result {
                count_true += 1;
                assert_eq!(qstate_copy.state[0], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[1], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[2], Complex64::new(1.0, 0.0));
                assert_eq!(qstate_copy.state[3], Complex64::new(0.0, 0.0));
            } else {
                count_false += 1;
                assert_eq!(qstate_copy.state[0], Complex64::new(1.0, 0.0));
                assert_eq!(qstate_copy.state[1], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[2], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[3], Complex64::new(0.0, 0.0));
            }
        }

        assert!(count_true > 400);
        assert!(count_true < 600);
        assert!(count_false > 400);
        assert!(count_false < 600);

        // Test case 6: Measure qubit 1 in superposition (|00> + |01>)/sqrt(2). Should return false with probability 0.5 and state becomes |00>, and true with 0.5 with state |01>
        let qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        let mut count_true = 0;
        let mut count_false = 0;
        for _ in 0..1000 {
            let mut qstate_copy = QState::with_state(qstate.state.clone());
            let result = qstate_copy.measure(1);
            round_state(&mut qstate_copy);
            if result {
                count_true += 1;
                assert_eq!(qstate_copy.state[0], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[1], Complex64::new(1.0, 0.0));
                assert_eq!(qstate_copy.state[2], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[3], Complex64::new(0.0, 0.0));
            } else {
                count_false += 1;
                assert_eq!(qstate_copy.state[0], Complex64::new(1.0, 0.0));
                assert_eq!(qstate_copy.state[1], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[2], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[3], Complex64::new(0.0, 0.0));
            }
        }

        assert!(count_true > 400);
        assert!(count_true < 600);
        assert!(count_false > 400);
        assert!(count_false < 600);

        // Test case 7: Measure qubit 0 in superposition (|01> + |11>)/sqrt(2). Should return true with probability 0.5 and state becomes |01>, and false with 0.5 with state |11>
        let qstate = QState::with_state(Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]));
        let mut count_true = 0;
        let mut count_false = 0;
        for _ in 0..1000 {
            let mut qstate_copy = QState::with_state(qstate.state.clone());
            let result = qstate_copy.measure(0);
            round_state(&mut qstate_copy);
            if result {
                count_true += 1;
                println!("{:?}", qstate_copy.state);
                assert_eq!(qstate_copy.state[0], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[1], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[2], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[3], Complex64::new(1.0, 0.0));
            } else {
                count_false += 1;
                println!("{:?}", qstate_copy.state);
                assert_eq!(qstate_copy.state[0], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[1], Complex64::new(1.0, 0.0));
                assert_eq!(qstate_copy.state[2], Complex64::new(0.0, 0.0));
                assert_eq!(qstate_copy.state[3], Complex64::new(0.0, 0.0));
            }
        }
        assert!(count_true > 400);
        assert!(count_true < 600);
        assert!(count_false > 400);
        assert!(count_false < 600);
    }
}
