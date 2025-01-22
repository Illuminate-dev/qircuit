mod gates;
// mod math;
mod qstate;

use std::f64::consts::PI;

pub use gates::Gate;
pub use qstate::QState;

pub fn round_state(state: &mut QState) {
    state
        .state
        .iter_mut()
        .for_each(|x| x.re = (x.re * 100.0).round() / 100.0);
}

/// Quantum search algorithm for the number 2
/// because N is so small, it doesn't really work well. starting in the zero state guarnetees that the
pub fn qsearch() -> u8 {
    // qubits 0 and 1 are search qubits, 2 and 3 are the check qubits
    // start in all 0 state
    let n = 4;
    let n_search = 2;
    let mut state = QState::from_classical(0, n);

    // apply hadamard to all search qubits
    state.apply(Gate::combined_gate(vec![Gate::H(0), Gate::H(1)], n));

    // apply oracle
    // state.apply(Gate::combined_gate(
    //     vec![Gate::X(1), Gate::CNOT(vec![0, 1], 2), Gate::X(1)],
    //     n,
    // ));

    // optimal number of grover iterations
    let n_iterations = ((PI / 4.0) * (n_search as f64).sqrt()).floor() as usize;

    for _ in 0..n_iterations {
        // reflect over |s>
        state.apply(Gate::combined_gate(
            vec![
                Gate::X(1),
                // Gate::X(0),
                Gate::CNOT(vec![0, 1], 3),
                Gate::Z(3),
                Gate::CNOT(vec![0, 1], 3),
                Gate::X(1),
                // Gate::X(0),
            ],
            n,
        ));

        // println!("State after oracle: {:?}", state);

        // reflect over |E>
        let hadamard_input = Gate::combined_gate(vec![Gate::H(0), Gate::H(1)], n);
        let x_input = Gate::combined_gate(vec![Gate::X(0), Gate::X(1)], n);
        state.apply(Gate::combined_gate(
            vec![
                Gate::X(2),
                Gate::H(2),
                hadamard_input.clone(),
                x_input.clone(),
                Gate::CNOT(vec![0, 1], 2),
                x_input,
                hadamard_input,
            ],
            n,
        ));
    }

    // println!("State after reflection: {:?}", state);

    u8::from(state.measure(0)) * 2 + u8::from(state.measure(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_qsearch() {
        let count = &mut [0; 4];

        let n = 100;

        for _ in 0..n {
            let x = qsearch();
            count[x as usize] += 1;
        }

        assert_eq!(count[2], n);
    }
}
