mod gates;
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

/// Given an *n_search* bit number *num*, this builds a search oracle
fn build_simple_oracle_gates(num: u8, n_search: usize) -> Vec<Gate> {
    let mut gates = Vec::new();

    for i in 0..n_search {
        if (num & (1 << (n_search - 1 - i))) == 0 {
            gates.push(Gate::X(i));
        };
    }

    gates.push(Gate::CNOT(
        (0..n_search).collect::<Vec<usize>>(),
        n_search + 1,
    ));

    gates.push(Gate::Z(n_search + 1));

    gates.push(Gate::CNOT(
        (0..n_search).collect::<Vec<usize>>(),
        n_search + 1,
    ));

    for i in (0..n_search).rev() {
        if (num & (1 << (n_search - 1 - i))) == 0 {
            gates.push(Gate::X(i));
        };
    }

    gates
}

fn build_diffusion_gate(n: usize) -> Gate {
    let hadamard_input = Gate::combined_gate((0..n - 2).map(|i| Gate::H(i)).collect(), n);
    let x_input = Gate::combined_gate((0..n - 2).map(|i| Gate::X(i)).collect(), n);

    Gate::combined_gate(
        vec![
            Gate::X(n - 2),
            Gate::H(n - 2),
            hadamard_input.clone(),
            x_input.clone(),
            Gate::CNOT((0..n - 2).collect::<Vec<usize>>(), n - 2),
            x_input,
            hadamard_input,
            Gate::H(n - 2),
            Gate::X(n - 2),
        ],
        n,
    )
}

/// Quantum search algorithm for the n_search bit number num
///     n_search: number of qubits to use for searching
pub fn qsearch(num: u8, n_search: usize) -> u8 {
    let n = n_search + 2;
    let mut state = QState::from_classical(0, n);

    // apply hadamard to all search qubits
    for i in 0..n_search {
        state.apply(Gate::H(i));
    }

    // optimal number of grover iterations: ⌊π/4 · √(2^n_search)⌋
    let n_iterations = ((PI / 4.0) * (2_usize.pow(n_search as u32) as f64).sqrt()).floor() as usize;
    let n_iterations = n_iterations.max(1);

    let oracle_gate = Gate::combined_gate(build_simple_oracle_gates(num, n_search), n);
    let diffusion_gate = build_diffusion_gate(n);

    for _ in 0..n_iterations {
        state.apply(oracle_gate.clone());
        state.apply(diffusion_gate.clone());
    }

    (0..n_search)
        .map(|i| u8::from(state.measure(n_search - 1 - i)) * (2 as u8).pow(i as u32))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qsearch_2_bits() {
        const N_BITS: u32 = 2;

        for i in 0..((2_usize).pow(N_BITS)) {
            let count = &mut [0; (2_usize).pow(N_BITS)];

            let n = 100;

            for _ in 0..n {
                let x = qsearch(i as u8, N_BITS as usize);
                count[x as usize] += 1;
            }

            assert!(
                count[i] >= n - 5,
                "target={i}: got {} correct (expected at least {})",
                count[i],
                n - 5
            );
        }
    }

    #[test]
    fn test_qsearch_3_bits() {
        const N_BITS: u32 = 3;

        for i in 0..((2_usize).pow(N_BITS)) {
            let count = &mut [0; (2_usize).pow(N_BITS)];

            let n = 100;

            for _ in 0..n {
                let x = qsearch(i as u8, N_BITS as usize);
                count[x as usize] += 1;
            }

            assert!(
                count[i] >= n - 15,
                "target={i}: got {} correct (expected at least {})",
                count[i],
                n - 15
            );
        }
    }
}
