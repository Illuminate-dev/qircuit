use crate::{Gate, QState};

/// given an oracle and n qubits, determine if the function is balanced or constant
/// \x,y> -> |x,y + f(x)>
/// returns bool -> true = balanced
pub fn deutsch_jozsa(oracle: Gate, n: usize) -> bool {
    // output qubit has to start at 1 in order to start with the state |0> - |1> / sqrt(2)
    let mut state = QState::from_classical(1, n);

    // create an equal superposition of everything (H on input and on y)
    state.apply(Gate::combined_gate((0..n).map(|i| Gate::H(i)).collect(), n));

    // apply the oracle
    state.apply(oracle);

    // un-hadamard the input
    state.apply(Gate::combined_gate(
        (0..n - 1).map(|i| Gate::H(i)).collect(),
        n,
    ));

    // the last bit is in a superposition
    state.measure_all() >= 2
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_deutsch_jozsa_balanced() {
        let n = 3;
        let left_bit_is_1 = Gate::CNOT(vec![0], 2);
        assert!(deutsch_jozsa(left_bit_is_1, n));
    }

    #[test]
    fn test_deutsch_jozsa_constant() {
        let n = 3;
        assert!(!deutsch_jozsa(Gate::X(n - 1), n));
    }
}
