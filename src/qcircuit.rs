use crate::Gate;

/// Struct representing a quantum circuit
/// a quantum circuit is essentially a bunch of gates together, and evaluation at the end
pub struct QuantumCircuit {
    gates: Vec<Gate>,
}
