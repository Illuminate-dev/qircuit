# Qircuit

Qircuit is a Rust library for quantum computing. It provides implementations of various quantum gates and quantum states, allowing for the simulation of quantum algorithms.

## Status

- Math library implementation might have been a bad idea - currently hard to implement powers of complex numbers. Temporary solution will be to use a scaled-down complex number struct with `f64` for each.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

To get started with Qircuit, simply add it to your `Cargo.toml`, or use the command `cargo add --git https://github.com/Illuminate-dev/qircuit`

## Usage

Here's how to use Qircuit:

1. Create a quantum state and apply quantum gates to it.
2. Simulate quantum algorithms using the provided gates and states.

Example:
```rust
use qircuit::{Gate, QState};

fn main() {
    let n = 4; // number of qubits
    let mut state = QState::from_classical(0, n);

    // Apply Hadamard gate to the first qubit
    state.apply(Gate::H(0));

    // Apply CNOT gate with control qubit 0 and target qubit 1
    state.apply(Gate::CNOT(vec![0], 1));

    // Measure the state
    let result = state.measure(0);
    println!("Measurement result: {}", result);
}
```

## Features

- **Quantum Gates**: Implementation of various quantum gates such as X, Y, Z, H (Hadamard), CNOT, and custom gates.
- **Quantum States**: Representation and manipulation of quantum states.
- **Quantum Algorithms**: Example implementation of quantum search algorithm.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to [list anyone or any resources you'd like to thank or give credit to].
- Contributor: [Illuminate-dev](https://github.com/Illuminate-dev)
