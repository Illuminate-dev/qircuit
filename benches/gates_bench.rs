use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use qircuit::{Gate, QState};

fn bench_h_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("h_single");

    for &n in &[2, 4, 6, 8, 10] {
        let state = QState::new(n);
        group.bench_with_input(format!("{}q", n), &state, |b, s| {
            b.iter(|| {
                let mut s = s.clone();
                s.apply(black_box(&Gate::H(0)));
            });
        });
    }

    group.finish();
}

fn bench_cnot(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnot");

    for &n in &[2, 4, 6, 8, 10] {
        let state = QState::new(n);
        group.bench_with_input(format!("{}q", n), &state, |b, s| {
            b.iter(|| {
                let mut s = s.clone();
                s.apply(black_box(&Gate::CNOT(vec![0], 1)));
            });
        });
    }

    group.finish();
}

fn bench_hadamard_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard_layer");

    for &n in &[2, 4, 6, 8, 10] {
        let state = QState::new(n);
        let gates: Vec<Gate> = (0..n).map(Gate::H).collect();
        group.bench_with_input(format!("{}q", n), &state, |b, s| {
            b.iter(|| {
                let mut s = s.clone();
                for g in &gates {
                    s.apply(black_box(g));
                }
            });
        });
    }

    group.finish();
}

fn bench_combined_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_sequence");

    for &n in &[2, 4, 6] {
        let state = QState::new(n);
        let seq = (0..n)
            .flat_map(|i| vec![Gate::H(i), Gate::X(i)])
            .collect::<Vec<_>>();
        let combined = Gate::combined_gate(seq, n);

        group.bench_with_input(format!("{}q", n), &state, |b, s| {
            b.iter(|| {
                let mut s = s.clone();
                s.apply(black_box(&combined));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_h_single,
    bench_cnot,
    bench_hadamard_layer,
    bench_combined_sequence,
);
criterion_main!(benches);
