mod algorithms;
mod gates;
mod qstate;

pub use algorithms::deutsch_jozsa::deutsch_jozsa;
pub use algorithms::grovers::qsearch;
pub use gates::Gate;
pub use qstate::QState;

pub fn round_state(state: &mut QState) {
    state
        .state
        .iter_mut()
        .for_each(|x| x.re = (x.re * 100.0).round() / 100.0);
}
