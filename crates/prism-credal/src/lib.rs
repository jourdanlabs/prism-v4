//! PRISM bytecode validation and exact credal inference.
mod ir;
mod model;
mod policy;
mod simplex;

pub use ir::{
    Decl, Link, Observation, ParseError, Polarity, Probability, Program, Query, QueryOp, Reason,
    SentenceTag, Span, parse_bytecode,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "UPPERCASE")]
pub enum Verdict {
    Answer {
        value: String,
        target: String,
        via: &'static str,
        interval: Value,
        credal: Value,
        witnesses: Vec<Decl>,
    },
    Refused {
        code: &'static str,
        details: Value,
    },
}

impl Verdict {
    pub(crate) fn refused(code: &'static str, details: Value) -> Self {
        Self::Refused { code, details }
    }
}

/// Check bytecode premise licensing and compute the conditional credal verdict.
pub fn solve_credal(program: &Program) -> Verdict {
    policy::solve(program)
}

/// Bound the mathematical model directly, including underdetermined intervals.
/// Call [solve_credal] for observation and premise-licensing checks.
pub fn bound_polytope(program: &Program) -> Verdict {
    model::mathematical(program)
}

/// Verify a Farkas certificate against the original compiled rows for the query.
/// The multipliers are in the order returned by the verdict's `farkas` field.
pub fn verify_farkas(program: &Program, multipliers: &[num_rational::BigRational]) -> bool {
    model::verify_farkas(program, multipliers)
}
