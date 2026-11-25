use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use num_rational::BigRational;
use num_traits::One;
use serde_json::{Value, json};

use crate::{
    Program, Verdict,
    ir::{Decl, Observation, Polarity, ratio_string},
    model::{mathematical, selected_partition},
};

fn premises(
    program: &Program,
    predicate: impl Fn(&Decl) -> bool,
) -> impl Iterator<Item = (usize, &Decl)> {
    program
        .decls
        .iter()
        .enumerate()
        .filter(move |(_, d)| predicate(d))
}

fn conflicting_values<K: Eq + Hash, V: Eq>(
    entries: impl Iterator<Item = (usize, K, V)>,
) -> Option<(usize, usize)> {
    let mut seen = HashMap::new();
    for (i, key, value) in entries {
        if let Some((first, previous)) = seen.get(&key) {
            if *previous != value {
                return Some((*first, i));
            }
        } else {
            seen.insert(key, (i, value));
        }
    }
    None
}

fn numerical_certificate(program: &Program, details: &mut Value) {
    if let Verdict::Refused {
        code: "CONFLICTING_FIGURES",
        details: certificate,
    } = mathematical(program)
    {
        for field in ["farkas", "rowDeclarations"] {
            if let Some(value) = certificate.get(field) {
                details[field] = value.clone();
            }
        }
    }
}

fn conflict(program: &Program, pair: (usize, usize)) -> Verdict {
    let mut details = json!({"declarations": [pair.0, pair.1]});
    if program.decls[pair.0].scoped_to(&program.query.ctx) {
        numerical_certificate(program, &mut details);
    }
    Verdict::refused("CONFLICTING_FIGURES", details)
}

fn available(
    program: &Program,
    predicate: impl Fn(&Decl) -> bool,
) -> (Option<&Decl>, Option<&Decl>) {
    let mut other = None;
    for (_, d) in premises(program, predicate) {
        if d.scoped_to(&program.query.ctx) {
            return (Some(d), other);
        }
        other.get_or_insert(d);
    }
    (None, other)
}

fn outside(program: &Program, d: &Decl, what: String) -> Verdict {
    let ctx = match d {
        Decl::Obs { ctx, .. }
        | Decl::Prior { ctx, .. }
        | Decl::Lik { ctx, .. }
        | Decl::PosteriorStated { ctx, .. } => ctx,
        _ => unreachable!(
            "context diagnostics only inspect required numerical and observation premises"
        ),
    };
    Verdict::refused(
        "CONTEXT_MISMATCH",
        json!({
            "what": what, "haveCtx": ctx, "queryCtx": program.query.ctx, "span": d.span()
        }),
    )
}

fn obs_matches(d: &Decl, g: &Observation) -> bool {
    matches!(d, Decl::Obs { ev, pol, .. } if *ev == g.ev && *pol == g.pol)
}

fn lik_matches(d: &Decl, event: &str, hyp: &str, polarity: Polarity) -> bool {
    matches!(d, Decl::Lik { ev, hyp: h, pol, .. } if ev == event && h == hyp && *pol == polarity)
}

fn licensed(program: &Program, a: &str, b: &str, h: &str) -> bool {
    program
        .decls
        .iter()
        .filter(|d| d.scoped_to(&program.query.ctx))
        .any(|d| match d {
            Decl::CondIndep { hyp, evs, .. } => {
                (hyp == h || hyp == "*") && evs.iter().any(|e| e == a) && evs.iter().any(|e| e == b)
            }
            Decl::DetLink { a: x, b: y, .. } => (x == a && y == b) || (x == b && y == a),
            Decl::JointLik { hyp, spec, .. } => {
                hyp == h && spec.iter().any(|g| g.ev == a) && spec.iter().any(|g| g.ev == b)
            }
            _ => false,
        })
}

pub(crate) fn solve(program: &Program) -> Verdict {
    let q = &program.query;
    let mut seen = HashSet::new();
    for d in &program.decls {
        if let Decl::Obs { ev, span, .. } = d
            && !seen.insert(ev)
        {
            return Verdict::refused("DUPLICATE_EVIDENCE", json!({"ev": ev, "span": span}));
        }
    }
    seen.clear();
    for g in &q.given {
        if !seen.insert(&g.ev) {
            return Verdict::refused("DUPLICATE_EVIDENCE", json!({"ev": g.ev, "inQuery": true}));
        }
    }

    for kind in 0..3 {
        let figures = program.decls.iter().enumerate().filter_map(|(i, d)| {
            let (hyp, ev, pol, value, ctx) = match (kind, d) {
                (
                    0,
                    Decl::Prior {
                        hyp, value, ctx, ..
                    },
                ) => (hyp, None, Polarity::Pos, value, ctx),
                (
                    1,
                    Decl::Lik {
                        hyp,
                        ev,
                        pol,
                        value,
                        ctx,
                        ..
                    },
                )
                | (
                    2,
                    Decl::PosteriorStated {
                        hyp,
                        ev,
                        pol,
                        value,
                        ctx,
                        ..
                    },
                ) => (hyp, Some(ev), *pol, value, ctx),
                _ => return None,
            };
            Some((i, (hyp, ev, pol == Polarity::Pos, ctx), value))
        });
        if let Some(pair) = conflicting_values(figures) {
            return conflict(program, pair);
        }
    }

    let partition = selected_partition(program);
    let hyps = partition.map_or(&[][..], |(_, h)| h);
    let mut needed = Vec::new();
    let mut missing_obs = Vec::new();
    let mut mismatch = None;
    for g in &q.given {
        let (valid, other) = available(program, |d| obs_matches(d, g));
        if let Some(d) = valid {
            needed.push(d);
        } else if let Some(d) = other {
            mismatch = Some(outside(program, d, format!("OBS {}", g.ev)));
        } else {
            missing_obs.push(g);
        }
    }
    let mut missing_priors = Vec::new();
    let mut sum = BigRational::from_integer(0.into());
    for h in hyps {
        let (valid, other) = available(
            program,
            |d| matches!(d, Decl::Prior { hyp, .. } if hyp == h),
        );
        if let Some(d @ Decl::Prior { value, .. }) = valid {
            needed.push(d);
            sum += &value.0;
        } else if let Some(d) = other {
            mismatch.get_or_insert_with(|| outside(program, d, format!("PRIOR {h}")));
        } else {
            missing_priors.push(h);
        }
    }
    let mut missing_liks = Vec::new();
    let is_event_target = !program.target_is_hyp && !q.given.iter().any(|g| g.ev == q.target);
    let factors = q
        .given
        .iter()
        .map(|g| (g.ev.as_str(), g.pol, false))
        .chain(is_event_target.then_some((q.target.as_str(), Polarity::Pos, true)));
    for (event, pol, is_target) in factors {
        for h in hyps {
            let (valid, other) = available(program, |d| lik_matches(d, event, h, pol));
            let opposite = if pol == Polarity::Pos {
                Polarity::Neg
            } else {
                Polarity::Pos
            };
            let (complement, other_complement) =
                available(program, |d| lik_matches(d, event, h, opposite));
            // The decision layer uses complemented positive likelihoods for
            // negative evidence, and either polarity for an event target.
            let complement = complement.filter(|_| is_target || pol == Polarity::Neg);
            let other = other.or(other_complement.filter(|_| is_target || pol == Polarity::Neg));
            if let Some(d) = valid.or(complement) {
                needed.push(d);
            } else if let Some(d) = other {
                mismatch
                    .get_or_insert_with(|| outside(program, d, format!("LIK {event} GIVEN {h}")));
            } else {
                missing_liks.push(json!({"ev": event, "hyp": h, "pol": pol, "target": is_target}));
            }
        }
    }

    if let Some(verdict) = mismatch {
        return verdict;
    }
    let direct_signature = |d: &Decl| {
        matches!(d, Decl::PosteriorStated { ev, pol, hyp, .. }
            if program.target_is_hyp && *hyp == q.target && q.given.len() == 1
                && *ev == q.given[0].ev && *pol == q.given[0].pol)
    };
    let (direct_post, elsewhere) = available(program, direct_signature);
    if direct_post.is_none()
        && let Some(d @ Decl::PosteriorStated { ev, hyp, .. }) = elsewhere
    {
        return outside(program, d, format!("POSTERIOR_STATED {hyp} GIVEN {ev}"));
    }
    if !missing_obs.is_empty() {
        return Verdict::refused(
            "MISSING_OBSERVATION_WITNESS",
            json!({"givens": missing_obs}),
        );
    }
    for d in needed.iter().copied().chain(direct_post) {
        for na in &program.decls {
            if let Decl::NonAsserted { span, reason } = na
                && d.span().overlaps(*span)
            {
                return Verdict::refused(
                    "SCOPED_OR_RETRACTED_PREMISE",
                    json!({
                        "span": d.span(), "reason": reason, "nonAssertedSpan": span
                    }),
                );
            }
        }
    }

    let factors_missing =
        partition.is_none() || !missing_priors.is_empty() || !missing_liks.is_empty();
    if factors_missing {
        for d in &program.decls {
            if let Decl::PosteriorStated { ev, hyp, span, .. } = d
                && !direct_signature(d)
                && d.scoped_to(&q.ctx)
                && (partition.is_none() || hyps.contains(hyp))
                && q.given.iter().any(|g| g.ev == *ev)
            {
                return Verdict::refused(
                    "REVERSED_CONDITIONAL",
                    json!({
                        "span": span, "stated": format!("POSTERIOR_STATED {hyp} GIVEN {ev}")
                    }),
                );
            }
        }
        let result = mathematical(program);
        match &result {
            Verdict::Answer { .. }
            | Verdict::Refused {
                code:
                    "DEGENERATE_PARTITION"
                    | "ZERO_SUPPORT_POSSIBLE"
                    | "CONFLICTING_FIGURES"
                    | "POLYTOPE_DIMENSION_LIMIT"
                    | "SOLVER_LIMIT",
                ..
            } => return result,
            _ => {}
        }
        if partition.is_none() || !missing_priors.is_empty() {
            return Verdict::refused("MISSING_BASE_RATE", json!({"missingHyps": missing_priors}));
        }
        if sum.is_one() {
            return Verdict::refused("INCOMPLETE_FACTORIZATION", json!({"missing": missing_liks}));
        }
    }
    if !sum.is_one() {
        let mut details =
            json!({"reason": "priors do not sum to 1 over partition", "sum": ratio_string(&sum)});
        numerical_certificate(program, &mut details);
        return Verdict::refused("MISSING_BASE_RATE", details);
    }

    let mut events: Vec<_> = q.given.iter().map(|g| g.ev.as_str()).collect();
    if is_event_target {
        events.push(&q.target);
    }
    for (i, &a) in events.iter().enumerate() {
        for &b in &events[i + 1..] {
            for h in hyps {
                if !licensed(program, a, b, h) {
                    let result = mathematical(program);
                    let interval = match &result {
                        Verdict::Answer { interval, .. } => Some(interval),
                        Verdict::Refused { details, .. } => details.get("interval"),
                    };
                    let mut details = json!({"hyp": h, "evs": [a, b], "needed": events});
                    if let Some(interval) = interval {
                        details["interval"] = interval.clone();
                    }
                    return Verdict::refused("UNLICENSED_INDEPENDENCE", details);
                }
            }
        }
    }

    for d in &program.decls {
        if let Decl::Conflict { span_a, span_b } = d
            && needed.iter().any(|d| d.span().overlaps(*span_a))
            && needed.iter().any(|d| d.span().overlaps(*span_b))
        {
            return Verdict::refused(
                "CONFLICTING_FIGURES",
                json!({"spanA": span_a, "spanB": span_b, "via": "CONFLICT decl"}),
            );
        }
    }
    mathematical(program)
}
