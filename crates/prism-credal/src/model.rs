use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use serde_json::{Value, json};

use crate::{
    Verdict,
    ir::{Decl, Link, Observation, Polarity, Program, ratio_string},
    simplex::{Feasibility, Row, Tableau, verifies_farkas},
};

const MAX_WORLDS: usize = 128;

pub(crate) struct Model<'a> {
    hyps: &'a [String],
    events: Vec<&'a str>,
    worlds: Vec<(usize, usize)>,
    rows: Vec<Row>,
    sources: Vec<usize>,
}

impl Model<'_> {
    fn bit(&self, event: &str) -> usize {
        self.events
            .iter()
            .position(|&e| e == event)
            .expect("compiled event")
    }

    fn matches_bits(&self, bits: usize, spec: &[Observation]) -> bool {
        spec.iter()
            .all(|g| (bits & (1 << self.bit(&g.ev)) != 0) == (g.pol == Polarity::Pos))
    }

    fn row(
        &mut self,
        source: usize,
        rhs: BigInt,
        coefficient: impl Fn(&Self, usize, usize) -> BigInt,
    ) {
        let mut row: Row = self
            .worlds
            .iter()
            .map(|&(h, bits)| coefficient(self, h, bits))
            .collect();
        row.push(rhs);
        self.rows.push(row);
        self.sources.push(source);
    }

    // P(N | D) = n/d compiles to d·mass(N∩D) − n·mass(D) = 0.
    fn conditional(
        &mut self,
        source: usize,
        value: &BigRational,
        predicates: impl Fn(&Self, usize, usize) -> (bool, bool),
    ) {
        self.row(source, BigInt::zero(), |m, h, bits| {
            let (num, den) = predicates(m, h, bits);
            match (num, den) {
                (true, true) => value.denom() - value.numer(),
                (false, true) => -value.numer(),
                _ => BigInt::zero(),
            }
        });
    }
}

pub(crate) fn selected_partition(program: &Program) -> Option<(usize, &[String])> {
    let q = &program.query;
    program.decls.iter().enumerate().find_map(|(i, d)| match d {
        Decl::PartExhaustive { hyps, .. } if !program.target_is_hyp || hyps.contains(&q.target) => {
            Some((i, hyps.as_slice()))
        }
        _ => None,
    })
}

fn compile(program: &Program) -> Result<Model<'_>, (&'static str, Value)> {
    let q = &program.query;
    let Some((partition, hyps)) = selected_partition(program) else {
        return Err((
            "MISSING_BASE_RATE",
            json!({"detail": "no PART_EXHAUSTIVE partition for target"}),
        ));
    };
    let scoped = |d: &Decl| d.scoped_to(&q.ctx);
    let mut eligible: Vec<_> = program
        .decls
        .iter()
        .enumerate()
        .filter(|(_, d)| scoped(d))
        .collect();
    let mut events = Vec::new();
    fn include<'a>(events: &mut Vec<&'a str>, id: &'a str) {
        if !events.contains(&id) {
            events.push(id);
        }
    }
    for g in &q.given {
        include(&mut events, &g.ev);
    }
    if !program.target_is_hyp {
        include(&mut events, &q.target);
    }
    // Preserve the reference model's event ordering for stable world indices.
    eligible.sort_by_key(|(_, d)| match d {
        Decl::Prior { .. } => 0,
        Decl::Lik { .. } => 1,
        Decl::PosteriorStated { .. } => 2,
        Decl::JointLik { .. } => 3,
        Decl::DetLink { .. } => 4,
        Decl::CondIndep { .. } => 5,
        _ => 6,
    });
    for (_, d) in &eligible {
        match d {
            Decl::Lik { ev, .. } | Decl::PosteriorStated { ev, .. } => include(&mut events, ev),
            Decl::JointLik { spec, .. } => {
                for g in spec {
                    include(&mut events, &g.ev);
                }
            }
            Decl::DetLink { a, b, .. } => {
                include(&mut events, a);
                include(&mut events, b);
            }
            Decl::CondIndep { evs, .. } => {
                for ev in evs {
                    include(&mut events, ev);
                }
            }
            _ => {}
        }
    }
    let count = u32::try_from(events.len())
        .ok()
        .and_then(|k| 1usize.checked_shl(k));
    let Some(patterns) = count.filter(|&n| hyps.len() <= MAX_WORLDS / n) else {
        return Err((
            "POLYTOPE_DIMENSION_LIMIT",
            json!({
                "hypotheses": hyps.len(), "events": events.len(), "maxWorlds": MAX_WORLDS
            }),
        ));
    };

    let mut m = Model {
        hyps,
        events,
        worlds: Vec::new(),
        rows: Vec::new(),
        sources: Vec::new(),
    };
    let links: Vec<_> = eligible
        .iter()
        .filter_map(|(_, d)| match d {
            Decl::DetLink { a, b, rel, .. } => Some((1 << m.bit(a), 1 << m.bit(b), rel)),
            _ => None,
        })
        .collect();
    for h in 0..hyps.len() {
        for bits in 0..patterns {
            let allowed = links.iter().all(|&(a, b, rel)| {
                let (a, b) = (bits & a != 0, bits & b != 0);
                match rel {
                    Link::Equiv => a == b,
                    Link::Implies => !a || b,
                    Link::Mutex => !(a && b),
                }
            });
            if allowed {
                m.worlds.push((h, bits));
            }
        }
    }
    m.row(partition, BigInt::one(), |_, _, _| BigInt::one());
    let hyp = |name: &str| hyps.iter().position(|h| h == name);
    let marginal = |ev: &str, hyp: &str| {
        [Polarity::Pos, Polarity::Neg].into_iter().find_map(|pol| {
            eligible.iter().find_map(|(_, d)| match d {
                Decl::Lik {
                    ev: e,
                    hyp: h,
                    pol: p,
                    value,
                    ..
                } if e == ev && h == hyp && *p == pol => Some(if pol == Polarity::Pos {
                    value.0.clone()
                } else {
                    BigRational::one() - &value.0
                }),
                _ => None,
            })
        })
    };
    for &(i, d) in &eligible {
        match d {
            Decl::Prior { hyp: h, value, .. } => {
                if let Some(h) = hyp(h) {
                    m.row(i, value.0.numer().clone(), |_, w, _| {
                        if w == h {
                            value.0.denom().clone()
                        } else {
                            BigInt::zero()
                        }
                    });
                }
            }
            Decl::Lik {
                ev,
                pol,
                hyp: h,
                value,
                ..
            }
            | Decl::PosteriorStated {
                ev,
                pol,
                hyp: h,
                value,
                ..
            } => {
                if let Some(h) = hyp(h) {
                    let mask = 1 << m.bit(ev);
                    m.conditional(i, &value.0, |_, w, bits| {
                        let observed = (bits & mask != 0) == (*pol == Polarity::Pos);
                        (
                            w == h && observed,
                            if matches!(d, Decl::Lik { .. }) {
                                w == h
                            } else {
                                observed
                            },
                        )
                    });
                }
            }
            Decl::JointLik {
                spec,
                hyp: h,
                value,
                ..
            } => {
                if let Some(h) = hyp(h) {
                    m.conditional(i, &value.0, |m, w, bits| {
                        (w == h && m.matches_bits(bits, spec), w == h)
                    });
                }
            }
            Decl::CondIndep {
                evs, hyp: scope, ..
            } => {
                for (h, name) in hyps
                    .iter()
                    .enumerate()
                    .filter(|(_, h)| scope == "*" || scope == *h)
                {
                    let mut known: Vec<(usize, BigRational)> = Vec::new();
                    let mut repeated = 0;
                    for ev in evs {
                        let bit = 1 << m.bit(ev);
                        if let Some((_, p)) = known.iter().find(|(b, _)| *b == bit) {
                            // Independence of an event from itself forces its
                            // mass to be degenerate. Higher multiplicities
                            // then impose no further constraints.
                            if repeated & bit == 0 {
                                let square = p * p;
                                m.conditional(i, &square, |_, w, bits| {
                                    (w == h && bits & bit != 0, w == h)
                                });
                                repeated |= bit;
                            }
                        } else if let Some(p) = marginal(ev, name) {
                            known.push((bit, p));
                        }
                    }
                    for subset in 0..(1usize << known.len()) {
                        if subset.count_ones() < 2 {
                            continue;
                        }
                        let mut product = BigRational::one();
                        let mut mask = 0;
                        for (k, (bit, p)) in known.iter().enumerate() {
                            if subset & (1 << k) != 0 {
                                product *= p;
                                mask |= bit;
                            }
                        }
                        m.conditional(i, &product, |_, w, bits| {
                            (w == h && bits & mask == mask, w == h)
                        });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(m)
}

pub(crate) fn mathematical(program: &Program) -> Verdict {
    match compile(program) {
        Ok(model) => match solve_model(program, model) {
            Ok(verdict) => verdict,
            Err(detail) => Verdict::refused("SOLVER_LIMIT", json!({"detail": detail})),
        },
        Err((code, details)) => Verdict::refused(code, details),
    }
}

fn solve_model(program: &Program, m: Model<'_>) -> Result<Verdict, &'static str> {
    let q = &program.query;
    let w = m.worlds.len();
    let mut tableau = match Tableau::feasible(&m.rows)? {
        Feasibility::Infeasible(y) => {
            return Ok(Verdict::refused(
                "CONFLICTING_FIGURES",
                json!({
                    "farkas": y.iter().map(ratio_string).collect::<Vec<_>>(),
                    "rowDeclarations": m.sources,
                }),
            ));
        }
        Feasibility::Feasible(tableau) => tableau,
    };
    let evidence: Row = m
        .worlds
        .iter()
        .map(|&(_, bits)| BigInt::from(m.matches_bits(bits, &q.given)))
        .collect();
    let target_hyp = m.hyps.iter().position(|id| *id == q.target);
    let target_bit = (!program.target_is_hyp).then(|| 1 << m.bit(&q.target));
    let mut objective: Row = m
        .worlds
        .iter()
        .zip(&evidence)
        .map(|(&(h, bits), e)| {
            let target = target_bit.map_or(target_hyp == Some(h), |bit| bits & bit != 0);
            if target ^ q.negated {
                e.clone()
            } else {
                BigInt::zero()
            }
        })
        .collect();

    let mut e_min = BigRational::one();
    let mut e_max = BigRational::one();
    if !q.given.is_empty() {
        let negative: Row = evidence.iter().map(|x| -x).collect();
        e_min = -tableau.maximize(&negative)?.value;
        e_max = tableau.maximize(&evidence)?.value;
        if e_max.is_zero() {
            return Ok(Verdict::refused(
                "DEGENERATE_PARTITION",
                json!({"cause": "ZERO_SUPPORT", "eMax": "0/1"}),
            ));
        }
        if e_min.is_zero() {
            return Ok(Verdict::refused(
                "ZERO_SUPPORT_POSSIBLE",
                json!({"eMin": "0/1", "eMax": ratio_string(&e_max)}),
            ));
        }
    }

    tableau = tableau.condition(&evidence)?;
    objective.push(BigInt::zero());
    let negative: Row = objective.iter().map(|x| -x).collect();
    let minimum = tableau.maximize(&negative)?;
    let maximum = tableau.maximize(&objective)?;
    let p_min = -minimum.value;
    let interval = json!({"min": ratio_string(&p_min), "max": ratio_string(&maximum.value)});
    let encode_vertex = |xs: &[BigRational]| -> Value {
        let support: Vec<_> = xs[..w]
            .iter()
            .enumerate()
            .filter(|(_, x)| !x.is_zero())
            .map(|(i, x)| json!({"i": i, "v": ratio_string(x)}))
            .collect();
        json!({"uStar": support, "tStar": ratio_string(&xs[w])})
    };
    if p_min != maximum.value {
        return Ok(Verdict::refused(
            "UNDERDETERMINED_INTERVAL",
            json!({
                "interval": interval, "eMin": ratio_string(&e_min), "eMax": ratio_string(&e_max),
                "vertices": {"min": encode_vertex(&minimum.vertex), "max": encode_vertex(&maximum.vertex)},
            }),
        ));
    }

    let self_given = !program.target_is_hyp && q.given.iter().any(|g| g.ev == q.target);
    let mut credal = encode_vertex(&maximum.vertex);
    credal["worlds"] = json!(w);
    credal["eMin"] = json!(ratio_string(&e_min));
    credal["eMax"] = json!(ratio_string(&e_max));
    Ok(Verdict::Answer {
        value: ratio_string(&p_min),
        target: q.target.clone(),
        via: if self_given { "TAUTOLOGY" } else { "CREDAL_LP" },
        interval,
        credal,
        witnesses: program
            .decls
            .iter()
            .filter(|d| d.scoped_to(&q.ctx))
            .cloned()
            .collect(),
    })
}

pub(crate) fn verify_farkas(program: &Program, multipliers: &[BigRational]) -> bool {
    compile(program).is_ok_and(|model| verifies_farkas(&model.rows, multipliers))
}
