use std::{collections::HashSet, error::Error, fmt, sync::LazyLock};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de, de::DeserializeOwned};
use serde_json::Value;

#[derive(Debug)]
pub struct ParseError(pub(crate) String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ParseError {}

fn invalid(message: impl Into<String>) -> ParseError {
    ParseError(message.into())
}

pub(crate) fn ratio_string(value: &BigRational) -> String {
    format!("{}/{}", value.numer(), value.denom())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probability(pub(crate) BigRational);

impl Probability {
    pub fn as_rational(&self) -> &BigRational {
        &self.0
    }
}

impl Serialize for Probability {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&ratio_string(&self.0))
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        static RATIONAL: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"^(-?[0-9]+)(?:\s*/\s*([0-9]+))?$").unwrap());
        let input = String::deserialize(deserializer)?;
        let parts = RATIONAL
            .captures(input.trim())
            .ok_or_else(|| de::Error::custom("expected integer or rational string"))?;
        let numerator: BigInt = parts[1].parse().map_err(de::Error::custom)?;
        let denominator: BigInt = parts
            .get(2)
            .map_or("1", |m| m.as_str())
            .parse()
            .map_err(de::Error::custom)?;
        if denominator.is_zero() {
            return Err(de::Error::custom("zero denominator"));
        }
        let value = BigRational::new(numerator, denominator);
        if value.is_negative() || value > BigRational::one() {
            return Err(de::Error::custom("probability out of [0,1]"));
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Span {
    pub s: usize,
    pub e: usize,
}

impl Span {
    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.s < other.e && other.s < self.e
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Polarity {
    #[default]
    Pos,
    Neg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub ev: String,
    #[serde(default)]
    pub pol: Polarity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Link {
    Equiv,
    Implies,
    Mutex,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Reason {
    Quoted,
    Retracted,
    Hypothetical,
    HedgedApprox,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SentenceTag {
    Premise,
    Query,
    ContextScope,
    NonAsserted,
    Fluff,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Decl {
    Hyp {
        id: String,
        span: Span,
    },
    PartExhaustive {
        hyps: Vec<String>,
        span: Span,
    },
    Event {
        id: String,
        span: Span,
    },
    Prior {
        hyp: String,
        value: Probability,
        ctx: String,
        span: Span,
    },
    Lik {
        ev: String,
        pol: Polarity,
        hyp: String,
        value: Probability,
        ctx: String,
        span: Span,
    },
    PosteriorStated {
        hyp: String,
        ev: String,
        pol: Polarity,
        value: Probability,
        ctx: String,
        span: Span,
    },
    JointLik {
        spec: Vec<Observation>,
        hyp: String,
        value: Probability,
        ctx: String,
        span: Span,
    },
    DetLink {
        a: String,
        rel: Link,
        b: String,
        ctx: String,
        span: Span,
    },
    Obs {
        ev: String,
        pol: Polarity,
        ctx: String,
        span: Span,
    },
    CondIndep {
        evs: Vec<String>,
        hyp: String,
        ctx: String,
        span: Span,
    },
    SentTag {
        idx: usize,
        tag: SentenceTag,
        ctx: Option<String>,
        span: Span,
    },
    Conflict {
        #[serde(rename = "spanA")]
        span_a: Span,
        #[serde(rename = "spanB")]
        span_b: Span,
    },
    NonAsserted {
        span: Span,
        reason: Reason,
    },
}

impl Decl {
    pub(crate) fn span(&self) -> Span {
        match *self {
            Self::Hyp { span, .. }
            | Self::PartExhaustive { span, .. }
            | Self::Event { span, .. }
            | Self::Prior { span, .. }
            | Self::Lik { span, .. }
            | Self::PosteriorStated { span, .. }
            | Self::JointLik { span, .. }
            | Self::DetLink { span, .. }
            | Self::Obs { span, .. }
            | Self::CondIndep { span, .. }
            | Self::SentTag { span, .. }
            | Self::NonAsserted { span, .. } => span,
            Self::Conflict { span_a, .. } => span_a,
        }
    }

    pub(crate) fn scoped_to(&self, query_ctx: &str) -> bool {
        let ctx = match self {
            Self::Prior { ctx, .. }
            | Self::Lik { ctx, .. }
            | Self::PosteriorStated { ctx, .. }
            | Self::JointLik { ctx, .. }
            | Self::DetLink { ctx, .. }
            | Self::Obs { ctx, .. }
            | Self::CondIndep { ctx, .. } => ctx,
            _ => return false,
        };
        ctx == "*" || ctx == query_ctx
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QueryOp {
    AskProb,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub op: QueryOp,
    pub target: String,
    #[serde(default)]
    pub negated: bool,
    pub given: Vec<Observation>,
    pub ctx: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Program {
    pub(crate) decls: Vec<Decl>,
    pub(crate) query: Query,
    #[serde(skip)]
    pub(crate) target_is_hyp: bool,
}

impl Program {
    pub fn new(decls: Vec<Decl>, query: Query) -> Result<Self, ParseError> {
        validate(&decls, &query)?;
        let target_is_hyp = decls
            .iter()
            .any(|d| matches!(d, Decl::Hyp { id, .. } if *id == query.target));
        Ok(Self {
            decls,
            query,
            target_is_hyp,
        })
    }

    pub fn from_json(value: Value) -> Result<Self, ParseError> {
        #[derive(Deserialize)]
        struct JsonProgram {
            decls: Vec<Decl>,
            query: Query,
        }
        match value {
            Value::String(bytecode) => parse_bytecode(&bytecode),
            other => {
                let JsonProgram { decls, query } = decode(other)?;
                Self::new(decls, query)
            }
        }
    }

    pub fn declarations(&self) -> &[Decl] {
        &self.decls
    }

    pub fn query(&self) -> &Query {
        &self.query
    }
}

fn decode<T: DeserializeOwned>(value: Value) -> Result<T, ParseError> {
    serde_json::from_value(value).map_err(|e| invalid(e.to_string()))
}

const KEYWORDS: &[&str] = &[
    "HYP",
    "PART_EXHAUSTIVE",
    "EVENT",
    "PRIOR",
    "LIK",
    "POSTERIOR_STATED",
    "OBS",
    "COND_INDEP",
    "CONFLICT",
    "NON_ASSERTED",
    "ASK_PROB",
    "GIVEN",
    "POS",
    "NEG",
    "NOT",
    "END",
    "QUOTED",
    "RETRACTED",
    "HYPOTHETICAL",
    "HEDGED_APPROX",
    "JOINT_LIK",
    "DET_LINK",
    "SENT_TAG",
    "EQUIV",
    "IMPLIES",
    "MUTEX",
    "PREMISE",
    "QUERY",
    "CONTEXT_SCOPE",
    "FLUFF",
];

fn identifier(s: &str) -> bool {
    static ID: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_-]*$").unwrap());
    ID.is_match(s)
}

fn validate(decls: &[Decl], query: &Query) -> Result<(), ParseError> {
    if decls.is_empty() {
        return Err(invalid("program has no declarations"));
    }
    if !identifier(&query.ctx) {
        return Err(invalid("query requires a concrete context identifier"));
    }
    let mut hyps = HashSet::new();
    let mut events = HashSet::new();
    for decl in decls {
        let (set, id) = match decl {
            Decl::Hyp { id, .. } => (&mut hyps, id),
            Decl::Event { id, .. } => (&mut events, id),
            _ => continue,
        };
        if !identifier(id) || KEYWORDS.contains(&id.as_str()) || !set.insert(id.as_str()) {
            return Err(invalid(format!("invalid or duplicate identifier {id}")));
        }
    }
    let need = |set: &HashSet<&str>, id: &str, kind: &str| {
        if !set.contains(id) {
            return Err(invalid(format!("undeclared {kind} {id}")));
        }
        Ok(())
    };
    let hyp = |id: &str| need(&hyps, id, "HYP");
    let ev = |id: &str| need(&events, id, "EVENT");
    let ctx = |id: &str| {
        if id == "*" || identifier(id) {
            Ok(())
        } else {
            Err(invalid("invalid context identifier"))
        }
    };
    let span = |sp: Span| {
        if sp.s < sp.e {
            Ok(())
        } else {
            Err(invalid("malformed span"))
        }
    };
    for decl in decls {
        span(decl.span())?;
        match decl {
            Decl::PartExhaustive { hyps, .. } => {
                if hyps.is_empty() || hyps.iter().collect::<HashSet<_>>().len() != hyps.len() {
                    return Err(invalid("partition must have distinct hypotheses"));
                }
                hyps.iter().try_for_each(|id| hyp(id))?;
            }
            Decl::Prior { hyp: h, ctx: c, .. } => {
                hyp(h)?;
                ctx(c)?;
            }
            Decl::Lik {
                hyp: h,
                ev: e,
                ctx: c,
                ..
            }
            | Decl::PosteriorStated {
                hyp: h,
                ev: e,
                ctx: c,
                ..
            } => {
                hyp(h)?;
                ev(e)?;
                ctx(c)?;
            }
            Decl::Obs { ev: e, ctx: c, .. } => {
                ev(e)?;
                ctx(c)?;
            }
            Decl::DetLink { a, b, ctx: c, .. } => {
                ev(a)?;
                ev(b)?;
                ctx(c)?;
            }
            Decl::JointLik {
                spec,
                hyp: h,
                ctx: c,
                ..
            } => {
                if spec.len() < 2 {
                    return Err(invalid("JOINT_LIK needs at least two events"));
                }
                spec.iter().try_for_each(|g| ev(&g.ev))?;
                hyp(h)?;
                ctx(c)?;
            }
            Decl::CondIndep {
                evs,
                hyp: h,
                ctx: c,
                ..
            } => {
                if evs.is_empty() {
                    return Err(invalid("COND_INDEP needs at least one event"));
                }
                evs.iter().try_for_each(|e| ev(e))?;
                if h != "*" {
                    hyp(h)?;
                }
                ctx(c)?;
            }
            Decl::SentTag { ctx: Some(c), .. } => ctx(c)?,
            Decl::Conflict { span_b, .. } => span(*span_b)?,
            _ => {}
        }
    }
    if !hyps.contains(query.target.as_str()) && !events.contains(query.target.as_str()) {
        return Err(invalid("undeclared ASK_PROB target"));
    }
    query.given.iter().try_for_each(|g| ev(&g.ev))
}

struct Tokens<'a> {
    rest: &'a str,
    index: usize,
}

impl<'a> Tokens<'a> {
    fn peek(&self) -> Result<Option<&'a str>, ParseError> {
        static TOKEN: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"^(?:[\[\]():,*]|[A-Za-z_][A-Za-z0-9_-]*|-?[0-9]+(?:\s*/\s*[0-9]+)?)")
                .unwrap()
        });
        let rest = self.rest.trim_start();
        if rest.is_empty() {
            return Ok(None);
        }
        TOKEN
            .find(rest)
            .map(|m| Some(m.as_str()))
            .ok_or_else(|| invalid(format!("unexpected character at token {}", self.index)))
    }

    fn next(&mut self) -> Result<&'a str, ParseError> {
        let token = self
            .peek()?
            .ok_or_else(|| invalid("unexpected end of program"))?;
        self.rest = &self.rest.trim_start()[token.len()..];
        self.index += 1;
        Ok(token)
    }

    fn expect(&mut self, expected: &str) -> Result<(), ParseError> {
        let token = self.next()?;
        if token != expected {
            return Err(invalid(format!(
                "expected {expected}, got {token} at token {}",
                self.index - 1
            )));
        }
        Ok(())
    }

    fn int(&mut self) -> Result<usize, ParseError> {
        let token = self.next()?;
        let token = if token.starts_with('-') && token[1..].bytes().all(|b| b == b'0') {
            &token[1..]
        } else {
            token
        };
        token
            .parse()
            .map_err(|_| invalid("expected nonnegative integer"))
    }

    fn span(&mut self) -> Result<Span, ParseError> {
        self.expect("[")?;
        let s = self.int()?;
        self.expect(",")?;
        let e = self.int()?;
        self.expect("]")?;
        Ok(Span { s, e })
    }

    fn atom<T: DeserializeOwned>(&mut self) -> Result<T, ParseError> {
        decode(Value::String(self.next()?.to_owned()))
    }

    fn name(&mut self) -> Result<String, ParseError> {
        Ok(self.next()?.to_owned())
    }

    fn list<T>(
        &mut self,
        read: impl Fn(&mut Self) -> Result<T, ParseError>,
    ) -> Result<Vec<T>, ParseError> {
        self.expect("(")?;
        let mut values = Vec::new();
        while self.peek()? != Some(")") {
            values.push(read(self)?);
        }
        self.expect(")")?;
        Ok(values)
    }

    fn observation(&mut self, require_pol: bool) -> Result<Observation, ParseError> {
        let ev = self.name()?;
        let pol = if require_pol || self.peek()? == Some(":") {
            self.expect(":")?;
            self.atom()?
        } else {
            Polarity::Pos
        };
        Ok(Observation { ev, pol })
    }

    fn declaration(&mut self, op: &str) -> Result<Decl, ParseError> {
        Ok(match op {
            "HYP" => Decl::Hyp {
                id: self.name()?,
                span: self.span()?,
            },
            "EVENT" => Decl::Event {
                id: self.name()?,
                span: self.span()?,
            },
            "PART_EXHAUSTIVE" => Decl::PartExhaustive {
                hyps: self.list(Self::name)?,
                span: self.span()?,
            },
            "PRIOR" => Decl::Prior {
                hyp: self.name()?,
                value: self.atom()?,
                ctx: self.name()?,
                span: self.span()?,
            },
            "LIK" => {
                let ev = self.name()?;
                let pol = self.atom()?;
                self.expect("GIVEN")?;
                Decl::Lik {
                    ev,
                    pol,
                    hyp: self.name()?,
                    value: self.atom()?,
                    ctx: self.name()?,
                    span: self.span()?,
                }
            }
            "POSTERIOR_STATED" => {
                let hyp = self.name()?;
                self.expect("GIVEN")?;
                Decl::PosteriorStated {
                    hyp,
                    ev: self.name()?,
                    pol: self.atom()?,
                    value: self.atom()?,
                    ctx: self.name()?,
                    span: self.span()?,
                }
            }
            "JOINT_LIK" => {
                let spec = self.list(|p| p.observation(true))?;
                self.expect("GIVEN")?;
                Decl::JointLik {
                    spec,
                    hyp: self.name()?,
                    value: self.atom()?,
                    ctx: self.name()?,
                    span: self.span()?,
                }
            }
            "DET_LINK" => Decl::DetLink {
                a: self.name()?,
                rel: self.atom()?,
                b: self.name()?,
                ctx: self.name()?,
                span: self.span()?,
            },
            "OBS" => Decl::Obs {
                ev: self.name()?,
                pol: self.atom()?,
                ctx: self.name()?,
                span: self.span()?,
            },
            "COND_INDEP" => {
                let evs = self.list(Self::name)?;
                self.expect("GIVEN")?;
                Decl::CondIndep {
                    evs,
                    hyp: self.name()?,
                    ctx: self.name()?,
                    span: self.span()?,
                }
            }
            "SENT_TAG" => {
                let idx = self.int()?;
                let tag = self.atom()?;
                let ctx = if self.peek()? == Some("[") {
                    None
                } else {
                    Some(self.name()?)
                };
                Decl::SentTag {
                    idx,
                    tag,
                    ctx,
                    span: self.span()?,
                }
            }
            "CONFLICT" => Decl::Conflict {
                span_a: self.span()?,
                span_b: self.span()?,
            },
            "NON_ASSERTED" => Decl::NonAsserted {
                span: self.span()?,
                reason: self.atom()?,
            },
            _ => return Err(invalid(format!("unknown declaration {op}"))),
        })
    }
}

pub fn parse_bytecode(bytecode: &str) -> Result<Program, ParseError> {
    let mut tokens = Tokens {
        rest: bytecode,
        index: 0,
    };
    let mut decls = Vec::new();
    loop {
        let op = tokens.next()?;
        if op == "ASK_PROB" {
            break;
        }
        decls.push(tokens.declaration(op)?);
    }
    let negated = tokens.peek()? == Some("NOT");
    if negated {
        tokens.expect("NOT")?;
    }
    let target = tokens.name()?;
    tokens.expect("GIVEN")?;
    let given = tokens.list(|p| p.observation(false))?;
    let ctx = tokens.name()?;
    tokens.expect("END")?;
    if tokens.peek()?.is_some() {
        return Err(invalid("trailing tokens after END"));
    }
    Program::new(
        decls,
        Query {
            op: QueryOp::AskProb,
            target,
            negated,
            given,
            ctx,
        },
    )
}
