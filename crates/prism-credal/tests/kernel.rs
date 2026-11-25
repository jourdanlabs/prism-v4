use prism_credal::{bound_polytope, parse_bytecode, solve_credal, verify_farkas};
use serde_json::{Value, json};

const BASE: &str = "
HYP h [0,1]
PART_EXHAUSTIVE (h) [0,1]
PRIOR h 1 * [0,1]
EVENT a [0,1]
EVENT b [0,1]
";

fn program(body: &str) -> prism_credal::Program {
    parse_bytecode(&format!("{BASE}{body}\nEND")).unwrap()
}

fn solve(body: &str) -> Value {
    serde_json::to_value(solve_credal(&program(body))).unwrap()
}

#[test]
fn conditional_extrema_return_exact_distinct_vertices() {
    let p = program(
        "
LIK a POS GIVEN h 1/2 * [0,1]
LIK b POS GIVEN h 3/4 * [0,1]
OBS b POS q [0,1]
ASK_PROB a GIVEN (b) q",
    );
    let decision = serde_json::to_value(bound_polytope(&p)).unwrap();
    let details = &decision["details"];
    assert_eq!(decision["code"], "UNDERDETERMINED_INTERVAL");
    assert_eq!(details["interval"], json!({"min": "1/3", "max": "2/3"}));
    assert_eq!(
        details["vertices"],
        json!({
            "min": {"tStar": "4/3", "uStar": [{"i": 1, "v": "2/3"}, {"i": 2, "v": "1/3"}, {"i": 3, "v": "1/3"}]},
            "max": {"tStar": "4/3", "uStar": [{"i": 0, "v": "1/3"}, {"i": 1, "v": "1/3"}, {"i": 3, "v": "2/3"}]}
        })
    );
    let policy = serde_json::to_value(solve_credal(&p)).unwrap();
    assert_eq!(policy["code"], "UNLICENSED_INDEPENDENCE");
    assert_eq!(policy["details"]["interval"], details["interval"]);
}

#[test]
fn joint_and_each_deterministic_relation_constrain_the_worlds() {
    for (licence, value) in [
        ("DET_LINK a EQUIV b q [0,1]", "1/1"),
        ("DET_LINK a IMPLIES b q [0,1]", "1/1"),
        ("DET_LINK a MUTEX b q [0,1]", "0/1"),
        ("JOINT_LIK (a:POS b:POS) GIVEN h 1/8 q [0,1]", "1/4"),
        ("COND_INDEP (a b) GIVEN * q [0,1]", "1/2"),
    ] {
        let body = format!(
            "
LIK a NEG GIVEN h 1/2 * [0,1]
LIK b POS GIVEN h 1/2 * [0,1]
{licence}
OBS a POS q [0,1]
ASK_PROB b GIVEN (a) q"
        );
        assert_eq!(solve(&body)["value"], value, "{licence}");
    }
}

#[test]
fn foreign_context_licenses_and_marginals_cannot_answer() {
    let body = "
LIK a POS GIVEN h 1/2 * [0,1]
LIK b POS GIVEN h 1/2 * [0,1]
DET_LINK a EQUIV b foreign [0,1]
COND_INDEP (a b) GIVEN * foreign [0,1]
JOINT_LIK (a:POS b:POS) GIVEN h 1/2 foreign [0,1]
OBS a POS q [0,1]
ASK_PROB b GIVEN (a) q";
    assert_eq!(solve(body)["code"], "UNLICENSED_INDEPENDENCE");
    assert_eq!(
        solve(&body.replace("LIK b POS GIVEN h 1/2 *", "LIK b POS GIVEN h 1/2 foreign"))["code"],
        "CONTEXT_MISMATCH"
    );
}

#[test]
fn zero_support_and_possible_zero_support_have_distinct_verdicts() {
    let fixed = solve(
        "
LIK a POS GIVEN h 0 * [0,1]
OBS a POS q [0,1]
ASK_PROB a GIVEN (a) q",
    );
    assert_eq!(fixed["code"], "DEGENERATE_PARTITION");
    assert_eq!(fixed["details"]["cause"], "ZERO_SUPPORT");
    let free = solve("OBS a POS q [0,1]\nASK_PROB a GIVEN (a) q");
    assert_eq!(free["code"], "ZERO_SUPPORT_POSSIBLE");
    assert_eq!(free["details"]["eMax"], "1/1");
}

#[test]
fn certificate_proves_nonnegative_world_mass_is_impossible() {
    // One MUTEX pair leaves worlds 00, 10, 01. Its two 3/4 marginals
    // independently require 3/4 mass on each of the last two worlds.
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};
    let p = program(
        "
DET_LINK a MUTEX b * [0,1]
LIK a POS GIVEN h 3/4 * [0,1]
LIK b POS GIVEN h 3/4 * [0,1]
ASK_PROB h GIVEN () q",
    );
    let decision = serde_json::to_value(solve_credal(&p)).unwrap();
    assert_eq!(decision["code"], "CONFLICTING_FIGURES");
    assert_eq!(decision["details"]["rowDeclarations"], json!([1, 2, 6, 7]));
    let y = decision["details"]["farkas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().parse::<BigRational>().unwrap())
        .collect::<Vec<_>>();
    let a_b = [[1, 1, 1, 1], [1, 1, 1, 1], [-3, 1, -3, 0], [-3, -3, 1, 0]];
    let dot = |j: usize| {
        a_b.iter()
            .zip(&y)
            .map(|(row, v)| v * BigRational::from_integer(row[j].into()))
            .sum::<BigRational>()
    };
    assert!((0..3).all(|j| dot(j) >= BigRational::zero()));
    assert!(dot(3).is_negative());
    assert!(verify_farkas(&p, &y));
    assert!(!verify_farkas(&p, &y[..y.len() - 1]));
    assert!(!verify_farkas(
        &p,
        &y.iter().map(|v| -v).collect::<Vec<_>>()
    ));
}

#[test]
fn posterior_and_bigint_negation_remain_exact() {
    let decision = solve("
LIK a POS GIVEN h 99999999999999999999999999999999999999 / 100000000000000000000000000000000000000 * [0,1]
ASK_PROB NOT a GIVEN () q");
    assert_eq!(
        decision["value"],
        "1/100000000000000000000000000000000000000"
    );

    let p = parse_bytecode(
        "
HYP h [0,1]
HYP k [0,1]
PART_EXHAUSTIVE (h k) [0,1]
EVENT a [0,1]
DET_LINK a EQUIV a * [0,1]
LIK a POS GIVEN h 1 * [0,1]
LIK a POS GIVEN k 1 * [0,1]
POSTERIOR_STATED h GIVEN a POS 4/7 q [0,1]
OBS a POS q [0,1]
ASK_PROB NOT h GIVEN (a) q
END",
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(solve_credal(&p)).unwrap()["value"],
        "3/7"
    );
}

#[test]
fn bytecode_validation_rejects_malformed_or_unbound_input() {
    for bad in [
        "ASK_PROB h GIVEN () *",
        "ASK_PROB h GIVEN (unknown) q",
        "ASK_PROB unknown GIVEN () q",
        "LIK a POS GIVEN h 1/0 q [0,1]\nASK_PROB h GIVEN () q",
        "LIK a POS GIVEN h 2 q [0,1]\nASK_PROB h GIVEN () q",
        "HYP h [0,1]\nASK_PROB h GIVEN () q",
        "PART_EXHAUSTIVE (h h) [0,1]\nASK_PROB h GIVEN () q",
        "SENT_TAG 0 QUERY [1,1]\nASK_PROB h GIVEN () q",
        "ASK_PROB h GIVEN () q END",
    ] {
        assert!(
            parse_bytecode(&format!("{BASE}{bad}\nEND")).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn world_limit_applies_to_the_query_context_before_solving() {
    let mut body = String::new();
    for i in 0..8 {
        body.push_str(&format!(
            "EVENT e{i} [0,1]\nLIK e{i} POS GIVEN h 1/2 foreign [0,1]\n"
        ));
    }
    body.push_str("ASK_PROB h GIVEN () q");
    assert_eq!(solve(&body)["value"], "1/1");
    let scoped = body.replace("foreign", "q");
    assert_eq!(solve(&scoped)["code"], "POLYTOPE_DIMENSION_LIMIT");
}
