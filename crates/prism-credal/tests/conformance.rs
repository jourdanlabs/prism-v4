use std::{
    collections::HashSet,
    io::Write,
    process::{Command, Stdio},
};

use prism_credal::{Program, bound_polytope};
use serde_json::{Value, json};

fn cli(program: &Value) -> Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-credal"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(child.stdin.take().unwrap(), program).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    serde_json::from_slice(&output.stdout).unwrap()
}

fn compare(actual: &Value, reference: &Value, case: &str) {
    assert_eq!(actual["status"], reference["status"], "{case}: {actual}");
    if reference["status"] == "ANSWER" {
        for field in ["value", "interval", "target", "via"] {
            assert_eq!(actual[field], reference[field], "{case}: {field}");
        }
        assert_eq!(
            json!([actual["credal"]["eMin"], actual["credal"]["eMax"]]),
            reference["support"],
            "{case}: evidence support"
        );
    } else {
        assert_eq!(actual["code"], reference["code"], "{case}: {actual}");
        assert_eq!(
            actual["details"]["interval"], reference["interval"],
            "{case}: refusal interval"
        );
    }
}

#[test]
fn reference_bench_matches_all_programs_and_trap_families() {
    let output = Command::new("node")
        .args(["tests/reference.mjs"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Node is required for the JavaScript benchmark");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut solvable = 0;
    let mut kernel_traps = 0;
    let mut gate_traps = 0;
    let mut trap_families = HashSet::new();
    let stdout = String::from_utf8(output.stdout).unwrap();
    for line in stdout.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let case = row["id"].as_str().unwrap();
        let bytecode = cli(&row["program"]);
        let ast = cli(&row["ast"]);
        assert_eq!(bytecode, ast, "{case}: input encodings");
        compare(&bytecode, &row["kernel"], case);
        let program = Program::from_json(row["ast"].clone()).unwrap();
        compare(
            &serde_json::to_value(bound_polytope(&program)).unwrap(),
            &row["polytope"],
            case,
        );
        match row["kind"].as_str().unwrap() {
            "solve" => {
                compare(&bytecode, &row["withText"], case);
                assert_eq!(bytecode["value"], row["expected"]["rational"], "{case}");
                solvable += 1;
            }
            "trap" => {
                compare(&bytecode, &row["withText"], case);
                assert_eq!(bytecode["code"], row["expected"]["code"], "{case}");
                assert!(trap_families.insert(row["trapFamily"].as_str().unwrap().to_owned()));
                kernel_traps += 1;
            }
            "gate" => {
                compare(&bytecode, &row["withText"], case);
                assert_eq!(row["gate"], row["expected"], "{case}: invariance gate");
                assert!(trap_families.insert(row["trapFamily"].as_str().unwrap().to_owned()));
                gate_traps += 1;
            }
            kind => panic!("unexpected benchmark row kind {kind}"),
        }
    }
    assert_eq!(
        (solvable, kernel_traps, gate_traps, trap_families.len()),
        (200, 13, 1, 14)
    );
}

#[test]
fn cli_returns_parse_errors_as_json() {
    assert_eq!(
        cli(&json!({"decls": [], "query": {}}))["code"],
        "PARSE_ERROR"
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-credal"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let verdict: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(verdict["code"], "PARSE_ERROR");
}
