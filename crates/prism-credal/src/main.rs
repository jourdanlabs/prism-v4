use std::io::{self, Write};

use prism_credal::{Program, Verdict, solve_credal};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let verdict = match serde_json::from_reader(io::stdin().lock()) {
        Ok(value) => match Program::from_json(value) {
            Ok(program) => solve_credal(&program),
            Err(error) => parse_refusal(error),
        },
        Err(error) => parse_refusal(error),
    };
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &verdict)?;
    writeln!(stdout)?;
    Ok(())
}

fn parse_refusal(error: impl std::fmt::Display) -> Verdict {
    Verdict::Refused {
        code: "PARSE_ERROR",
        details: json!({"detail": error.to_string()}),
    }
}
