use std::{env, fs, process};

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: l0 <file.l0>"); process::exit(2);
    });
    if let Err(e) = fs::metadata(&path) { eprintln!("{path}: {e}"); process::exit(2); }
    match l0::execute_interactive_file(&path) {
        Ok(()) => {},
        Err(error) => { eprintln!("{path}: {error}"); process::exit(1); }
    }
}
