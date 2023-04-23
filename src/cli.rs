//
// This is the cli module for the aiha project.
//

use std::env;
use aiha::scan::Scan;

fn main() {
    let scan = Scan {};

    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("os") => println!("{}", scan.os_scan()),
        Some("arch") => println!("{}", scan.arch_scan()),
        _ => {
            println!("Usage: {} [os | arch]", args[0]);
            std::process::exit(1);
        }
    }
}
