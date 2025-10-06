use crate::db::ContextData;
use std::process::Command;

pub fn gather_context() -> ContextData {
    ContextData {
        cwd: get_cwd(),
        gateway: get_gateway(),
        subnet: get_subnet(),
        dns: get_dns(),
        shell_history: get_shell_history(),
        running_processes: get_running_processes(),
    }
}

fn get_cwd() -> String {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| String::from("unknown"))
}

fn get_gateway() -> String {
    // GATEWAY=$(netstat -nr | grep default | grep -v ':' | head -1 | awk '{print $2}')
    let output = Command::new("sh")
        .arg("-c")
        .arg("netstat -nr | grep default | grep -v ':' | head -1 | awk '{print $2}'")
        .output();

    match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => String::from("unknown"),
    }
}

fn get_subnet() -> String {
    // SUBNET=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d. -f1-2)
    let output = Command::new("sh")
        .arg("-c")
        .arg("ifconfig | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d. -f1-2")
        .output();

    match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => String::from("unknown"),
    }
}

fn get_dns() -> String {
    // DNS=$(scutil --dns | grep nameserver | head -1 | awk '{print $3}')
    let output = Command::new("sh")
        .arg("-c")
        .arg("scutil --dns | grep nameserver | head -1 | awk '{print $3}'")
        .output();

    match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => String::from("unknown"),
    }
}

fn get_shell_history() -> String {
    // Get last 10 commands from shell history
    // Try bash history first, then zsh
    let bash_output = Command::new("sh")
        .arg("-c")
        .arg("tail -10 ~/.bash_history 2>/dev/null")
        .output();

    if let Ok(out) = bash_output {
        let history = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !history.is_empty() {
            return history;
        }
    }

    // Try zsh history
    let zsh_output = Command::new("sh")
        .arg("-c")
        .arg("tail -10 ~/.zsh_history 2>/dev/null | cut -d';' -f2-")
        .output();

    match zsh_output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => String::from("unknown"),
    }
}

fn get_running_processes() -> String {
    // Get ps output - just user processes, not system ones
    let output = Command::new("ps")
        .arg("-u")
        .arg(std::env::var("USER").unwrap_or_else(|_| String::from("unknown")))
        .arg("-o")
        .arg("pid,comm")
        .output();

    match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => String::from("unknown"),
    }
}
