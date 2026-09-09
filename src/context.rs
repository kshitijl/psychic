use crate::db::ContextData;
use std::process::Command;

pub fn gather_context() -> ContextData {
    ContextData {
        cwd: get_cwd(),
        gateway: get_gateway(),
        subnet: get_subnet(),
        dns: get_dns(),
        timezone: get_timezone(),
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

fn get_timezone() -> String {
    // Try TZ environment variable first
    if let Ok(tz) = std::env::var("TZ")
        && !tz.is_empty()
    {
        return tz;
    }

    // Try reading /etc/localtime symlink on Unix systems
    if let Ok(link) = std::fs::read_link("/etc/localtime")
        && let Some(tz_path) = link.to_str()
    {
        // Extract timezone from path like /usr/share/zoneinfo/America/Los_Angeles
        if let Some(tz) = tz_path.strip_prefix("/usr/share/zoneinfo/") {
            return tz.to_string();
        }
        if let Some(tz) = tz_path.strip_prefix("/var/db/timezone/zoneinfo/") {
            return tz.to_string();
        }
    }

    // Fallback to UTC
    String::from("UTC")
}
