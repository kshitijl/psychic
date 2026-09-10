use crate::db::ContextData;

/// What is worth recording about the session, which is very little.
///
/// This used to shell out three times per launch - `netstat` for the default
/// gateway, `ifconfig` for the subnet, and a DNS lookup - into columns that no
/// query, feature or view ever read. The idea was that a laptop's network would
/// stand in for "where am I, home or work", and nothing was ever built on it.
///
/// What is left needs no subprocess: the directory psychic was launched in,
/// which `is_under_cwd` and the visit features are computed against, and the
/// timezone, which is read from the environment.
pub fn gather_context() -> ContextData {
    ContextData {
        cwd: get_cwd(),
        timezone: get_timezone(),
    }
}

fn get_cwd() -> String {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| String::from("unknown"))
}

fn get_timezone() -> String {
    std::env::var("TZ").unwrap_or_else(|_| String::from("unknown"))
}
