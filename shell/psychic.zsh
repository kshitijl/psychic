# Psychic shell integration for zsh
# Add this to your ~/.zshrc:
#   eval "$(psychic zsh)"

# =============================================================================
# Hook to track directory changes
# =============================================================================

# Hook function that runs after every directory change
function __psychic_hook() {
    \command psychic track-visit "$(\builtin pwd)" 2>/dev/null &!
}

# Initialize hook (add to chpwd_functions array)
\builtin typeset -ga chpwd_functions
# Remove any existing instances of our hook (avoid duplicates)
# shellcheck disable=SC2034,SC2296
chpwd_functions=("${(@)chpwd_functions:#__psychic_hook}")
# Add our hook
chpwd_functions+=(__psychic_hook)

# =============================================================================
# Navigation commands
# =============================================================================

# Jump to a directory using psychic
# Ctrl-J in psychic will print the directory and exit
p() {
    \builtin local result
    result="$(\command psychic --on-cwd-visit=print-to-stdout)" && \builtin cd -- "${result}"
}

# Jump to a directory using psychic (directories only)
# Enter on a directory will print it and exit (no navigation mode)
pd() {
    \builtin local result
    result="$(\command psychic --filter=dirs --on-dir-click=print-to-stdout)" && \builtin cd -- "${result}"
}

# Jump to a directory using psychic (current directory files only)
# Navigate within current dir, Ctrl-J to cd there
pc() {
    \builtin local result
    result="$(\command psychic --filter=cwd --on-cwd-visit=print-to-stdout)" && \builtin cd -- "${result}"
}
