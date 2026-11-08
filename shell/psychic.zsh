# Psychic shell integration for zsh
# Add this to your ~/.zshrc:
#   eval "$(psychic zsh)"
# or run `psychic init`, which does that for you.

# Tells psychic the integration is loaded, so the TUI does not suggest setting it up.
export PSYCHIC_SHELL=zsh

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
# `pd xy`: cd to xy if it is a directory, else to the best match for xy without
#   opening the TUI (`psychic jump`); if nothing matches, open it with xy typed in
# `pd -i xy`: open the TUI with xy typed in
pd() {
    \builtin local result
    if [[ "$1" == -i ]]; then
        \builtin shift
        result="$(\command psychic --filter=dirs --on-dir-click=print-to-stdout --query="$*")"
    elif (( $# == 0 )); then
        result="$(\command psychic --filter=dirs --on-dir-click=print-to-stdout)"
    elif [[ $# -eq 1 && -d "$1" ]]; then
        result="$1"
    else
        result="$(\command psychic jump -- "$*")" ||
            result="$(\command psychic --filter=dirs --on-dir-click=print-to-stdout --query="$*")"
    fi && \builtin cd -- "${result}"
}

# Jump to a directory using psychic (current directory files only)
# Navigate within current dir, Ctrl-J to cd there
pc() {
    \builtin local result
    result="$(\command psychic --filter=cwd --on-cwd-visit=print-to-stdout)" && \builtin cd -- "${result}"
}

# =============================================================================
# Ctrl-T: insert a path picked in psychic at the cursor
# =============================================================================

# Enter on a file or directory prints it instead of opening it. A pick under
# the current directory is inserted relative to it, anything else absolute,
# quoted for the shell.
__psychic_insert_path() {
    \builtin local result
    result="$(\command psychic --on-file-click=print-to-stdout --on-dir-click=print-to-stdout)"
    if [[ -n "${result}" ]]; then
        [[ "${result}" == "${PWD}/"* ]] && result="${result#"${PWD}"/}"
        LBUFFER+="${(q)result} "
    fi
    \builtin zle reset-prompt
}
\builtin zle -N __psychic_insert_path

# Bound to Ctrl-T only where Ctrl-T still does what zsh does by default, so an
# fzf (or anything else) binding is never taken over. Set PSYCHIC_NO_CTRL_T
# before loading this to skip it; bind the widget yourself with, e.g.,
#   bindkey '^X^F' __psychic_insert_path
if [[ -z "${PSYCHIC_NO_CTRL_T-}" ]]; then
    for __psychic_keymap in emacs viins; do
        case "$(\builtin bindkey -M "${__psychic_keymap}" '^T' 2>/dev/null)" in
            *transpose-chars|*undefined-key|*self-insert|'')
                \builtin bindkey -M "${__psychic_keymap}" '^T' __psychic_insert_path ;;
        esac
    done
    \builtin unset __psychic_keymap
fi
