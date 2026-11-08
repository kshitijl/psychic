# Psychic shell integration for bash
# Add this to your ~/.bashrc:
#   eval "$(psychic bash)"
# or run `psychic init bash`, which does that for you.

# Tells psychic the integration is loaded, so the TUI does not suggest setting it up.
export PSYCHIC_SHELL=bash

# =============================================================================
# Hook to track directory changes
# =============================================================================

# bash has no chpwd hook, so this runs before every prompt and acts only when
# the directory has changed since the last one. Starting at the directory the
# shell opened in, so opening a shell is not counted as a visit.
__psychic_last_pwd="${PWD}"
__psychic_hook() {
    if [[ "${__psychic_last_pwd}" != "${PWD}" ]]; then
        __psychic_last_pwd="${PWD}"
        # In a subshell, so the shell prints no job number or "Done" line.
        (\command psychic track-visit "${PWD}" >/dev/null 2>&1 &)
    fi
}

if [[ ";${PROMPT_COMMAND:-};" != *";__psychic_hook;"* ]]; then
    PROMPT_COMMAND="__psychic_hook${PROMPT_COMMAND:+;${PROMPT_COMMAND}}"
fi

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
# quoted for the shell. Needs bash 4 or newer, for READLINE_LINE; macOS ships
# bash 3.2, where the binding is skipped.
__psychic_insert_path() {
    \builtin local result
    result="$(\command psychic --on-file-click=print-to-stdout --on-dir-click=print-to-stdout)" || return
    [[ -n "${result}" ]] || return
    [[ "${result}" == "${PWD}/"* ]] && result="${result#"${PWD}"/}"
    result="$(\builtin printf '%q ' "${result}")"
    READLINE_LINE="${READLINE_LINE:0:READLINE_POINT}${result}${READLINE_LINE:READLINE_POINT}"
    READLINE_POINT=$((READLINE_POINT + ${#result}))
}

# Bound to Ctrl-T only in an interactive bash 4+, and only where Ctrl-T still
# does what readline does by default, so an fzf (or anything else) binding is
# never taken over. Set PSYCHIC_NO_CTRL_T before loading this to skip it; bind
# the function yourself with, e.g.,
#   bind -x '"\C-x\C-f": __psychic_insert_path'
if [[ $- == *i* && -z "${PSYCHIC_NO_CTRL_T-}" && "${BASH_VERSINFO[0]}" -ge 4 ]]; then
    if ! \builtin bind -X 2>/dev/null | \command grep -q '^"\\C-t"' &&
        \builtin bind -p 2>/dev/null | \command grep -q '^"\\C-t": transpose-chars'; then
        \builtin bind -x '"\C-t": __psychic_insert_path'
    fi
fi
