# Psychic shell integration for fish
# Add this to your ~/.config/fish/config.fish:
#   psychic fish | source
# or run `psychic init fish`, which does that for you.

# Tells psychic the integration is loaded, so the TUI does not suggest setting it up.
set -gx PSYCHIC_SHELL fish

# =============================================================================
# Hook to track directory changes
# =============================================================================

# Runs whenever $PWD changes. Disowned, so fish prints no "Job has ended" line.
function __psychic_hook --on-variable PWD
    command psychic track-visit "$PWD" >/dev/null 2>&1 &
    disown 2>/dev/null
end

# =============================================================================
# Navigation commands
# =============================================================================

# Jump to a directory using psychic
# Ctrl-J in psychic will print the directory and exit
function p
    set -l result (command psychic --on-cwd-visit=print-to-stdout)
    and builtin cd -- $result
end

# Jump to a directory using psychic (directories only)
# Enter on a directory will print it and exit (no navigation mode)
# `pd xy`: cd to xy if it is a directory, else to the best match for xy without
#   opening the TUI (`psychic jump`); if nothing matches, open it with xy typed in
# `pd -i xy`: open the TUI with xy typed in
function pd
    set -l result
    if test "$argv[1]" = -i
        set result (command psychic --filter=dirs --on-dir-click=print-to-stdout --query="$argv[2..-1]")
    else if test (count $argv) -eq 0
        set result (command psychic --filter=dirs --on-dir-click=print-to-stdout)
    else if test (count $argv) -eq 1; and test -d "$argv[1]"
        set result $argv[1]
    else
        set result (command psychic jump -- "$argv")
        or set result (command psychic --filter=dirs --on-dir-click=print-to-stdout --query="$argv")
    end
    and builtin cd -- $result
end

# Jump to a directory using psychic (current directory files only)
# Navigate within current dir, Ctrl-J to cd there
function pc
    set -l result (command psychic --filter=cwd --on-cwd-visit=print-to-stdout)
    and builtin cd -- $result
end

# =============================================================================
# Ctrl-T: insert a path picked in psychic at the cursor
# =============================================================================

# Enter on a file or directory prints it instead of opening it. A pick under
# the current directory is inserted relative to it, anything else absolute,
# quoted for the shell.
function __psychic_insert_path
    set -l result (command psychic --on-file-click=print-to-stdout --on-dir-click=print-to-stdout)
    if test -n "$result"
        if string match -q -- "$PWD/*" $result
            set result (string sub --start (math (string length -- "$PWD/") + 1) -- $result)
        end
        commandline --insert -- (string escape -- $result)" "
    end
    commandline --function repaint
end

# Bound to Ctrl-T only where nothing else (fzf, say) has claimed it. Set
# PSYCHIC_NO_CTRL_T before loading this to skip it; bind the function yourself
# with, e.g.,
#   bind \cx\cf __psychic_insert_path
if not set -q PSYCHIC_NO_CTRL_T
    set -l __psychic_current (bind \ct 2>/dev/null)
    if test -z "$__psychic_current"; or string match -q -- '*transpose-chars*' $__psychic_current
        bind \ct __psychic_insert_path
        bind -M insert \ct __psychic_insert_path
    end
end
