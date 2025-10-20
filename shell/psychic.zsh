# Psychic shell integration for zsh
# Add this to your ~/.zshrc:
#   eval "$(psychic zsh)"

# Jump to a directory using psychic
p() {
    \builtin local result
    result="$(\command psychic --shell-integration)" && \builtin cd -- "${result}"
}

# Jump to a directory using psychic (directories only)
pd() {
    \builtin local result
    result="$(\command psychic --shell-integration --filter=dirs)" && \builtin cd -- "${result}"
}
