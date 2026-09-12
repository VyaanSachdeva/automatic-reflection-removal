_unreflectanything() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local subcommands="train test inference download cache verify evaluate cite completion"
    local options="-h --help"
    COMPREPLY=($(compgen -W "$subcommands $options" -- "$cur"))
}

complete -F _unreflectanything unreflectanything unreflect ura
