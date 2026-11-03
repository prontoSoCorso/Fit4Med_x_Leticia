"""Claude Code hook: block every `git push` (also `git -C <dir> push`, global options, `&&` chains).

Claude Code sends each Bash command as JSON on stdin before running it.
Exit code 2 cancels the command; exit code 0 lets it run.
"""
import json
import re
import sys

command = json.load(sys.stdin).get("tool_input", {}).get("command", "")

VALUE = r"""("[^"]*"|'[^']*'|\S+)"""  # value of an option, quoted or not
# "git" as a word, then any global options (-C <dir>, -c <key=value>, --flag), then "push"
PUSH = rf"""(^|[\s;&|(`'"/])git(\s+(-C|-c|--git-dir|--work-tree)\s+{VALUE}|\s+-\S+)*\s+push\b"""

if re.search(PUSH, command):
    print("Blocked: Claude may commit but never push. Push by hand if needed.", file=sys.stderr)
    sys.exit(2)
