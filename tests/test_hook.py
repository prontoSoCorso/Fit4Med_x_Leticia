"""The git-push hook blocks every `git push` and nothing else."""
import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "block_git_push.py"
BLOCKED = ["git push", "git push -f origin main", "git -C /some/dir push", 'git -C "/a b" push',
           "git --no-pager push", "git -c user.name=x push", "cd repo && git push", 'bash -c "git push"']
ALLOWED = ["git status", 'git commit -m "fix push button"', "git log --grep=push", "ls -la"]


def hook_exit_code(command):
    event = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run([sys.executable, str(HOOK)], input=event, text=True, capture_output=True).returncode


def test_hook_blocks_only_git_push():
    assert [c for c in BLOCKED if hook_exit_code(c) != 2] == []
    assert [c for c in ALLOWED if hook_exit_code(c) != 0] == []
