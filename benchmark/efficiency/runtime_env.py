"""Runtime environment helpers for project-local benchmark entry points."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def prepend_interpreter_bin_to_path() -> str:
    """Expose companion tools installed beside the running Python executable.

    Invoking ``.venv/bin/python`` does not activate a virtual environment. uv
    venvs also symlink Python to a shared base interpreter, so the symlink must
    remain unresolved: tools such as ninja live in the project ``.venv/bin``.
    """

    interpreter_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_bin, *[entry for entry in path_entries if entry != interpreter_bin]]
    )
    return interpreter_bin
