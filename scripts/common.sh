# shellcheck shell=bash
# Sourced by the pipeline scripts after ROOT_DIR is set.
# Supports macOS, Linux/WSL and Git Bash on Windows.

# UTF-8 file/stdio encoding regardless of the Windows code page (e.g. cp949).
export PYTHONUTF8=1

# venv layout: bin/python (POSIX) or Scripts/python.exe (Windows).
PYTHON="$ROOT_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"

# Python output parsed by the shell. On Windows, text-mode stdout ends lines
# with \r\n; strip \r. The exit status of python is kept (pipefail).
py() { "$PYTHON" "$@" | tr -d '\r'; }
