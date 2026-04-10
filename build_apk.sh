#!/usr/bin/env bash
# Build a debug APK for FoodScan. Safe even if `.venv` is activated: we drop
# venv from PATH and unset VIRTUAL_ENV so p4a never runs `pip install --user`
# inside a venv.
#
# Prerequisites (Homebrew):
#   brew install autoconf automake libtool pkg-config cmake openssl@3
#
# Then:
#   chmod +x build_apk.sh && ./build_apk.sh

set -euo pipefail
cd "$(dirname "$0")"

P4A_PY=".buildozer/android/platform/python-for-android/pythonforandroid/prerequisites.py"
if [[ -f "$P4A_PY" ]] && grep -q 'openssl@1.1' "$P4A_PY" 2>/dev/null; then
  echo "Patching p4a: openssl@1.1 -> openssl@3 (Homebrew no longer ships 1.1)"
  sed -i '' 's/openssl@1.1/openssl@3/g' "$P4A_PY"
fi

# Strip project .venv from PATH so subprocesses never use venv pip (--user breaks there).
_newpath=""
IFS=':'
for _p in ${PATH:-}; do
  case "$_p" in
    *"/.venv/bin" | *"/.venv" ) ;;
    *) _newpath="${_newpath:+${_newpath}:}${_p}" ;;
  esac
done
unset IFS
export PATH="/usr/bin:/bin:/opt/homebrew/bin:/opt/homebrew/sbin:${_newpath}"
unset VIRTUAL_ENV

exec /usr/bin/python3 -m buildozer android debug "$@"
