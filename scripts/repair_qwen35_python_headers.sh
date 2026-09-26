#!/usr/bin/env bash
set -e
B="${QWEN35_BUNDLE:-/workspace/tuzhitianwang/qwen35_ascend_server_bundle}"
V="${QWEN35_VENV:-/workspace/tuzhitianwang/.venv-qwen35-checked}"
H="$V/.native-python310"
test "$(uname -m)" = aarch64
PV=$("$V/bin/python" -c 'import platform, sys; assert sys.version_info[:2] == (3, 10); print(platform.python_version())')
T=$(mktemp -d "$V/python-headers-download.XXXXXX")
cd "$T"
if ! apt-get download libpython3.10-dev:arm64; then
  apt-get update
  apt-get download libpython3.10-dev:arm64
fi
debs=(libpython3.10-dev_*.deb)
test "${#debs[@]}" = 1
DEB="${debs[0]}"
test "$(dpkg-deb -f "$DEB" Architecture)" = arm64
PKG=$(dpkg-deb -f "$DEB" Version)
case "${PKG#*:}" in
  "$PV"-*) ;;
  *) printf 'STOP: Python %s does not match header package %s\n' "$PV" "$PKG"; exit 1 ;;
esac
dpkg-deb -x "$DEB" "$H"
test -f "$H/usr/include/python3.10/Python.h"
test -f "$H/usr/include/aarch64-linux-gnu/python3.10/pyconfig.h"
export CPATH="$H/usr/include/python3.10:$H/usr/include${CPATH:+:$CPATH}"
printf '#include <Python.h>\nstatic_assert(PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 10, "Python version");\nstatic_assert(sizeof(void*) == 8, "64-bit target");\n' | c++ -x c++ -fsyntax-only -
printf 'PASS python_headers (%s)\n' "$PKG"
source /usr/local/Ascend/cann/set_env.sh
export NON_MEGATRON=true
cd "$B/third_party/MindSpeed-MM"
"$V/bin/python" "$B/scripts/audit_qwen35_runtime.py" --probe triton_ops
R="$B/results/ascend_qwen35/environment/headers-recheck-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$R"
set +e
"$V/bin/python" "$B/scripts/audit_qwen35_runtime.py" --output "$R/dependency_audit.json"
code=$?
set -e
if [ "$code" != 0 ]; then
  "$V/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({"failed_checks":r.get("failed_checks"), "probes":{k:v for k,v in r.get("probes",{}).items() if not v.get("passed")}, "dependencies":r.get("dependencies",{}).get("unexpected")},ensure_ascii=False,indent=2))' "$R/dependency_audit.json"
fi
printf 'Header search path for subsequent training: %s\n' "$CPATH"
exit "$code"
