#!/usr/bin/env bash
#
# ODC-0002 baseline capture.
#
# Executes the procedure in docs/specs/ODC-0002-v2-baseline.md against a pinned
# revision inside a scratch tree created OUTSIDE the repository, then emits both
# deliverables:
#
#   docs/baselines/v2.0.4-environment.json   (normative, machine-consumed)
#   docs/baselines/v2.0.4.md                 (human rendering of the same model)
#
# The script never writes inside the working tree except for those two files.
# It never deletes the operator's .build/ or DerivedData/; it ignores them by
# construction because every measurement runs in the scratch tree.
#
# Usage:
#   scripts/capture-baseline.sh                      # cold pass, then warm pass
#   scripts/capture-baseline.sh --keep-scratch       # leave the scratch tree
#   scripts/capture-baseline.sh --reuse SCRATCH_DIR  # re-render from evidence
#   scripts/capture-baseline.sh --render-only DIR    # skip measurement entirely
#
set -uo pipefail

REVISION="${ODC_BASELINE_REVISION:-59da80b}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/docs/baselines"
KEEP_SCRATCH=0
REUSE=""
RENDER_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --keep-scratch) KEEP_SCRATCH=1; shift ;;
    --reuse) REUSE="$2"; KEEP_SCRATCH=1; shift 2 ;;
    --render-only) REUSE="$2"; RENDER_ONLY=1; KEEP_SCRATCH=1; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -n "$REUSE" ]; then
  SCRATCH="$REUSE"
else
  SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/odc0002.XXXXXX")"
fi
E="$SCRATCH/evidence"
mkdir -p "$E"

cleanup() {
  if [ "$KEEP_SCRATCH" -eq 0 ] && [ -d "$SCRATCH" ]; then
    rm -rf "$SCRATCH"
  fi
}
trap cleanup EXIT

spm() { swift package --cache-path "$SCRATCH/spm-cache" --scratch-path "$SCRATCH/.build" "$@"; }

measure() {
  echo "capture-baseline: scratch tree created outside the repository"

  # ---- step 1: populate the scratch tree from the pinned revision -----------
  ( cd "$REPO_ROOT" && git archive "$REVISION" ) | tar -x -C "$SCRATCH"
  mkdir -p "$SCRATCH/spm-cache" "$SCRATCH/.build"
  ( cd "$REPO_ROOT" && git rev-parse "$REVISION" ) > "$E/repo-revision.txt"
  ( cd "$REPO_ROOT" && git describe --always --dirty "$REVISION" ) > "$E/repo-describe.txt" 2>/dev/null \
    || ( cd "$REPO_ROOT" && git rev-parse --short "$REVISION" ) > "$E/repo-describe.txt"
  # "dirty" describes the measured tracked content, so untracked files (such as
  # the deliverables this run is about to write) are deliberately excluded.
  ( cd "$REPO_ROOT" && git status --porcelain --untracked-files=no ) > "$E/repo-status.txt"
  ( cd "$REPO_ROOT" && git status --porcelain --ignored ) > "$E/repo-status-ignored.txt"
  ( cd "$REPO_ROOT" && git clean -ndx ) > "$E/repo-clean-dry-run.txt"

  # ---- steps 2 and 3: pinned environment -----------------------------------
  swift --version > "$E/swift-version.txt" 2>&1
  xcodebuild -version > "$E/xcodebuild-version.txt" 2>&1
  xcodebuild -showsdks > "$E/xcodebuild-showsdks.txt" 2>&1
  xcode-select -p > "$E/xcode-select.txt" 2>&1
  sw_vers > "$E/sw-vers.txt" 2>&1
  sysctl -n hw.model machdep.cpu.brand_string hw.ncpu hw.memsize > "$E/sysctl.txt" 2>&1

  # ---- step 4: manifest only, no resolution --------------------------------
  ( cd "$SCRATCH" && spm tools-version ) > "$E/cold-tools-version.txt" 2>&1
  ( cd "$SCRATCH" && spm dump-package ) > "$E/cold-dump-package.json" 2>"$E/cold-dump-package.err"

  # ---- step 5: cold resolve, twice -----------------------------------------
  cp "$SCRATCH/Package.resolved" "$E/cold-resolved-before.json"
  shasum -a 256 "$SCRATCH/Package.resolved" | awk '{print $1}' > "$E/cold-resolved-before.sha"
  ( cd "$SCRATCH" && spm resolve ) > "$E/cold-resolve-1.log" 2>&1
  cp "$SCRATCH/Package.resolved" "$E/cold-resolved-after1.json"
  shasum -a 256 "$SCRATCH/Package.resolved" | awk '{print $1}' > "$E/cold-resolved-after1.sha"
  ( cd "$SCRATCH" && spm resolve ) > "$E/cold-resolve-2.log" 2>&1
  shasum -a 256 "$SCRATCH/Package.resolved" | awk '{print $1}' > "$E/cold-resolved-after2.sha"

  # ---- step 6: describe, post resolution -----------------------------------
  ( cd "$SCRATCH" && spm describe --type json ) > "$E/cold-describe.json" 2>"$E/cold-describe.err"

  # ---- step 7: artifact slices ---------------------------------------------
  XCF="$(find "$SCRATCH/.build/artifacts" -maxdepth 3 -name 'llama.xcframework' -type d | head -1)"
  echo "$XCF" > "$E/xcframework-path.txt"
  plutil -convert json -o "$E/xcf-info.json" "$XCF/Info.plist"
  : > "$E/xcf-slices.txt"
  for SLICE in $(python3 -c "
import json
d = json.load(open('$E/xcf-info.json'))
print(' '.join(lib['LibraryIdentifier'] for lib in d['AvailableLibraries']))
"); do
    A="$XCF/$SLICE/libllama_combined.a"
    {
      echo "SLICE $SLICE"
      echo "BYTES $(stat -f%z "$A")"
      echo "NMLINES $(nm -gU "$A" 2>/dev/null | wc -l | tr -d ' ')"
      echo "DEFINED $(nm -gU "$A" 2>/dev/null | grep -c '^[0-9a-f]\{8,\} [A-Za-z] ')"
      ar -t "$A" | sed 's/^/OBJECT /' 
    } >> "$E/xcf-slices.txt"
  done

  # ---- steps 8 to 11, cold, then steps 4 to 13 warm ------------------------
  SIM_SDK="$(xcrun --sdk iphonesimulator --show-sdk-path)"
  DEV_SDK="$(xcrun --sdk iphoneos --show-sdk-path)"

  run_cells() {
    local state="$1"
    ( cd "$SCRATCH" && swift build --cache-path "$SCRATCH/spm-cache" --scratch-path "$SCRATCH/.build" \
        --sdk "$SIM_SDK" --triple arm64-apple-ios17.0-simulator -c debug ) \
        > "$E/$state-build-ios-simulator.log" 2>&1
    echo $? > "$E/$state-build-ios-simulator.exit"
    ( cd "$SCRATCH" && swift build --cache-path "$SCRATCH/spm-cache" --scratch-path "$SCRATCH/.build" \
        --sdk "$DEV_SDK" --triple arm64-apple-ios17.0 -c debug ) \
        > "$E/$state-build-ios-device.log" 2>&1
    echo $? > "$E/$state-build-ios-device.exit"
    ( cd "$SCRATCH" && swift build --cache-path "$SCRATCH/spm-cache" --scratch-path "$SCRATCH/.build" -c debug ) \
        > "$E/$state-build-macos.log" 2>&1
    echo $? > "$E/$state-build-macos.exit"
    ( cd "$SCRATCH" && swift test --cache-path "$SCRATCH/spm-cache" --scratch-path "$SCRATCH/.build" ) \
        > "$E/$state-test-macos.log" 2>&1
    echo $? > "$E/$state-test-macos.exit"
    ( cd "$SCRATCH" && xcodebuild -project OnDeviceCatalyst.xcodeproj -scheme OnDeviceCatalyst \
        -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build ) \
        > "$E/$state-build-xcodeproj.log" 2>&1
    echo $? > "$E/$state-build-xcodeproj.exit"
  }

  run_cells cold

  # ---- step 12: source duplication -----------------------------------------
  ( cd "$SCRATCH" && python3 - "$E" <<'PYCENSUS'
import hashlib, json, os, subprocess, sys
evidence = sys.argv[1]
pkg, app = "Sources/OnDeviceCatalyst", "OnDeviceCatalyst"

def swift_files(root):
    found = set()
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith(".swift"):
                found.add(os.path.relpath(os.path.join(dirpath, name), root))
    return found

package_files, app_files = swift_files(pkg), swift_files(app)
rows = []
for rel in sorted(package_files & app_files):
    p, a = os.path.join(pkg, rel), os.path.join(app, rel)
    proc = subprocess.run(
        'diff "%s" "%s" | grep -c \'^[<>]\'' % (p, a),
        shell=True, capture_output=True, text=True)
    changed = int(proc.stdout.strip() or 0)
    ps = hashlib.sha256(open(p, "rb").read()).hexdigest()
    as_ = hashlib.sha256(open(a, "rb").read()).hexdigest()
    rows.append({"path": rel, "identical": changed == 0, "changed_lines": changed,
                 "package_sha256": ps, "app_sha256": as_})
rows.sort(key=lambda r: (-r["changed_lines"], r["path"]))
json.dump({
    "shared": rows,
    "package_only": sorted(package_files - app_files),
    "app_only": sorted(app_files - package_files),
    "package_file_count": len(package_files),
    "app_file_count": len(app_files),
}, open(os.path.join(evidence, "duplicate-sources.json"), "w"), indent=1)
PYCENSUS
  )

  ( cd "$SCRATCH" && grep -c 'XCRemoteSwiftPackageReference' OnDeviceCatalyst.xcodeproj/project.pbxproj ) \
    > "$E/xcremote-count.txt" 2>/dev/null || echo 0 > "$E/xcremote-count.txt"
  ( cd "$SCRATCH" && grep -rn '#if !targetEnvironment' Sources/ | wc -l | tr -d ' ' ) \
    > "$E/negated-simulator-guard-count.txt"
  ( cd "$SCRATCH" && grep -rn '#if targetEnvironment(simulator)' Sources/ ) \
    > "$E/simulator-guard-sites.txt" 2>/dev/null
  ( cd "$SCRATCH" && grep -n 'func test' Tests/OnDeviceCatalystTests/OnDeviceCatalystTests.swift ) \
    > "$E/test-cases.txt"
  ( cd "$SCRATCH" && find Tests -maxdepth 1 -name '*.swift' | sort ) > "$E/orphaned-tests.txt"

  # ---- step 15: warm pass ---------------------------------------------------
  # The lockfile is restored to its archived content first, so the warm lockfile
  # measurement answers the same question the cold one did. That restores a
  # tracked file inside the scratch tree only, and changes no cache state.
  cp "$E/cold-resolved-before.json" "$SCRATCH/Package.resolved"
  ( cd "$SCRATCH" && spm tools-version ) > "$E/warm-tools-version.txt" 2>&1
  ( cd "$SCRATCH" && spm dump-package ) > "$E/warm-dump-package.json" 2>&1
  shasum -a 256 "$SCRATCH/Package.resolved" | awk '{print $1}' > "$E/warm-resolved-before.sha"
  ( cd "$SCRATCH" && spm resolve ) > "$E/warm-resolve-1.log" 2>&1
  shasum -a 256 "$SCRATCH/Package.resolved" | awk '{print $1}' > "$E/warm-resolved-after1.sha"
  ( cd "$SCRATCH" && spm resolve ) > "$E/warm-resolve-2.log" 2>&1
  shasum -a 256 "$SCRATCH/Package.resolved" | awk '{print $1}' > "$E/warm-resolved-after2.sha"
  ( cd "$SCRATCH" && spm describe --type json ) > "$E/warm-describe.json" 2>&1
  run_cells warm
}

if [ "$RENDER_ONLY" -eq 0 ]; then
  measure
fi

mkdir -p "$OUT_DIR"
python3 "$REPO_ROOT/scripts/render-baseline.py" --evidence "$E" --scratch "$SCRATCH" --out "$OUT_DIR"
status=$?
if [ $status -ne 0 ]; then
  echo "capture-baseline: rendering failed" >&2
  exit $status
fi

echo "capture-baseline: wrote docs/baselines/v2.0.4-environment.json and docs/baselines/v2.0.4.md"
if [ "$KEEP_SCRATCH" -eq 1 ]; then
  echo "capture-baseline: scratch tree retained (path deliberately not echoed into any deliverable)"
fi
