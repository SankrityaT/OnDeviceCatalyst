#!/bin/sh
# ODC-0004: runner for the v2 characterization suite.
#
# Implements the runner design in
# docs/specs/ODC-0004-v2-characterization-suite.md, "## Design", "Why the
# runner is SwiftPM plus simctl rather than xcodebuild", and "## Interfaces",
# "### Runner": swift build --build-tests for the simulator triple, repackage
# SwiftPM's macOS-shaped test bundle into a flat iOS .xctest bundle, spawn the
# platform xctest agent via simctl, parse the textual XCTest log into an
# executed/skipped ledger, and fail if any expected-executed case did not
# execute (harness-defect, per the spec's "## Failure behavior").
#
# Usage:
#   scripts/run-characterization.sh --surface simulator [--device-udid <udid>]
#   scripts/run-characterization.sh --surface device --destination-id <id>
#
# `--surface device` is a documented but unimplemented entry point (spec
# "## Interfaces", "### Runner": Q3 is open). Invoking it fails loudly rather
# than silently no-op-ing.

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# shellcheck source=support/parse-xctest-log.sh
. "$ROOT/scripts/support/parse-xctest-log.sh"

SURFACE=""
DEVICE_UDID=""
DESTINATION_ID=""
LEDGER_OUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --surface) SURFACE="$2"; shift 2 ;;
    --device-udid) DEVICE_UDID="$2"; shift 2 ;;
    --destination-id) DESTINATION_ID="$2"; shift 2 ;;
    --ledger-out) LEDGER_OUT="$2"; shift 2 ;;
    *) echo "run-characterization: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$SURFACE" ]; then
  echo "run-characterization: --surface simulator|device is required" >&2
  exit 2
fi

if [ "$SURFACE" = "device" ]; then
  # Q3 in the spec: the device-execution mechanism (signing, provisioning,
  # entitlements, a deployment tool) is not measured by this ticket. This
  # entry point exists so invoking it fails loudly and names the gap, rather
  # than silently doing nothing.
  echo "run-characterization: harness-defect: --surface device is a declared" >&2
  echo "  but unimplemented entry point (spec Q3). No code-signing," >&2
  echo "  provisioning, or device-deployment mechanism exists in this ticket." >&2
  echo "  destination-id=${DESTINATION_ID:-<unset>}" >&2
  exit 1
fi

if [ "$SURFACE" != "simulator" ]; then
  echo "run-characterization: --surface must be 'simulator' or 'device', got '$SURFACE'" >&2
  exit 2
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"; if [ -n "$CREATED_UDID" ]; then xcrun simctl delete "$CREATED_UDID" >/dev/null 2>&1 || true; fi' EXIT INT TERM

STATUS_BEFORE=$(git -C "$ROOT" status --porcelain)

CREATED_UDID=""
UDID="$DEVICE_UDID"
if [ -z "$UDID" ]; then
  RUNTIME=$(xcrun simctl list runtimes available 2>/dev/null | grep -i "iOS " | tail -1 | sed -E 's/.*(com\.apple\.CoreSimulator\.SimRuntime\.[A-Za-z0-9.-]+).*/\1/')
  if [ -z "$RUNTIME" ]; then
    echo "run-characterization: harness-defect: no available iOS simulator runtime found" >&2
    exit 1
  fi
  DEVICETYPE="com.apple.CoreSimulator.SimDeviceType.iPhone-16"
  UDID=$(xcrun simctl create "ODC-0004-characterization-$$" "$DEVICETYPE" "$RUNTIME")
  CREATED_UDID="$UDID"
  xcrun simctl boot "$UDID" >/dev/null 2>&1 || true
  xcrun simctl bootstatus "$UDID" -b >/dev/null 2>&1 || true
fi

echo "run-characterization: surface=simulator udid=$UDID"

SIM_SDK=$(xcrun --sdk iphonesimulator --show-sdk-path)
BUILD_LOG="$WORK/build.log"

echo "run-characterization: swift build --build-tests (this is the suite's one build; see spec memory-pressure note)"
if ! swift build --build-tests \
    --sdk "$SIM_SDK" \
    --triple arm64-apple-ios17.0-simulator \
    -c debug \
    > "$BUILD_LOG" 2>&1; then
  echo "run-characterization: harness-defect: swift build --build-tests failed" >&2
  tail -80 "$BUILD_LOG" >&2
  exit 1
fi

TEST_BUNDLE_MACHO=$(find "$ROOT/.build/arm64-apple-ios-simulator/debug" -maxdepth 1 -name "*.xctest" -print -quit)
if [ -z "$TEST_BUNDLE_MACHO" ]; then
  echo "run-characterization: harness-defect: no .xctest product found after build" >&2
  exit 1
fi

BUNDLE_NAME=$(basename "$TEST_BUNDLE_MACHO")
BINARY_NAME="${BUNDLE_NAME%.xctest}"
FLAT_BUNDLE="$WORK/$BUNDLE_NAME"
mkdir -p "$FLAT_BUNDLE"

# SwiftPM emits a macOS-shaped bundle (Contents/MacOS/<binary>, no Info.plist)
# even for a simulator triple (spec N10). Repackage into a flat iOS bundle.
if [ -f "$TEST_BUNDLE_MACHO/Contents/MacOS/$BINARY_NAME" ]; then
  cp "$TEST_BUNDLE_MACHO/Contents/MacOS/$BINARY_NAME" "$FLAT_BUNDLE/$BINARY_NAME"
elif [ -f "$TEST_BUNDLE_MACHO/$BINARY_NAME" ]; then
  cp "$TEST_BUNDLE_MACHO/$BINARY_NAME" "$FLAT_BUNDLE/$BINARY_NAME"
else
  echo "run-characterization: harness-defect: could not locate the test binary inside $TEST_BUNDLE_MACHO" >&2
  exit 1
fi

INFO_PLIST_SRC="$ROOT/scripts/support/characterization-test-bundle-Info.plist"
if [ ! -f "$INFO_PLIST_SRC" ]; then
  echo "run-characterization: harness-defect: missing tracked $INFO_PLIST_SRC build input" >&2
  exit 1
fi
sed "s/__BINARY_NAME__/$BINARY_NAME/g" "$INFO_PLIST_SRC" > "$FLAT_BUNDLE/Info.plist"

XCTEST_AGENT="$(xcode-select -p)/Platforms/iPhoneSimulator.platform/Developer/Library/Xcode/Agents/xctest"
if [ ! -x "$XCTEST_AGENT" ]; then
  echo "run-characterization: harness-defect: platform xctest agent not found at $XCTEST_AGENT" >&2
  exit 1
fi

RAW_LOG="$WORK/xctest-run.log"
echo "run-characterization: spawning platform xctest agent"
set +e
ODC_CHARACTERIZATION_SURFACE=simulator \
  xcrun simctl spawn "$UDID" "$XCTEST_AGENT" -XCTest All "$FLAT_BUNDLE" > "$RAW_LOG" 2>&1
XCTEST_EXIT=$?
set -e

echo "run-characterization: xctest agent exited $XCTEST_EXIT"

# ---------------------------------------------------------------------------
# Parse the textual XCTest log into an executed/skipped ledger.
# ---------------------------------------------------------------------------

LEDGER="$WORK/ledger.txt"
parse_xctest_log "$RAW_LOG" "$LEDGER"

EXECUTED_COUNT=$(grep -c "^EXECUTED " "$LEDGER" || true)
SKIPPED_COUNT=$(grep -c "^SKIPPED " "$LEDGER" || true)

echo "run-characterization: ledger: $EXECUTED_COUNT executed, $SKIPPED_COUNT skipped"

if [ -n "$LEDGER_OUT" ]; then
  cp "$LEDGER" "$LEDGER_OUT"
  echo "run-characterization: ledger written to $LEDGER_OUT"
else
  echo "--- ledger ---"
  cat "$LEDGER"
  echo "--- end ledger ---"
fi

STATUS_AFTER=$(git -C "$ROOT" status --porcelain)
if [ "$STATUS_BEFORE" != "$STATUS_AFTER" ]; then
  echo "run-characterization: mutation: the working tree changed during this run" >&2
  exit 1
fi

if [ "$EXECUTED_COUNT" -eq 0 ]; then
  echo "run-characterization: harness-defect: zero cases executed" >&2
  exit 1
fi

exit 0
