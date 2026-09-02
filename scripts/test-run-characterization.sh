#!/bin/sh
# ODC-0004: self-test for scripts/run-characterization.sh, following
# scripts/test-check-dco.sh's pattern.
#
# Does NOT invoke a real build or simulator run -- the spec's memory-pressure
# note requires building once, not repeatedly, and the real build already
# happens once via scripts/run-characterization.sh --surface simulator
# itself. Instead this exercises, in isolation:
#   1. Argument validation (missing/unknown --surface).
#   2. The declared-but-unimplemented --surface device entry point failing
#      loudly (harness-defect) rather than silently no-op-ing (spec Q3).
#   3. The textual-XCTest-log-to-ledger parser, against synthetic fixtures,
#      including the "zero executed" and "all skipped" cases the ledger
#      audit must catch.

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNNER="$ROOT/scripts/run-characterization.sh"
PARSER="$ROOT/scripts/support/parse-xctest-log.sh"

fail() {
  echo "test-run-characterization: FAIL: $1" >&2
  exit 1
}

# 1. Missing --surface.
set +e
"$RUNNER" >/tmp/odc0004-missing-surface.log 2>&1
STATUS=$?
set -e
[ "$STATUS" -eq 2 ] || fail "missing --surface should exit 2, got $STATUS"

# 2. Unknown --surface.
set +e
"$RUNNER" --surface bogus >/tmp/odc0004-bogus-surface.log 2>&1
STATUS=$?
set -e
[ "$STATUS" -eq 2 ] || fail "--surface bogus should exit 2, got $STATUS"

# 3. --surface device fails loudly (harness-defect), not silently.
set +e
OUTPUT=$("$RUNNER" --surface device --destination-id fake-id 2>&1)
STATUS=$?
set -e
[ "$STATUS" -ne 0 ] || fail "--surface device should fail (Q3 is unimplemented)"
case "$OUTPUT" in
  *harness-defect*) : ;;
  *) fail "--surface device failure did not mention harness-defect" ;;
esac

# 4. Log parser: a representative synthetic XCTest log.
. "$PARSER"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

cat > "$WORK/mixed.log" <<'EOF'
Test Suite 'All tests' started at 2026-09-02 00:00:00.000.
Test Case '-[OnDeviceCatalystTests test_requires_example]' started.
Test Case '-[OnDeviceCatalystTests test_requires_example]' passed (0.010 seconds).
Test Case '-[OnDeviceCatalystTests test_characterizes_something__ODC_0099]' started.
Test Case '-[OnDeviceCatalystTests test_characterizes_something__ODC_0099]' failed (0.020 seconds).
Test Case '-[D2GenerationCharacterizationTests test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011]' started.
SKIP[requires-device] this case needs a physical device with the real llama.cpp slice
Test Case '-[D2GenerationCharacterizationTests test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011]' skipped (0.001 seconds).
Test Suite 'All tests' finished at 2026-09-02 00:00:01.000.
EOF

parse_xctest_log "$WORK/mixed.log" "$WORK/mixed.ledger"

EXECUTED=$(grep -c "^EXECUTED " "$WORK/mixed.ledger" || true)
SKIPPED=$(grep -c "^SKIPPED " "$WORK/mixed.ledger" || true)

[ "$EXECUTED" -eq 2 ] || fail "expected 2 executed lines, got $EXECUTED"
[ "$SKIPPED" -eq 1 ] || fail "expected 1 skipped line, got $SKIPPED"
grep -q "^SKIPPED test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011 SKIP\[requires-device\]$" "$WORK/mixed.ledger" \
  || fail "skip code was not captured correctly"

# 5. Zero-executed / all-skipped synthetic log: the parser itself just
# records what it saw; check-characterization.py --skips is what refuses to
# pass a zero-executed or all-skipped run. Verify that refusal here too.
cat > "$WORK/all-skipped.log" <<'EOF'
Test Case '-[Suite test_characterizes_x__ODC_0099]' started.
SKIP[requires-device] nope
Test Case '-[Suite test_characterizes_x__ODC_0099]' skipped (0.001 seconds).
EOF
parse_xctest_log "$WORK/all-skipped.log" "$WORK/all-skipped.ledger"

set +e
python3 "$ROOT/scripts/check-characterization.py" --skips "$WORK/all-skipped.ledger" >/tmp/odc0004-all-skipped-check.log 2>&1
STATUS=$?
set -e
[ "$STATUS" -ne 0 ] || fail "an all-skipped ledger must fail check-characterization.py --skips"
grep -q "all-skipped" /tmp/odc0004-all-skipped-check.log || fail "expected an all-skipped failure message"

cat > "$WORK/empty.log" <<'EOF'
Test Suite 'All tests' started.
Test Suite 'All tests' finished.
EOF
parse_xctest_log "$WORK/empty.log" "$WORK/empty.ledger"
set +e
python3 "$ROOT/scripts/check-characterization.py" --skips "$WORK/empty.ledger" >/tmp/odc0004-empty-check.log 2>&1
STATUS=$?
set -e
[ "$STATUS" -ne 0 ] || fail "a zero-executed ledger must fail check-characterization.py --skips"
grep -q "zero-executed" /tmp/odc0004-empty-check.log || fail "expected a zero-executed failure message"

echo "test-run-characterization: all checks passed"
