#!/bin/sh
# ODC-0004: shared log-parsing logic for scripts/run-characterization.sh and
# its self-test, scripts/test-run-characterization.sh. Kept as a standalone,
# sourceable file specifically so the self-test can exercise the parsing
# logic against a synthetic log without invoking a real build or simulator
# run (see spec's memory-pressure note: build once, not repeatedly).
#
# parse_xctest_log <raw-log-path> <ledger-out-path>
#
# Recognizes lines of the textual XCTest protocol shape:
#   Test Case '-[Suite testName]' started.
#   Test Case '-[Suite testName]' passed (0.001 seconds).
#   Test Case '-[Suite testName]' failed (0.001 seconds).
#   Test Case '-[Suite testName]' skipped (0.001 seconds).
# The class portion of "Suite" is module-qualified (e.g.
# "OnDeviceCatalystTests.D8InitializationFailureCharacterizationTests") when
# run through the platform xctest agent, so the name pattern allows ".".
#
# For a skipped case, the SKIP[...] code is read from the stdout emitted
# between that specific test's own "started" and "skipped" lines (via awk,
# scoped per test), not from a blind substring search across the whole log,
# so that two tests never contaminate each other's skip code.

parse_xctest_log() {
  raw_log="$1"
  ledger_out="$2"
  : > "$ledger_out"

  awk '
    match($0, /-\[[A-Za-z0-9_.]+ ([A-Za-z0-9_]+)\]/) {
      name = substr($0, RSTART, RLENGTH)
      sub(/^-\[[A-Za-z0-9_.]+ /, "", name)
      sub(/\]$/, "", name)
      if ($0 ~ / started\.?$/) {
        current = name
        code = ""
        next
      }
      if ($0 ~ / passed /) { print "EXECUTED " name; current = ""; next }
      if ($0 ~ / failed /) { print "EXECUTED " name; current = ""; next }
      if ($0 ~ / skipped /) { print "SKIPPED " name " " code; current = ""; next }
    }
    current != "" && match($0, /SKIP\[[a-z-]+\]/) {
      code = substr($0, RSTART, RLENGTH)
    }
  ' "$raw_log" > "$ledger_out"
}
