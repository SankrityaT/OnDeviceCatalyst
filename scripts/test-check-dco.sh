#!/bin/sh
set -eu

CHECKER=$(cd "$(dirname "$0")" && pwd)/check-dco.sh
TEST_REPO=$(mktemp -d)
trap 'rm -rf "$TEST_REPO"' EXIT INT TERM

cd "$TEST_REPO"
git init -q
git config user.name "DCO Test"
git config user.email "dco@example.com"

touch fixture
git add fixture
git commit -q -s -m "signed root"
BASE=$(git rev-parse HEAD)

echo signed >> fixture
git add fixture
git commit -q -s -m "signed change"
DCO_BASE_REF="$BASE" "$CHECKER" >/dev/null

SIGNED_HEAD=$(git rev-parse HEAD)
echo unsigned >> fixture
git add fixture
git commit -q -m "unsigned change"
if DCO_BASE_REF="$SIGNED_HEAD" "$CHECKER" >/dev/null 2>&1; then
  echo "DCO test failed: unsigned commit passed" >&2
  exit 1
fi

git commit --amend -q -m "mismatched sign-off

Signed-off-by: Someone Else <else@example.com>"
if DCO_BASE_REF="$SIGNED_HEAD" "$CHECKER" >/dev/null 2>&1; then
  echo "DCO test failed: mismatched sign-off passed" >&2
  exit 1
fi

echo "DCO checker tests passed"
