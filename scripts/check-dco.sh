#!/bin/sh
set -eu

DCO_BASE_REF=${DCO_BASE_REF:-HEAD^}
DCO_HEAD_REF=${DCO_HEAD_REF:-HEAD}

if ! git rev-parse --verify "$DCO_BASE_REF" >/dev/null 2>&1; then
  echo "DCO base ref does not exist: $DCO_BASE_REF" >&2
  exit 2
fi

if ! git rev-parse --verify "$DCO_HEAD_REF" >/dev/null 2>&1; then
  echo "DCO head ref does not exist: $DCO_HEAD_REF" >&2
  exit 2
fi

COMMITS=$(git rev-list --no-merges "$DCO_BASE_REF..$DCO_HEAD_REF")
if [ -z "$COMMITS" ]; then
  echo "DCO check: no non-merge commits in range"
  exit 0
fi

FAILED=0
for COMMIT in $COMMITS; do
  AUTHOR=$(git show -s --format='%an <%ae>' "$COMMIT")
  if ! git show -s --format='%B' "$COMMIT" | grep -Fqi "Signed-off-by: $AUTHOR"; then
    echo "DCO check failed: $COMMIT lacks matching sign-off for $AUTHOR" >&2
    FAILED=1
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "Add sign-off with: git commit --amend -s" >&2
  exit 1
fi

echo "DCO check passed"
