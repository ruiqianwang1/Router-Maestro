#!/usr/bin/env bash
# Rebase the local-only patch onto the latest upstream.
#
# This fork's master is deliberately NOT a pure upstream mirror: it is
# "upstream/master + the local-only patch". The patch is never merged upstream,
# so it has to be replayed onto every new upstream tip. Rebase (not merge) keeps
# it as "upstream HEAD + N commits", so the conflict surface stays constant
# instead of accumulating with every sync.
#
# Because master carries the patch, PR branches must be cut from upstream/master,
# never from master:
#
#     git checkout -b feat/whatever upstream/master
#
# rerere records how each conflict was resolved and replays that resolution on
# the next sync, so a conflict only has to be solved by hand once.
#
# Usage: scripts/sync-upstream.sh [branch]   (default: current branch)
set -euo pipefail

BRANCH="${1:-$(git branch --show-current)}"

if [ -z "$BRANCH" ]; then
    echo "error: detached HEAD; pass a branch name explicitly." >&2
    exit 1
fi

# rerere is what makes a repeatedly-rebased patch bearable; enable it in case
# this is a fresh clone.
git config rerere.enabled true
git config rerere.autoupdate true

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: working tree has uncommitted changes; commit or stash first." >&2
    exit 1
fi

echo "==> Fetching upstream"
git fetch upstream

before="$(git rev-parse HEAD)"

echo "==> Rebasing $BRANCH onto upstream/master"
git checkout "$BRANCH"
if ! git rebase upstream/master; then
    cat >&2 <<EOF

Rebase stopped on a conflict.

  1. Resolve the conflicted files, then: git add <files>
  2. Continue with:                       git rebase --continue
  3. Or abort entirely with:              git rebase --abort

rerere has recorded the resolution, so the next sync should replay it
automatically.

If anything goes wrong, the pre-rebase tip is recoverable:
  git reset --hard $before
EOF
    exit 1
fi

echo "==> Running tests"
# --basetemp works around a Windows PermissionError in pytest's default
# tmp_path handling.
uv run pytest tests/ -q --basetemp=.pytest-tmp

echo
echo "==> Done. $BRANCH is upstream/master + $(git rev-list --count upstream/master..HEAD) local commit(s)."
echo "    Pre-rebase tip was $before (recoverable via git reset --hard)."
