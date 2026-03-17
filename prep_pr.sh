#!/usr/bin/env bash
# prep_pr.sh — strips toolkit files from the current branch before opening a PR
# Usage: ./prep_pr.sh
# Run after cleaning up commit history. Modifies the current branch in place.

set -euo pipefail

TOOLKIT_FILES=(
  "AGENTS.md"
  "CLAUDE.md"
  "jules_handoff.sh"
  ".coderabbit.yaml"
  "scout_report.md"
  ".github/workflows/sync-upstream.yml"
  ".github/workflows/scout.yml"
)

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'
die()  { echo -e "${RED}error: $1${NC}" >&2; exit 1; }
warn() { echo -e "${YELLOW}warn:  $1${NC}"; }
ok()   { echo -e "${GREEN}ok:    $1${NC}"; }

command -v git &>/dev/null || die "git not found"
git rev-parse --git-dir &>/dev/null || die "not inside a git repo"

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null) \
  || die "HEAD is detached — check out your working branch first"

[[ "$BRANCH" == "master" || "$BRANCH" == "main" ]] \
  && die "you're on $BRANCH — check out your working branch first"

git diff --quiet && git diff --cached --quiet \
  || die "you have uncommitted changes — commit or stash them first"

echo ""
echo "  Branch : $BRANCH"
echo "  Will strip toolkit files and add a cleanup commit."
echo ""
read -rp "Proceed? [y/N] " confirm
[[ "${confirm,,}" == "y" ]] || { echo "Aborted."; exit 0; }

STRIPPED=()
for f in "${TOOLKIT_FILES[@]}"; do
  if git ls-files --error-unmatch "$f" &>/dev/null 2>&1; then
    git rm -f "$f"
    STRIPPED+=("$f")
  fi
done

if [[ ${#STRIPPED[@]} -eq 0 ]]; then
  warn "no toolkit files found — nothing to strip"
  exit 0
fi

git commit -m "chore: strip toolkit files before PR

Local workflow helpers not intended for upstream:
$(printf '  - %s\n' "${STRIPPED[@]}")"

ok "stripped: ${STRIPPED[*]}"
echo ""
echo "  Next:"
echo "    git push origin $BRANCH"
echo "    open PR from $BRANCH → upstream master"