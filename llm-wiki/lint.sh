#!/usr/bin/env bash
# llmwiki lint — mechanical health check over the vault. See llmwiki/SCHEMA.md.
#
# Usage: llmwiki/lint.sh          (from anywhere)
# Exit 0 = clean, 1 = findings.
#
# Checks:
#   1. frontmatter   — every page (except index.md/log.md/SCHEMA.md) carries the required keys
#   2. links         — every [[wikilink]] resolves to a page in the vault
#   3. orphans       — pages with no inbound link other than from index.md/log.md or themselves
#   4. stale sources — raw/ snapshot hashes no longer match the hash recorded on the source page
#
# All page paths are vault-relative; raw paths on source pages are repo-relative
# ("llmwiki/raw/x") and are resolved by stripping the vault's own name.

set -uo pipefail
VAULT=$(cd "$(dirname "$0")" && pwd) || exit 2
cd "$VAULT" || exit 2
VAULT_NAME=$(basename "$VAULT")

FAIL=0
findings() { printf '%s\n' "$*"; FAIL=1; }

PAGES=$(find . -name '*.md' -not -path './raw/*' \
  | sed 's|^\./||' \
  | grep -v -E '^(index|log|SCHEMA)\.md$' \
  | sort)

if [ -z "$PAGES" ]; then
  printf 'llmwiki lint: no pages found under %s\n' "$VAULT"; exit 1
fi

REQUIRED="title type status updated sources verified tags"

# --- 1. frontmatter -----------------------------------------------------------
while IFS= read -r page; do
  [ -n "$page" ] || continue
  if [ "$(head -1 "$page")" != "---" ]; then
    findings "FRONTMATTER  $page: does not start with '---'"
    continue
  fi
  closing=$(awk 'NR>1 && $0=="---"{print NR; exit}' "$page")
  if [ -z "$closing" ]; then
    findings "FRONTMATTER  $page: frontmatter not terminated"
    continue
  fi
  fm=$(sed -n "2,$((closing - 1))p" "$page")
  for key in $REQUIRED; do
    printf '%s\n' "$fm" | grep -q "^${key}:" || findings "FRONTMATTER  $page: missing '$key'"
  done
done <<< "$PAGES"

# --- link inventory -----------------------------------------------------------
# BASENAMES: every page name a link may resolve to (incl. index/log/SCHEMA).
BASENAMES=$(find . -name '*.md' -not -path './raw/*' | sed 's|^\./||; s|\.md$||' | sed 's|.*/||' | sort -u)

LINKMAP=$(mktemp); trap 'rm -f "$LINKMAP"' EXIT
while IFS= read -r page; do
  [ -n "$page" ] || continue
  grep -o '\[\[[^]]*\]\]' "$page" 2>/dev/null \
    | sed -e 's/^\[\[//' -e 's/\]\]$//' -e 's/|.*$//' -e 's/#.*$//' \
    | while IFS= read -r target; do
        [ -n "$target" ] && printf '%s\t%s\n' "$page" "$target"
      done
done <<< "$PAGES" > "$LINKMAP"

# --- 2. broken links ----------------------------------------------------------
cut -f2 "$LINKMAP" | sort -u | while IFS= read -r target; do
  printf '%s\n' "$BASENAMES" | grep -qx "$target" || printf '%s\n' "$target"
done > /tmp/.llmwiki_broken.$$ 2>/dev/null
if [ -s /tmp/.llmwiki_broken.$$ ]; then
  while IFS= read -r target; do
    holders=$(awk -F'\t' -v t="$target" '$2==t{printf "%s ", $1}' "$LINKMAP")
    findings "BROKEN LINK  [[$target]] <- $holders"
  done < /tmp/.llmwiki_broken.$$
fi
rm -f /tmp/.llmwiki_broken.$$

# --- 3. orphans ---------------------------------------------------------------
while IFS= read -r page; do
  [ -n "$page" ] || continue
  base=$(basename "$page" .md)
  inbound=$(awk -F'\t' -v t="$base" \
    '$2==t && $1!="index.md" && $1!="log.md" {n++} END{printf "%d", n+0}' "$LINKMAP")
  [ "$inbound" -eq 0 ] && findings "ORPHAN       $page (no inbound link outside index.md/log.md)"
done <<< "$PAGES"

# --- 4. stale raw snapshots ---------------------------------------------------
for src in sources/*.md; do
  [ -f "$src" ] || continue
  raw=$(grep -m1 '^- raw path:' "$src" | sed 's/.*: *//' | sed 's/`//g')
  want=$(grep -m1 '^- sha256:' "$src" | sed 's/.*: *//' | sed 's/`//g')
  if [ -z "$raw" ] || [ -z "$want" ]; then
    findings "PROVENANCE   $src: raw path and/or sha256 not recorded"; continue
  fi
  case "$raw" in
    "$VAULT_NAME"/*) raw="${raw#"$VAULT_NAME"/}" ;;
  esac
  if [ ! -f "$raw" ]; then
    findings "PROVENANCE   $src: raw file missing: $raw"; continue
  fi
  have=$(sha256sum "$raw" | cut -d' ' -f1)
  [ "$want" = "$have" ] || findings "STALE SOURCE $src: $raw changed (recorded ${want:0:12}…, now ${have:0:12}…)"
done

# --- summary ------------------------------------------------------------------
if [ "$FAIL" -eq 0 ]; then
  printf 'llmwiki lint: clean (%s pages)\n' "$(printf '%s\n' "$PAGES" | grep -c .)"
else
  printf 'llmwiki lint: findings above\n'
fi
exit "$FAIL"
