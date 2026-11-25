#!/bin/bash
# Daily check: do our patches still apply on the current upstream OptiScaler master?
# usage: bash REBASE/check_apply.sh [master]   (default: patches on BASE; "master": on the fetched upstream master; read-only fetch; throw-away worktree check_wt_<date_time>; our patches never touch OptiScaler.ini)
set -u
R=C:/src/ext/fsr4r/REBASE; F=$R/fork; G="git -c safe.directory=*"
BASE=$(cat $R/BASE)
cd $F
if [ "${1:-}" = master ]; then $G fetch -q upstream master || exit 2; UP=$($G rev-parse FETCH_HEAD); else UP=$BASE; fi
echo "BASE $BASE  upstream master $UP"
T=$R/check_wt_$(date +%F_%H%M%S); $G worktree add -q --detach "$T" $UP || exit 2
cd "$T"; ok=0; bad=0
for p in $R/patches/*.patch; do
  if $G -c user.name=check -c user.email=check@x am -q --3way --ignore-whitespace "$p" >/dev/null 2>&1; then ok=$((ok+1)); else bad=$((bad+1)); echo "CONFLICT $(basename $p)"; $G am --abort >/dev/null 2>&1; fi
done
echo "applied $ok, conflicts $bad (worktree $T; stale worktrees: move to _raw_to_delete)"
grep -l "OptiScaler.ini" $R/patches/*.patch | head -3 | sed 's/^/ini touched by: /'
