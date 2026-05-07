#!/usr/bin/env bash
#
# watch-flaky-test-issues.sh
#
# Polls GitHub every $INTERVAL seconds (default 900 = 15 min) for the state of
# issues #95, #96, #97 on totalslacker/switchcraft. When all three are CLOSED,
# re-enables branch protection on `main` with required CI checks, posts a
# confirmation comment on issue #93, and exits.
#
# Why: those three issues remove `.disabled(if: CI)` skips and fix the LSM
# Indexer perf flake. Until they merge, "CI green on PR" is a fake gate
# (skipped tests can't fail CI). After they merge, CI is a meaningful gate
# again, so we restore the required-status-check protection.
#
# Run in background:
#   nohup ./scripts/watch-flaky-test-issues.sh > /dev/null 2>&1 &
#   disown
#
# Tail the log:
#   tail -f ~/.cache/switchcraft-watch-flaky.log
#
# Stop:
#   pkill -f watch-flaky-test-issues.sh

set -euo pipefail

REPO="totalslacker/switchcraft"
ISSUES=(95 96 97)
INTERVAL="${INTERVAL:-900}"
LOG_FILE="${LOG_FILE:-${HOME}/.cache/switchcraft-watch-flaky.log}"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

re_enable_protection() {
    log "All three issues closed — re-enabling branch protection on $REPO main"
    local payload
    payload=$(cat <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["swift test (macOS)", "swift test -c release (macOS)"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
)
    if printf '%s' "$payload" | gh api -X PUT "repos/$REPO/branches/main/protection" \
        -H "Accept: application/vnd.github+json" --input - >>"$LOG_FILE" 2>&1; then
        log "Branch protection restored on $REPO main."
        local body
        body="Branch protection on \`main\` automatically restored: #95, #96, and #97 all closed. \
Required status checks: \`swift test (macOS)\` and \`swift test -c release (macOS)\`; \
\`enforce_admins: true\`, \`strict: true\`, force-push and deletion blocked. \
(Posted by \`scripts/watch-flaky-test-issues.sh\`.)"
        gh issue comment 93 --repo "$REPO" --body "$body" >>"$LOG_FILE" 2>&1 \
            || log "WARN: comment on #93 failed but protection is in place — verify manually"
        return 0
    fi
    log "ERROR: branch protection PUT failed; see $LOG_FILE for the gh response"
    return 1
}

trap 'log "Watch script interrupted (signal received), exiting"; exit 130' INT TERM

log "Watch starting — interval=${INTERVAL}s repo=$REPO issues=${ISSUES[*]} pid=$$"

while true; do
    all_closed=true
    states=()
    for n in "${ISSUES[@]}"; do
        s=$(gh issue view "$n" --repo "$REPO" --json state --jq .state 2>/dev/null || echo "ERR")
        states+=("#$n=$s")
        if [[ "$s" != "CLOSED" ]]; then
            all_closed=false
        fi
    done
    log "tick: ${states[*]}"

    if $all_closed; then
        if re_enable_protection; then
            log "Done."
            exit 0
        else
            log "Will retry in ${INTERVAL}s"
        fi
    fi

    sleep "$INTERVAL"
done
