#!/usr/bin/env bash
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then gtimeout "$secs" "$@"
  else
    "$@" & local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) & local watcher=$!
    wait "$pid" 2>/dev/null; local rc=$?
    kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; return $rc
  fi
}

portable_mktemp() { local prefix="${1:-validate}"; mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp; }
CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"; REPO_DIR="${2:-.}"
if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"; exit 1
fi
if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"; exit 1
fi
PING_URL="${PING_URL%/}"; export PING_URL; PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
info() { printf "  ${YELLOW}INFO:${NC}  %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC}\n" "$1"
  exit 1
}

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ===========================================================================
# STEP 1 — Ping HF Space
# ===========================================================================
log "${BOLD}Step 1/3: Pinging HF Space${NC}"
info "Sending POST $PING_URL/reset with empty body..."

CURL_OUTPUT=$(portable_mktemp "validate-curl"); CLEANUP_FILES+=("$CURL_OUTPUT")

START_T=$(date +%s)
HTTP_CODE=$(curl -v -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>&1 | tee /tmp/curl_verbose.log | grep -oP '(?<=< HTTP/)[0-9.]+\s[0-9]+' | tail -1 | awk '{print $2}' || echo "")

# Fallback: get code from curl exit
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 || printf "000")

END_T=$(date +%s)
ELAPSED=$((END_T - START_T))

info "HTTP response code : $HTTP_CODE"
info "Response time      : ${ELAPSED}s"
info "Response body preview:"
if [ -s "$CURL_OUTPUT" ]; then
  head -c 300 "$CURL_OUTPUT" | tr -d '\000-\010\013\014\016-\037' && printf "\n"
else
  info "(empty response body)"
fi

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  stop_at "Step 1"
fi

# ===========================================================================
# STEP 2 — Docker build
# ===========================================================================
printf "\n"
log "${BOLD}Step 2/3: Docker build${NC}"

if ! command -v docker &>/dev/null; then
  fail "docker not found"; stop_at "Step 2"
fi

info "Docker version: $(docker --version)"
info "Docker context: $REPO_DIR"

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found"; stop_at "Step 2"
fi

info "Dockerfile location: $DOCKER_CONTEXT/Dockerfile"
info "Dockerfile contents:"
printf "  ---\n"
head -20 "$DOCKER_CONTEXT/Dockerfile" | while IFS= read -r line; do printf "  %s\n" "$line"; done
printf "  ---\n"

info "Starting docker build (timeout=${DOCKER_BUILD_TIMEOUT}s) ..."
info "You will see live build output below:"
printf "\n"

BUILD_START=$(date +%s)
BUILD_OK=false
if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build --progress=plain "$DOCKER_CONTEXT"; then
  BUILD_OK=true
fi
BUILD_END=$(date +%s)
BUILD_ELAPSED=$((BUILD_END - BUILD_START))

printf "\n"
info "Build duration: ${BUILD_ELAPSED}s"

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed after ${BUILD_ELAPSED}s"
  stop_at "Step 2"
fi

# ===========================================================================
# STEP 3 — openenv validate
# ===========================================================================
printf "\n"
log "${BOLD}Step 3/3: openenv validate${NC}"

if ! command -v openenv &>/dev/null; then
  fail "openenv not found"
  hint "pip install openenv-core"
  stop_at "Step 3"
fi

info "openenv version: $(openenv --version 2>/dev/null || echo 'unknown')"
info "Working directory: $REPO_DIR"
info "Files present:"
ls "$REPO_DIR" | while IFS= read -r f; do printf "  %s\n" "$f"; done

info "Running: openenv validate ..."
printf "\n"

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

printf "%s\n" "$VALIDATE_OUTPUT"
printf "\n"

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  stop_at "Step 3"
fi

# ===========================================================================
# SUMMARY
# ===========================================================================
printf "\n${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n\n"
exit 0
