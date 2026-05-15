#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Manufacturing Hybrid GraphRAG — self-bootstrapping launcher.
#
# On a fresh machine this script will, in order:
#   1. Locate (or install) a usable Python ≥ 3.10.
#   2. Create a `.venv/` and install everything from requirements.txt
#      (skipped automatically if your current Python already has the deps).
#   3. Copy `.env.example` → `.env` if the latter doesn't exist.
#   4. Run `npm ci` (or `npm install`) inside `web/` if Node deps aren't ready.
#   5. Start three background services:
#        - FastAPI   (uvicorn)         :8000
#        - Streamlit                   :8501
#        - Next.js   (next dev)        :3000
#      Logs land in `.run/logs/<service>.log`, PIDs in `.run/<service>.pid`.
#
# Subsequent runs are fast — install steps are skipped via marker files.
#
# Environment overrides:
#   PYTHON_BIN          path to a specific Python interpreter
#   USE_VENV=1          force venv creation even when system Python has deps
#   SKIP_INSTALL=1      skip all install probes (assume env is ready)
#   INSTALL_ONLY=1      install everything but do not start services
#   SKIP_WEB=1          skip the Next.js UI
#   SKIP_STREAMLIT=1    skip the Streamlit UI
#   SKIP_HITL_SMOKE=1   skip the offline HITL smoke test (preflight)
#   HITL_DEFAULT=on|off when creating a fresh .env, default the HITL gate
#                       to enabled (on, default) or disabled (off)
#   ADVANCED_DEFAULT=on|off  on a fresh .env, switch the new advanced
#                            patterns on (rerank · cache · parallel ·
#                            guardrails · tools); defaults to 'on' for the
#                            safe subset (parallel retrieval + guardrails)
#                            only — set to 'off' to leave every advanced
#                            flag at its requirements.txt default.
#   API_PORT / STREAMLIT_PORT / WEB_PORT    override listening ports
#   NODE_BIN            path to node (default: auto-detect)
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"
LOG_DIR="$RUN_DIR/logs"
VENV_DIR="$ROOT/.venv"
mkdir -p "$LOG_DIR"

API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
WEB_PORT="${WEB_PORT:-3000}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-kg-rag-qdrant}"

# Required deps that the API + Streamlit need to be importable. If any of
# these is missing from the chosen Python, we'll (re)install requirements.txt.
# Keep this list small/fast — heavy modules (sentence_transformers, torch) are
# probed via `find_spec` only, not actually imported, so the check stays under
# 1 second instead of 10+ seconds.
REQUIRED_PKGS=(streamlit fastapi uvicorn qdrant_client sentence_transformers networkx pandas openai langgraph langchain_core langgraph.checkpoint.sqlite pydantic_settings yaml)

# ── pretty helpers ─────────────────────────────────────────────────────────
say()  { printf "  · %s\n" "$*"; }
ok()   { printf "  ✓ %s\n" "$*"; }
warn() { printf "  ⚠ %s\n" "$*"; }
err()  { printf "❌ %s\n" "$*" >&2; }
hr()   { printf "─ %s\n" "$*"; }
sha()  { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }

# ── 1. Resolve a usable Python ─────────────────────────────────────────────

py_version_ge_310() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null
}

py_has_required_deps() {
  local py="$1"
  # Use importlib.util.find_spec — checks if the module is installed without
  # actually importing it (so we don't pay 10s+ for sentence_transformers/torch).
  "$py" - <<PYEOF 2>/dev/null
import importlib.util, sys
mods = "${REQUIRED_PKGS[*]}".split()
missing = [m for m in mods if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PYEOF
}

find_base_python() {
  # 1) explicit override always wins
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1 && py_version_ge_310 "$PYTHON_BIN"; then
      printf "%s" "$PYTHON_BIN"; return 0
    fi
    err "PYTHON_BIN=$PYTHON_BIN is missing or below 3.10."
    return 1
  fi
  # 2) common locations
  for cand in /opt/anaconda3/bin/python /usr/local/bin/python3 /opt/homebrew/bin/python3 \
              python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && py_version_ge_310 "$cand"; then
      printf "%s" "$(command -v "$cand")"; return 0
    fi
  done
  return 1
}

bootstrap_python() {
  hr "Locating Python ≥ 3.10"

  # If we've already created a venv in a previous run, just use it.
  if [[ -x "$VENV_DIR/bin/python" ]] && [[ -z "${USE_VENV:-}" ]]; then
    PY="$VENV_DIR/bin/python"
    say "using existing venv: $VENV_DIR"
  else
    BASE_PY="$(find_base_python || true)"
    if [[ -z "$BASE_PY" ]]; then
      err "Couldn't find Python 3.10+ on PATH."
      cat <<HINT >&2

  Install Python 3.10+ and retry:
    macOS:   brew install python@3.12
    Ubuntu:  sudo apt install python3.12 python3.12-venv

  Or point us at an existing interpreter:
    PYTHON_BIN=/path/to/python3.12 ./run.sh
HINT
      exit 1
    fi
    BASE_PY_VER="$("$BASE_PY" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
    say "base python: $BASE_PY  (v$BASE_PY_VER)"

    # If the base interpreter ALREADY has every dep we need, just use it directly
    # (preserves the user's existing setup — e.g. anaconda with everything installed).
    if [[ -z "${USE_VENV:-}" ]] && py_has_required_deps "$BASE_PY"; then
      PY="$BASE_PY"
      ok "base python already has the required packages — skipping venv"
    else
      [[ -d "$VENV_DIR" ]] || {
        say "creating venv at $VENV_DIR"
        "$BASE_PY" -m venv "$VENV_DIR"
      }
      PY="$VENV_DIR/bin/python"
      ok "venv ready: $VENV_DIR"
    fi
  fi

  PY_VER="$("$PY" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
  ok "python: $PY  (v$PY_VER)"
}

# ── 2. Install Python deps (idempotent via SHA marker) ─────────────────────

install_python_deps() {
  hr "Python dependencies"
  local req="$ROOT/requirements.txt"
  if [[ ! -f "$req" ]]; then
    warn "no requirements.txt — skipping pip install"
    return 0
  fi

  # Using base (non-venv) Python — only touch it if deps are actually missing.
  if [[ "$PY" != "$VENV_DIR/bin/python" ]]; then
    if py_has_required_deps "$PY"; then
      ok "base python already has the required packages — nothing to install"
      return 0
    fi
    err "PYTHON_BIN=$PY is missing required packages: ${REQUIRED_PKGS[*]}"
    cat <<HINT >&2
   Either install them manually:
     $PY -m pip install -r requirements.txt
   …or unset PYTHON_BIN so run.sh can create a managed venv:
     unset PYTHON_BIN; ./run.sh
HINT
    exit 1
  fi

  # Inside our managed venv — use a SHA marker so subsequent runs skip the install.
  local marker="$VENV_DIR/.req-hash"
  local cur_hash="$(sha "$req")"
  local prev_hash=""
  [[ -f "$marker" ]] && prev_hash="$(cat "$marker" 2>/dev/null || true)"

  if [[ -n "$cur_hash" ]] && [[ "$cur_hash" == "$prev_hash" ]] && py_has_required_deps "$PY"; then
    ok "requirements.txt unchanged — skipping pip install"
    return 0
  fi

  say "installing requirements.txt (first run may take a few minutes)…"
  "$PY" -m pip install --upgrade pip >>"$LOG_DIR/pip.log" 2>&1 || true
  if ! "$PY" -m pip install -r "$req" >>"$LOG_DIR/pip.log" 2>&1; then
    err "pip install failed. Tail of the log:"
    tail -n 40 "$LOG_DIR/pip.log" || true
    exit 1
  fi
  printf "%s" "$cur_hash" >"$marker"
  ok "Python deps installed"
}

# ── 3. Bootstrap .env ──────────────────────────────────────────────────────

# Flip a KEY=VALUE line in-place (creates the line if missing). Uses a sed
# fallback portable to macOS (BSD) and Linux (GNU).
env_set() {
  local file="$1" key="$2" val="$3"
  if grep -qE "^${key}=" "$file" 2>/dev/null; then
    # In-place edit. The 's|…|…|g' pattern uses '|' to avoid path escaping.
    if sed --version >/dev/null 2>&1; then
      sed -i -E "s|^${key}=.*|${key}=${val}|" "$file"
    else
      sed -i '' -E "s|^${key}=.*|${key}=${val}|" "$file"
    fi
  else
    printf "%s=%s\n" "$key" "$val" >>"$file"
  fi
}

bootstrap_env_file() {
  hr "Environment file (.env)"
  local default_hitl="${HITL_DEFAULT:-on}"

  if [[ ! -f "$ROOT/.env" ]]; then
    if [[ -f "$ROOT/.env.example" ]]; then
      cp "$ROOT/.env.example" "$ROOT/.env"
      ok "created .env from .env.example"
    else
      : >"$ROOT/.env"
      ok "created empty .env"
    fi

    # Fresh install: turn on the LangGraph + HITL stack so '📋 Approvals'
    # is visible out of the box. Override with HITL_DEFAULT=off ./run.sh
    # if you want a quiet first boot.
    if [[ "$default_hitl" == "on" ]]; then
      env_set "$ROOT/.env" USE_LANGGRAPH true
      env_set "$ROOT/.env" USE_HITL true
      env_set "$ROOT/.env" HITL_CHECKPOINT_BACKEND sqlite
      ok "HITL approval gate enabled by default (USE_LANGGRAPH=true · USE_HITL=true)"
    else
      ok "HITL flags left at false (HITL_DEFAULT=off)"
    fi
    warn "set OPENAI_API_KEY in .env to enable LLM modes (retrieval-only still works)"
    return 0
  fi

  ok ".env present"

  # Existing .env: this is a first-time upgrade if the HITL keys are missing.
  # Append the block AND turn the gate on by default (HITL_DEFAULT=on), so a
  # user pulling the new run.sh into an existing checkout gets the same
  # "just works" UX as a fresh clone. They can opt out with HITL_DEFAULT=off
  # or by editing .env after the fact. If the keys are already there, we
  # leave the user's values untouched.
  local missing=0
  local key
  for key in USE_HITL HITL_RISK_THRESHOLD HITL_AUTO_APPROVE_BELOW_USD \
              HITL_HIGH_RISK_KEYWORDS HITL_DB_PATH HITL_CHECKPOINT_BACKEND; do
    if ! grep -qE "^${key}=" "$ROOT/.env" 2>/dev/null; then
      missing=1
      break
    fi
  done

  if [[ "$missing" == "1" ]]; then
    local hitl_default_val="false"
    [[ "$default_hitl" == "on" ]] && hitl_default_val="true"
    say "appending HITL defaults (USE_HITL=${hitl_default_val})"
    {
      printf "\n"
      printf "# ── Human-in-the-Loop (HITL) approval gate ────────────────────────────────\n"
      printf "# Auto-added by run.sh. Flip USE_HITL to false (and remove USE_LANGGRAPH=true)\n"
      printf "# to disable the approval gate. See system_design/HITL_DESIGN.md for the PRD.\n"
      printf "USE_LANGGRAPH=%s\n" "$hitl_default_val"
      printf "USE_HITL=%s\n" "$hitl_default_val"
      printf "HITL_RISK_THRESHOLD=0.6\n"
      printf "HITL_AUTO_APPROVE_BELOW_USD=2000\n"
      printf "HITL_HIGH_RISK_KEYWORDS=lockout,tagout,hot work,fire,explosion,h2s,arc flash,confined space,fatal,injury,death,toxic,asphyxiation,radiation,permit-to-work,shutdown,emergency\n"
      printf "HITL_DB_PATH=data/processed/audit.sqlite\n"
      printf "HITL_CHECKPOINT_BACKEND=sqlite\n"
    } >>"$ROOT/.env"
    if [[ "$hitl_default_val" == "true" ]]; then
      ok "HITL approval gate ENABLED on first upgrade (USE_LANGGRAPH=true · USE_HITL=true)"
    else
      ok "HITL keys appended; USE_HITL=false (flip to true when ready)"
    fi
  fi

  # If the user already has USE_HITL set but no USE_LANGGRAPH, nudge them.
  if grep -qE '^USE_HITL=(1|true|yes|on)$' "$ROOT/.env" 2>/dev/null \
      && ! grep -qE '^USE_LANGGRAPH=(1|true|yes|on)$' "$ROOT/.env" 2>/dev/null; then
    warn "USE_HITL=true but USE_LANGGRAPH=false — the approval gate requires LangGraph."
    say "fixing: setting USE_LANGGRAPH=true so HITL actually runs"
    env_set "$ROOT/.env" USE_LANGGRAPH true
  fi

  bootstrap_advanced_flags
}

# ── 3a. Advanced patterns (rerank · cache · parallel · guardrails · tools) ─
#
# These are the six "production hardening" patterns layered on top of the
# core Hybrid GraphRAG engine. We split them into TWO buckets:
#
#   * SAFE-ON  — flags that have no external deps and only improve latency
#                or safety. Enabled by default on a fresh .env so the demo
#                ships with the right posture.
#                  USE_PARALLEL_RETRIEVAL=true
#                  USE_GUARDRAILS=true
#
#   * OPT-IN   — flags that pull additional models (cross-encoder), allocate
#                memory (semantic cache) or expose write surface (ERP/MES
#                tool calls). Left disabled by default and listed here so a
#                user can flip them in seconds.
#                  USE_RERANKER=false
#                  USE_SEMANTIC_CACHE=false
#                  USE_TOOLS=false
#
# Override the bucket with ADVANCED_DEFAULT=off ./run.sh if you want a
# completely clean .env.

bootstrap_advanced_flags() {
  local advanced_default="${ADVANCED_DEFAULT:-on}"
  local missing=0
  local key
  for key in USE_PARALLEL_RETRIEVAL USE_GUARDRAILS USE_RERANKER \
             USE_SEMANTIC_CACHE USE_TOOLS; do
    if ! grep -qE "^${key}=" "$ROOT/.env" 2>/dev/null; then
      missing=1
      break
    fi
  done

  if [[ "$missing" == "0" ]]; then
    return 0
  fi

  local safe_on="false"
  [[ "$advanced_default" == "on" ]] && safe_on="true"

  say "appending advanced-pattern defaults (safe-on=${safe_on})"
  {
    printf "\n"
    printf "# ── Advanced patterns: rerank · cache · parallel · guardrails · tools ─\n"
    printf "# Auto-added by run.sh. See README \"Advanced patterns\" + .env.example.\n"
    printf "# Safe-on (no extra deps, latency / safety wins):\n"
    printf "USE_PARALLEL_RETRIEVAL=%s\n" "$safe_on"
    printf "PARALLEL_RETRIEVAL_TIMEOUT_S=15.0\n"
    printf "USE_GUARDRAILS=%s\n" "$safe_on"
    printf "GUARDRAILS_REQUIRE_CITATIONS=true\n"
    printf "GUARDRAILS_MIN_CITATIONS=1\n"
    printf "GUARDRAILS_BLOCK_UNSAFE=true\n"
    printf "# Opt-in (extra cost / state / surface area — flip on when you need them):\n"
    printf "USE_RERANKER=false\n"
    printf "RERANKER_MODEL=BAAI/bge-reranker-base\n"
    printf "RERANK_CANDIDATE_POOL=20\n"
    printf "RERANK_BLEND_WEIGHT=0.7\n"
    printf "USE_SEMANTIC_CACHE=false\n"
    printf "SEMANTIC_CACHE_THRESHOLD=0.97\n"
    printf "SEMANTIC_CACHE_MAX_SIZE=256\n"
    printf "SEMANTIC_CACHE_TTL_SECONDS=3600\n"
    printf "USE_TOOLS=false\n"
    printf "TOOL_PLANNER_MODEL=qwen2.5:3b\n"
    printf "TOOL_PLANNER_USE_LLM=true\n"
  } >>"$ROOT/.env"
  if [[ "$safe_on" == "true" ]]; then
    ok "advanced patterns: parallel retrieval + guardrails ENABLED, rerank/cache/tools opt-in"
  else
    ok "advanced patterns: all flags appended at false (ADVANCED_DEFAULT=off)"
  fi
}

# ── 3b. Pre-create directories the HITL stack writes to ────────────────────

bootstrap_data_dirs() {
  hr "Data directories"
  mkdir -p "$ROOT/data/processed"
  ok "data/processed/ ready (HITL_DB_PATH lives here)"
}

# ── 3c. Offline preflight: run the HITL smoke test ─────────────────────────

run_hitl_smoke() {
  hr "HITL preflight (offline smoke test)"
  if [[ "${SKIP_HITL_SMOKE:-0}" == "1" ]]; then
    say "skipped (SKIP_HITL_SMOKE=1)"
    return 0
  fi
  if [[ ! -f "$ROOT/scripts/smoke_test_hitl.py" ]]; then
    say "scripts/smoke_test_hitl.py not present — skipping preflight"
    return 0
  fi
  local logfile="$LOG_DIR/hitl-smoke.log"
  if (cd "$ROOT" && PYTHONPATH="$ROOT" "$PY" scripts/smoke_test_hitl.py \
        >"$logfile" 2>&1); then
    ok "HITL smoke test passed (5/5) — log: ${logfile#"$ROOT/"}"
  else
    warn "HITL smoke test FAILED — log: ${logfile#"$ROOT/"}"
    tail -n 15 "$logfile" || true
    warn "continuing anyway (services will start; fix this before relying on HITL)"
  fi
}

# ── 4. Resolve Node & install web deps ─────────────────────────────────────

NODE="${NODE_BIN:-$(command -v node 2>/dev/null || true)}"
NPM="$(command -v npm 2>/dev/null || true)"

node_ok() {
  [[ -n "$NODE" ]] && [[ -n "$NPM" ]] && {
    local major
    major="$("$NODE" -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
    [[ "$major" -ge 18 ]]
  }
}

install_web_deps() {
  hr "Web (Next.js) dependencies"
  if ! node_ok; then
    warn "Node.js 18+ / npm not found — the Next.js UI will be skipped."
    cat <<HINT
       Install Node 20 (LTS) and rerun:
         macOS:   brew install node
         Ubuntu:  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs
       Or skip the web UI permanently with:  SKIP_WEB=1 ./run.sh
HINT
    SKIP_WEB=1
    return 0
  fi
  local node_ver
  node_ver="$("$NODE" --version 2>/dev/null || echo unknown)"
  say "node: $node_ver  · npm: $("$NPM" --version)"

  local pkg="$ROOT/web/package.json"
  local lock="$ROOT/web/package-lock.json"
  local marker="$ROOT/web/.deps-hash"
  local key_file="$lock"
  [[ -f "$lock" ]] || key_file="$pkg"
  local cur_hash="$(sha "$key_file")"
  local prev_hash=""
  [[ -f "$marker" ]] && prev_hash="$(cat "$marker" 2>/dev/null || true)"

  if [[ -d "$ROOT/web/node_modules" ]] && [[ "$cur_hash" == "$prev_hash" ]]; then
    ok "web/ deps already installed (lockfile unchanged)"
    return 0
  fi

  say "installing web/ dependencies (this can take 30–90s on the first run)…"
  local cmd
  if [[ -f "$lock" ]]; then
    cmd=(npm ci --silent --no-audit --no-fund)
  else
    cmd=(npm install --silent --no-audit --no-fund)
  fi
  if (cd "$ROOT/web" && "${cmd[@]}" >>"$LOG_DIR/web-install.log" 2>&1); then
    printf "%s" "$cur_hash" >"$marker"
    ok "web/ deps installed"
  else
    err "npm install failed. Tail of the log:"
    tail -n 40 "$LOG_DIR/web-install.log" || true
    exit 1
  fi
}

# ── 5. Service helpers ─────────────────────────────────────────────────────

free_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    say "port $port busy (pids: $pids) — releasing"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

wait_for_url() {
  local url="$1" label="$2" tries="${3:-60}"
  for ((i = 1; i <= tries; i++)); do
    local code
    code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null || echo 000)"
    if [[ "$code" =~ ^2[0-9]{2}$ ]]; then
      ok "$label up — $url"
      return 0
    fi
    sleep 1
  done
  warn "$label did not respond at $url within ${tries}s (continuing)"
  return 1
}

bg_run() {
  local name="$1"; shift
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"
  say "starting $name → log: ${logfile#"$ROOT/"}"
  (
    cd "$ROOT"
    nohup "$@" >>"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )
  sleep 0.4
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    err "$name failed to start. Tail of the log:"
    tail -n 30 "$logfile" || true
    exit 1
  fi
}

# ── 6. Banner + run ────────────────────────────────────────────────────────

cat <<BANNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Manufacturing Hybrid GraphRAG — run.sh
  root  : $ROOT
  ports : api=$API_PORT  streamlit=$STREAMLIT_PORT  web=$WEB_PORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANNER

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  bootstrap_python
  install_python_deps
  bootstrap_env_file
  bootstrap_data_dirs
  run_hitl_smoke
  install_web_deps
else
  hr "Install steps skipped (SKIP_INSTALL=1)"
  PY="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
  [[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"
  ok "python: $PY"
fi

if [[ "${INSTALL_ONLY:-0}" == "1" ]]; then
  echo ""
  ok "Install steps complete. INSTALL_ONLY=1 → not starting services."
  exit 0
fi

# Read the effective HITL state from .env (best-effort — used only for the
# end-of-run summary). Pure-shell whitespace trim (avoids `xargs` which trips
# on long values on macOS).
effective_flag() {
  local key="$1" default="$2"
  if [[ -f "$ROOT/.env" ]]; then
    local val
    val="$(grep -E "^${key}=" "$ROOT/.env" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'" || true)"
    # Trim leading/trailing whitespace without invoking xargs.
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    [[ -n "$val" ]] && { printf "%s" "$val"; return; }
  fi
  printf "%s" "$default"
}
HITL_FLAG="$(effective_flag USE_HITL false)"
LG_FLAG="$(effective_flag USE_LANGGRAPH false)"

# Effective state of the six advanced-pattern flags (for the end-of-run
# summary). Each `effective_flag <KEY> <default>` reads .env and falls
# back to the upstream default in config.py.
PAR_FLAG="$(effective_flag USE_PARALLEL_RETRIEVAL true)"
GR_FLAG="$(effective_flag USE_GUARDRAILS true)"
RR_FLAG="$(effective_flag USE_RERANKER false)"
SC_FLAG="$(effective_flag USE_SEMANTIC_CACHE false)"
TOOL_FLAG="$(effective_flag USE_TOOLS false)"

# Helper that turns a flag value into a one-word on/off label.
onoff() {
  case "$1" in
    1|true|yes|on) printf "on" ;;
    *)            printf "off" ;;
  esac
}

echo ""

# ─ 6a. Qdrant (vector store) ──
# Embedded Qdrant uses an exclusive file lock so it cannot be shared across
# the FastAPI + Streamlit + Next.js processes this script launches. We
# auto-boot a Qdrant container (or skip if QDRANT_URL is already configured
# to point at an external instance) and inject QDRANT_URL into every child.
QDRANT_URL_ENV="$(effective_flag QDRANT_URL "")"

start_qdrant_container() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "docker not found — leaving Qdrant in embedded mode."
    warn "Run only ONE python service at a time, or install docker."
    return 0
  fi

  # Remove a stale container with the same name (idempotent restarts).
  docker rm -f "$QDRANT_CONTAINER" >/dev/null 2>&1 || true

  local storage="$ROOT/doc_pipeline/vector_store/qdrant"
  mkdir -p "$storage"
  free_port "$QDRANT_PORT"

  say "starting qdrant container ($QDRANT_IMAGE)"
  docker run -d --rm \
    --name "$QDRANT_CONTAINER" \
    -p "$QDRANT_PORT":6333 \
    -v "$storage":/qdrant/storage \
    "$QDRANT_IMAGE" \
    >"$RUN_DIR/qdrant.cid" 2>"$LOG_DIR/qdrant.log" || {
      err "qdrant container failed to start. Tail of qdrant.log:"
      tail -n 30 "$LOG_DIR/qdrant.log" || true
      exit 1
    }
  wait_for_url "http://localhost:$QDRANT_PORT/" "qdrant" 60
  export QDRANT_URL="http://localhost:$QDRANT_PORT"
  ok "qdrant url: $QDRANT_URL"
}

hr "Qdrant vector store"
if [[ -n "$QDRANT_URL_ENV" ]]; then
  say "QDRANT_URL=$QDRANT_URL_ENV in .env — assuming external Qdrant. Skipping container."
  export QDRANT_URL="$QDRANT_URL_ENV"
else
  start_qdrant_container
fi

# ─ 6b. FastAPI ──
hr "FastAPI backend (api/server.py)"
free_port "$API_PORT"
PYTHONPATH="$ROOT" bg_run "api" "$PY" -m uvicorn api.server:app \
  --host 0.0.0.0 --port "$API_PORT"
wait_for_url "http://localhost:$API_PORT/api/health" "api" 180

# ─ 6b. Streamlit ──
if [[ "${SKIP_STREAMLIT:-0}" != "1" ]]; then
  hr "Streamlit UI (app.py)"
  free_port "$STREAMLIT_PORT"
  bg_run "streamlit" "$PY" -m streamlit run app.py \
    --server.port "$STREAMLIT_PORT" --server.headless true \
    --browser.gatherUsageStats false
  wait_for_url "http://localhost:$STREAMLIT_PORT/_stcore/health" "streamlit" 90
else
  hr "Streamlit UI skipped (SKIP_STREAMLIT=1)"
fi

# ─ 6c. Next.js ──
if [[ "${SKIP_WEB:-0}" != "1" ]] && node_ok; then
  hr "Next.js UI (web/)"
  free_port "$WEB_PORT"
  (
    cd "$ROOT/web"
    NEXT_PUBLIC_API_ORIGIN="http://localhost:$API_PORT" \
    nohup "$NPM" run dev -- --port "$WEB_PORT" \
      >>"$LOG_DIR/web.log" 2>&1 &
    echo $! >"$RUN_DIR/web.pid"
  )
  wait_for_url "http://localhost:$WEB_PORT/" "web" 180 || true
else
  hr "Next.js UI skipped"
fi

# HITL line for the final summary (depends on the effective flags).
HITL_SUMMARY=""
if [[ "$HITL_FLAG" =~ ^(1|true|yes|on)$ ]] && [[ "$LG_FLAG" =~ ^(1|true|yes|on)$ ]]; then
  HITL_SUMMARY="
   HITL gate    ENABLED  →  http://localhost:$STREAMLIT_PORT  (📋 Approvals tab)
                            http://localhost:$API_PORT/api/approvals/pending
                            http://localhost:$API_PORT/api/audit
                            DB: data/processed/audit.sqlite"
else
  HITL_SUMMARY="
   HITL gate    disabled  (set USE_HITL=true + USE_LANGGRAPH=true in .env to enable)"
fi

cat <<DONE

✅ Stack is up.

   API        http://localhost:$API_PORT/docs
   Streamlit  http://localhost:$STREAMLIT_PORT
   Web        http://localhost:$WEB_PORT
${HITL_SUMMARY}

   Advanced   parallel-retrieval=$(onoff "$PAR_FLAG")  guardrails=$(onoff "$GR_FLAG")  reranker=$(onoff "$RR_FLAG")
              semantic-cache=$(onoff "$SC_FLAG")  tools=$(onoff "$TOOL_FLAG")
              (toggle in .env — see README "Advanced patterns")

   Logs:    .run/logs/{api,streamlit,web,hitl-smoke}.log
   PIDs:    .run/{api,streamlit,web}.pid

   Stop everything:  ./stop.sh
   Status check:     ./status.sh
   Tail a service:   tail -f .run/logs/api.log
   Re-run smoke:     .venv/bin/python scripts/smoke_test_hitl.py
   Offline eval:     .venv/bin/python -m comparison.eval.run \\
                       --output comparison/eval/report.md
DONE
