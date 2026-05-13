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

# Required deps that the API + Streamlit need to be importable. If any of
# these is missing from the chosen Python, we'll (re)install requirements.txt.
# Keep this list small/fast — heavy modules (sentence_transformers, torch) are
# probed via `find_spec` only, not actually imported, so the check stays under
# 1 second instead of 10+ seconds.
REQUIRED_PKGS=(streamlit fastapi uvicorn faiss sentence_transformers networkx pandas openai langgraph langchain_core langgraph.checkpoint.sqlite)

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

bootstrap_env_file() {
  hr "Environment file (.env)"
  if [[ ! -f "$ROOT/.env" ]] && [[ -f "$ROOT/.env.example" ]]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    ok "created .env from .env.example"
    warn "set OPENAI_API_KEY in .env to enable LLM modes (retrieval-only still works)"
  elif [[ -f "$ROOT/.env" ]]; then
    ok ".env present"
  else
    warn "no .env or .env.example — defaults from config.py will be used"
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

echo ""

# ─ 6a. FastAPI ──
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

cat <<DONE

✅ Stack is up.

   API        http://localhost:$API_PORT/docs
   Streamlit  http://localhost:$STREAMLIT_PORT
   Web        http://localhost:$WEB_PORT

   Logs:    .run/logs/{api,streamlit,web}.log
   PIDs:    .run/{api,streamlit,web}.pid

   Stop everything:  ./stop.sh
   Status check:     ./status.sh
   Tail a service:   tail -f .run/logs/api.log
DONE
