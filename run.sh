#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Manufacturing Hybrid GraphRAG — bring up all three services in the background.
#
#   - api        → FastAPI  (uvicorn)   :8000
#   - streamlit  → Streamlit UI         :8501
#   - web        → Next.js  (dev)       :3000
#
# Logs and PID files are written to ./.run/
# Use stop.sh to bring everything down.
#
# Overridable via environment:
#   PYTHON_BIN  (default: auto-detect anaconda3, then python3)
#   API_PORT    (default: 8000)
#   STREAMLIT_PORT (default: 8501)
#   WEB_PORT    (default: 3000)
#   SKIP_WEB=1       skip the Next.js UI
#   SKIP_STREAMLIT=1 skip the Streamlit UI
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Resolve Python ──────────────────────────────────────────────────────────
if [[ -n "${PYTHON_BIN:-}" ]] && command -v "$PYTHON_BIN" >/dev/null; then
  PY="$PYTHON_BIN"
elif [[ -x "/opt/anaconda3/bin/python" ]]; then
  PY="/opt/anaconda3/bin/python"
elif command -v python3 >/dev/null; then
  PY="$(command -v python3)"
else
  echo "❌ No Python interpreter found. Set PYTHON_BIN=/path/to/python and retry." >&2
  exit 1
fi

API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
WEB_PORT="${WEB_PORT:-3000}"

# ── helpers ─────────────────────────────────────────────────────────────────
free_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "  · port $port busy (pids: $pids) — releasing"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

wait_for_url() {
  local url="$1" label="$2" tries="${3:-60}"
  for ((i = 1; i <= tries; i++)); do
    if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -qE "^(200|2[0-9]{2})$"; then
      echo "  ✓ $label up — $url"
      return 0
    fi
    sleep 1
  done
  echo "  ⚠ $label did not respond at $url within ${tries}s (continuing)"
  return 1
}

bg_run() {
  local name="$1"; shift
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"
  echo "─ starting $name → logs: ${logfile#"$ROOT/"}"
  (
    cd "$ROOT"
    nohup "$@" >>"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )
  sleep 0.4
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "❌ $name failed to start. See $logfile"
    tail -n 30 "$logfile" || true
    exit 1
  fi
}

# ── banner ──────────────────────────────────────────────────────────────────
cat <<BANNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Manufacturing Hybrid GraphRAG — run.sh
  python : $PY
  root   : $ROOT
  ports  : api=$API_PORT  streamlit=$STREAMLIT_PORT  web=$WEB_PORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANNER

# ── 1. FastAPI backend ──────────────────────────────────────────────────────
echo "▶ FastAPI backend (api/server.py)"
free_port "$API_PORT"
PYTHONPATH="$ROOT" bg_run "api" "$PY" -m uvicorn api.server:app \
  --host 0.0.0.0 --port "$API_PORT"
wait_for_url "http://localhost:$API_PORT/api/health" "api" 120

# ── 2. Streamlit ────────────────────────────────────────────────────────────
if [[ "${SKIP_STREAMLIT:-0}" != "1" ]]; then
  echo "▶ Streamlit UI (app.py)"
  free_port "$STREAMLIT_PORT"
  bg_run "streamlit" "$PY" -m streamlit run app.py \
    --server.port "$STREAMLIT_PORT" --server.headless true \
    --browser.gatherUsageStats false
  wait_for_url "http://localhost:$STREAMLIT_PORT/_stcore/health" "streamlit" 60
else
  echo "▶ Streamlit UI (skipped — SKIP_STREAMLIT=1)"
fi

# ── 3. Next.js UI ───────────────────────────────────────────────────────────
if [[ "${SKIP_WEB:-0}" != "1" ]]; then
  if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
    echo "▶ Next.js UI — skipped (Node/npm not on PATH). Install Node 18+ to enable."
  else
    echo "▶ Next.js UI (web/)"
    if [[ ! -d "$ROOT/web/node_modules" ]]; then
      echo "  · installing npm dependencies (first run only)…"
      (cd "$ROOT/web" && npm install --silent --no-audit --no-fund \
        >>"$LOG_DIR/web-install.log" 2>&1) || {
          echo "❌ npm install failed. See $LOG_DIR/web-install.log"
          exit 1
        }
    fi
    free_port "$WEB_PORT"
    (
      cd "$ROOT/web"
      NEXT_PUBLIC_API_ORIGIN="http://localhost:$API_PORT" \
      nohup npm run dev -- --port "$WEB_PORT" \
        >>"$LOG_DIR/web.log" 2>&1 &
      echo $! >"$RUN_DIR/web.pid"
    )
    wait_for_url "http://localhost:$WEB_PORT/" "web" 120 || true
  fi
fi

# ── summary ────────────────────────────────────────────────────────────────
cat <<DONE

✅ Stack is up.

   API        http://localhost:$API_PORT/docs
   Streamlit  http://localhost:$STREAMLIT_PORT
   Web        http://localhost:$WEB_PORT

   Logs:    .run/logs/{api,streamlit,web}.log
   PIDs:    .run/{api,streamlit,web}.pid

   Stop everything:  ./stop.sh
   Tail a service:   tail -f .run/logs/api.log
DONE
