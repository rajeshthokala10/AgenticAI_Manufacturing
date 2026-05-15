#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stop every background service started by run.sh.
# Uses PID files first, then falls back to port-based discovery.
# ---------------------------------------------------------------------------

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"

API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
WEB_PORT="${WEB_PORT:-3000}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-kg-rag-qdrant}"

kill_tree() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  local kids
  kids="$(pgrep -P "$pid" 2>/dev/null || true)"
  for kid in $kids; do
    kill_tree "$kid"
  done
  kill -TERM "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.3
  done
  kill -KILL "$pid" 2>/dev/null || true
}

stop_by_pidfile() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      echo "  · $name (pid=$pid)"
      kill_tree "$pid"
    fi
    rm -f "$pidfile"
  fi
}

free_port() {
  local port="$1" label="$2"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "  · $label port $port still busy (pids: $pids) — force killing"
    kill -9 $pids 2>/dev/null || true
  fi
}

echo "━ Stopping Manufacturing Hybrid GraphRAG stack ━"

for svc in api streamlit web; do
  stop_by_pidfile "$svc"
done

# Belt-and-suspenders: clear any lingering listeners on our ports.
sleep 0.6
free_port "$API_PORT"        "api"
free_port "$STREAMLIT_PORT"  "streamlit"
free_port "$WEB_PORT"        "web"

# Stop the Qdrant container started by run.sh (if any).
if command -v docker >/dev/null 2>&1; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$QDRANT_CONTAINER"; then
    echo "  · qdrant container ($QDRANT_CONTAINER)"
    docker rm -f "$QDRANT_CONTAINER" >/dev/null 2>&1 || true
  fi
fi
rm -f "$RUN_DIR/qdrant.cid"
free_port "$QDRANT_PORT"     "qdrant"

# Sweep up any orphan child processes by name match (best-effort).
for pat in "uvicorn api.server" "streamlit run app.py" "next dev"; do
  pids="$(pgrep -f "$pat" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "  · sweeping orphan: $pat (pids: $pids)"
    kill -9 $pids 2>/dev/null || true
  fi
done

echo "✓ All services stopped."
