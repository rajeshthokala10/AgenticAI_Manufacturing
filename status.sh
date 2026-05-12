#!/usr/bin/env bash
# Quick status overview for all three services.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"

API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
WEB_PORT="${WEB_PORT:-3000}"

probe() {
  local label="$1" url="$2" port="$3" pidfile="$RUN_DIR/$4.pid"
  local pid="-"
  [[ -f "$pidfile" ]] && pid="$(cat "$pidfile" 2>/dev/null || echo '-')"
  local lpid
  lpid="$(lsof -ti tcp:"$port" 2>/dev/null | head -n1 || true)"
  local http="-"
  http="$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null || echo "-")"
  printf "  %-10s  pid=%-7s  port=%-4s  listener=%-7s  http=%-3s  %s\n" \
    "$label" "$pid" "$port" "${lpid:--}" "$http" "$url"
}

echo "━ Manufacturing Hybrid GraphRAG · status ━"
probe "api"       "http://localhost:$API_PORT/api/health"     "$API_PORT"       "api"
probe "streamlit" "http://localhost:$STREAMLIT_PORT/_stcore/health" "$STREAMLIT_PORT" "streamlit"
probe "web"       "http://localhost:$WEB_PORT/"               "$WEB_PORT"       "web"

echo
echo "Install state:"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  ver="$("$ROOT/.venv/bin/python" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' 2>/dev/null || echo "?")"
  printf "  python venv  v%-7s  %s\n" "$ver" "$ROOT/.venv"
else
  printf "  python venv  -          (base Python in use; run.sh skipped venv)\n"
fi
if [[ -d "$ROOT/web/node_modules" ]]; then
  pkgs="$(ls "$ROOT/web/node_modules" 2>/dev/null | wc -l | tr -d ' ')"
  printf "  web/ deps    %-9s  (web/node_modules)\n" "${pkgs} pkgs"
else
  printf "  web/ deps    not installed (run: cd web && npm install)\n"
fi
if [[ -f "$ROOT/.env" ]]; then
  printf "  .env         present\n"
else
  printf "  .env         missing\n"
fi

if [[ -d "$RUN_DIR/logs" ]]; then
  echo
  echo "Logs:"
  ls -lh "$RUN_DIR/logs" 2>/dev/null | tail -n +2 | awk '{printf "  %s  %s\n", $5, $9}'
fi
