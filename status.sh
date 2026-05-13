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

# ── HITL approval gate ────────────────────────────────────────────────────
parse_env_flag() {
  local key="$1" default="$2"
  if [[ -f "$ROOT/.env" ]]; then
    local val
    val="$(grep -E "^${key}=" "$ROOT/.env" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'" || true)"
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    [[ -n "$val" ]] && { printf "%s" "$val"; return; }
  fi
  printf "%s" "$default"
}
HITL_FLAG="$(parse_env_flag USE_HITL false)"
LG_FLAG="$(parse_env_flag USE_LANGGRAPH false)"

echo
echo "HITL approval gate:"
if [[ "$HITL_FLAG" =~ ^(1|true|yes|on)$ ]] && [[ "$LG_FLAG" =~ ^(1|true|yes|on)$ ]]; then
  printf "  USE_HITL=%s · USE_LANGGRAPH=%s · backend=%s\n" \
    "$HITL_FLAG" "$LG_FLAG" "$(parse_env_flag HITL_CHECKPOINT_BACKEND sqlite)"

  # Pending queue
  pending_json="$(curl -s --max-time 3 "http://localhost:$API_PORT/api/approvals/pending" 2>/dev/null || true)"
  if [[ -n "$pending_json" ]]; then
    count="$(printf "%s" "$pending_json" | grep -oE '"count":[ ]*[0-9]+' | head -n1 | grep -oE '[0-9]+' || echo "?")"
    printf "  pending      %s approval(s) awaiting review\n" "${count:-?}"
  else
    printf "  pending      api not reachable\n"
  fi

  # Audit stats
  audit_json="$(curl -s --max-time 3 "http://localhost:$API_PORT/api/audit?limit=1" 2>/dev/null || true)"
  if [[ -n "$audit_json" ]]; then
    total="$(printf "%s" "$audit_json" | grep -oE '"total":[ ]*[0-9]+' | head -n1 | grep -oE '[0-9]+' || echo "?")"
    approved="$(printf "%s" "$audit_json" | grep -oE '"approved":[ ]*[0-9]+' | head -n1 | grep -oE '[0-9]+' || echo "?")"
    rejected="$(printf "%s" "$audit_json" | grep -oE '"rejected":[ ]*[0-9]+' | head -n1 | grep -oE '[0-9]+' || echo "?")"
    rate="$(printf "%s" "$audit_json" | grep -oE '"approval_rate":[ ]*[0-9.]+' | head -n1 | grep -oE '[0-9.]+' || echo "?")"
    printf "  audit log    total=%s  approved=%s  rejected=%s  approval_rate=%s\n" \
      "${total:-?}" "${approved:-?}" "${rejected:-?}" "${rate:-?}"
  fi

  db_path="$(parse_env_flag HITL_DB_PATH "data/processed/audit.sqlite")"
  if [[ -f "$ROOT/$db_path" ]]; then
    sz="$(ls -lh "$ROOT/$db_path" 2>/dev/null | awk '{print $5}')"
    printf "  sqlite db    %s (%s)\n" "$db_path" "$sz"
  else
    printf "  sqlite db    %s (not yet created)\n" "$db_path"
  fi
else
  printf "  disabled  (set USE_HITL=true + USE_LANGGRAPH=true in .env to enable)\n"
fi
