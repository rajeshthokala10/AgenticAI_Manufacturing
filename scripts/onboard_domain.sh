#!/usr/bin/env bash
# Onboard a new domain into the Manufacturing Hybrid GraphRAG pipeline.
#
# What this script does, in order:
#
#   1. Validates the domain name.
#   2. Places (or links) the schema YAML at ``schemas/<domain>.yaml``.
#   3. Ensures ``doc_pipeline/input_docs/<domain>/`` exists; optionally
#      copies a directory of seed documents into it.
#   4. Optionally runs a converter script (e.g. JSONL → .txt) before
#      ingestion.
#   5. Calls ``python main.py --rebuild --domain <name> --no-llm`` to build
#      the per-domain Qdrant collection + KG file.
#   6. Verifies the domain shows up in ``config.DOMAINS`` and prints the
#      resulting node / edge / vector counts.
#
# The rest of the architecture (Streamlit selector, FastAPI ``/api/domains``,
# Next.js header) auto-discovers any schema dropped into ``schemas/`` — no
# Python or TypeScript edits are required.
#
# Usage
# -----
#
#   scripts/onboard_domain.sh --domain medical \
#       --schema /tmp/medical.yaml \
#       --docs   /tmp/medical_pdfs/
#
#   scripts/onboard_domain.sh --domain aviation     # rebuild only
#
# Flags
# -----
#   --domain  NAME        (required) lowercase identifier (a-z, 0-9, _).
#   --schema  PATH        copy PATH → schemas/<domain>.yaml. Required on
#                         first onboarding; omit to rebuild an existing one.
#   --docs    DIR         copy DIR/* → doc_pipeline/input_docs/<domain>/
#                         (recursive). Skip if you've already staged them.
#   --convert CMD         optional shell command run after staging.
#                         Useful for JSONL/CSV → .txt converters.
#                         Example:  --convert "python scripts/my_conv.py --out OUTDIR"
#                         The literal ``OUTDIR`` is replaced with the
#                         domain's input directory.
#   --rebuild-only        skip schema + docs staging; just rebuild.
#   --keep-llm            do NOT pass --no-llm to main.py (slower; runs the
#                         orchestrator init too). Default is --no-llm.
#   -h | --help           show this help.
#
# Exit codes: 0 success · 64 usage · 65 missing input · 70 rebuild failed.

set -euo pipefail

# ─── argv parsing ───────────────────────────────────────────────────────────

domain=""
schema_src=""
docs_src=""
convert_cmd=""
rebuild_only=0
keep_llm=0

print_help() { sed -n '2,/^# Exit codes/p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)        domain="${2:-}"; shift 2 ;;
    --schema)        schema_src="${2:-}"; shift 2 ;;
    --docs)          docs_src="${2:-}"; shift 2 ;;
    --convert)       convert_cmd="${2:-}"; shift 2 ;;
    --rebuild-only)  rebuild_only=1; shift ;;
    --keep-llm)      keep_llm=1; shift ;;
    -h|--help)       print_help; exit 0 ;;
    *) echo "onboard_domain.sh: unknown flag $1" >&2; exit 64 ;;
  esac
done

if [[ -z "$domain" ]]; then
  echo "onboard_domain.sh: --domain is required" >&2
  print_help
  exit 64
fi

if ! [[ "$domain" =~ ^[a-z][a-z0-9_]*$ ]]; then
  echo "onboard_domain.sh: --domain must be lowercase a-z/0-9/underscore (got: $domain)" >&2
  exit 64
fi

# ─── paths ──────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA_DST="$ROOT/schemas/$domain.yaml"
INPUT_DST="$ROOT/doc_pipeline/input_docs/$domain"
PROCESSED_DIR="$ROOT/data/processed"

cd "$ROOT"

# ─── step 1: stage schema ──────────────────────────────────────────────────

if [[ $rebuild_only -eq 0 ]]; then
  if [[ -n "$schema_src" ]]; then
    if [[ ! -f "$schema_src" ]]; then
      echo "onboard_domain.sh: --schema file not found: $schema_src" >&2
      exit 65
    fi
    echo "[1/5] Copying schema → $SCHEMA_DST"
    cp "$schema_src" "$SCHEMA_DST"
  elif [[ ! -f "$SCHEMA_DST" ]]; then
    echo "onboard_domain.sh: $SCHEMA_DST missing and no --schema given" >&2
    echo "                  pass --schema PATH the first time you onboard a domain." >&2
    exit 65
  else
    echo "[1/5] Schema already present at $SCHEMA_DST — keeping as is"
  fi
else
  echo "[1/5] --rebuild-only: skipping schema staging"
fi

# Validate the schema parses + the ``domain:`` field matches the flag.
echo "[2/5] Validating schema (auto-discovery + domain id match)"
python - <<PY
import sys
sys.path.insert(0, "$ROOT")
import importlib
for m in list(sys.modules):
    if m.startswith("config") or m.startswith("core."):
        del sys.modules[m]
import config
from core.kg.schema import load_schema
s = load_schema("$SCHEMA_DST")
if s.domain.strip().lower() != "$domain":
    sys.exit(f"schema domain id {s.domain!r} does not match --domain '$domain'")
if "$domain" not in config.DOMAINS:
    sys.exit(f"auto-discovery missed '$domain'; check that schemas/$domain.yaml lives under {config.SCHEMAS_DIR}")
print(f"  schema OK · domain id={s.domain} · entities={len(s.entity_types)} · edges={len(s.edge_types)}")
print(f"  registry  · DOMAINS={config.DOMAINS}")
PY

# ─── step 3: stage input docs ──────────────────────────────────────────────

mkdir -p "$INPUT_DST"
if [[ $rebuild_only -eq 0 && -n "$docs_src" ]]; then
  if [[ ! -d "$docs_src" ]]; then
    echo "onboard_domain.sh: --docs directory not found: $docs_src" >&2
    exit 65
  fi
  echo "[3/5] Copying $docs_src/* → $INPUT_DST/"
  # rsync preserves directory structure; ``-a`` keeps mtimes.
  rsync -a "$docs_src"/ "$INPUT_DST"/
elif [[ $rebuild_only -eq 0 ]]; then
  echo "[3/5] No --docs given; $INPUT_DST left untouched ($(find "$INPUT_DST" -type f | wc -l) files already present)"
else
  echo "[3/5] --rebuild-only: skipping docs staging"
fi

# ─── step 4: optional converter ────────────────────────────────────────────

if [[ -n "$convert_cmd" ]]; then
  resolved="${convert_cmd//OUTDIR/$INPUT_DST}"
  echo "[4/5] Running converter: $resolved"
  bash -c "$resolved"
else
  echo "[4/5] No --convert command; skipping converter"
fi

# ─── step 5: rebuild ───────────────────────────────────────────────────────

extra=()
[[ $keep_llm -eq 0 ]] && extra+=(--no-llm)

echo "[5/5] python main.py --rebuild --domain $domain ${extra[*]}"
if ! python main.py --rebuild --domain "$domain" "${extra[@]}"; then
  echo "onboard_domain.sh: rebuild failed for domain $domain" >&2
  exit 70
fi

# ─── verify ────────────────────────────────────────────────────────────────

echo
echo "── Build report for '$domain' ────────────────────────────────────────"
python - <<PY
import json, sys
sys.path.insert(0, "$ROOT")
from core.knowledge_graph import KnowledgeGraph
import config

kg = KnowledgeGraph(domain="$domain")
kg.load()

# Vector-store manifest is the canonical chunk count.
manifest_path = config.VECTOR_STORE_DIR / f"{config.index_name('$domain')}.manifest.json"
n_chunks = "?"
if manifest_path.exists():
    n_chunks = json.loads(manifest_path.read_text()).get("n_chunks", "?")

print(f"  Qdrant collection : {config.qdrant_collection('$domain')}   ({n_chunks} vectors)")
print(f"  KG file           : {config.kg_path('$domain').relative_to(config.BASE_DIR)}")
print(f"  KG nodes          : {kg.graph.number_of_nodes()}")
print(f"  KG edges          : {kg.graph.number_of_edges()}")
print(f"  Schema rejects    : {len(kg._rejected)}")
print()
print(f"  Domain registry   : {config.DOMAINS}")
print(f"  Default domain    : {config.DEFAULT_DOMAIN}")
PY

echo
echo "✓ Domain '$domain' is live. The sidebar selector, /api/domains, and the"
echo "  Next.js header pick it up automatically on the next restart of run.sh."
