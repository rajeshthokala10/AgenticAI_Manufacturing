import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
EXCEL_DIR = DATA_DIR / "excel"
PROCESSED_DIR = DATA_DIR / "processed"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5
RRF_K = 60
MAX_CRITIC_RETRIES = 2

CHROMA_COLLECTION = "manufacturing_docs"
CHROMA_DIR = str(PROCESSED_DIR / "chromadb")
GRAPH_PATH = str(PROCESSED_DIR / "knowledge_graph.gpickle")

DOMAIN_ONTOLOGY = {
    "entity_types": [
        "Equipment", "Component", "Alarm", "FailureMode",
        "Symptom", "Cause", "Procedure", "SparePart", "Specification"
    ],
    "relation_types": [
        "HAS_COMPONENT", "TRIGGERS_ALARM", "CAUSES_FAILURE",
        "HAS_SYMPTOM", "RESOLVED_BY", "REQUIRES_PART",
        "FOLLOWS_PROCEDURE", "HAS_SPECIFICATION"
    ],
    "traversal_routes": {
        "symptom_to_fix": ["Symptom", "Cause", "FailureMode", "Procedure"],
        "alarm_to_procedure": ["Alarm", "Equipment", "FailureMode", "Procedure"],
        "equipment_to_parts": ["Equipment", "Component", "SparePart"],
    }
}
