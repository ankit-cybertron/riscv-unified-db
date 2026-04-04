from pathlib import Path
from ruamel.yaml import YAML
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("udb")

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

PARAM_DIR   = REPO_ROOT / "spec" / "std" / "isa" / "param"
CSR_DIR     = REPO_ROOT / "spec" / "std" / "isa" / "csr"
EXT_DIR     = REPO_ROOT / "spec" / "std" / "isa" / "ext"
PROSE_DIR   = REPO_ROOT / "spec" / "std" / "isa" / "prose"
SCHEMA_DEFS = REPO_ROOT / "spec" / "schemas" / "schema_defs.json"

OUTPUT_DIR          = SCRIPT_DIR / "output"
CORPUS_PATH         = OUTPUT_DIR / "param_corpus.json"
ANALYSIS_CORPUS_PATH = OUTPUT_DIR / "param_analysis_corpus.json"
REPORT_PATH         = OUTPUT_DIR / "UDB_PARAMETER_DATABASE_REPORT.md"
DB_DIR              = SCRIPT_DIR / "chroma_db"
GRAPH_PATH          = OUTPUT_DIR / "dependency_graph.json"
CHUNKS_PATH         = OUTPUT_DIR / "chunks.json"
FILTER_STATS_PATH   = OUTPUT_DIR / "filter_stats.md"

MOCK_PREFIX = "MOCK_"

CHUNK_KEYWORDS = {
    "optionality": ["should", "may", "can", "either", "optionally"],
    "warl_related": ["warl", "wlrl", "legal", "implementation", "configurable"],
}

DIR_TO_CHUNK_TYPE = {
    "ext":   "extension_desc",
    "csr":   "csr_field_desc",
    "param": "param_desc",
    "prose": "prose",
}

CSR_IDL_KEYS = {
    "sw_write(csr_value)",
    "type()",
    "reset_value()",
    "legal?(csr_value)",
    "sw_read()",
}

yaml = YAML()
yaml.preserve_quotes = True

try:
    with open(SCHEMA_DEFS, "r", encoding="utf-8") as _f:
        SCHEMA_DEFS_DATA = json.load(_f)
except Exception as _e:
    logger.warning(f"Could not load schema_defs.json: {_e}")
    SCHEMA_DEFS_DATA = {}
