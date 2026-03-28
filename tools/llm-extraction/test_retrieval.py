"""
RISC-V Parameter Retrieval Tests.

Improvements:
- Multi-category test coverage
- Top-1 / Top-3 / Top-5 accuracy
- Better semantic queries
- Failure diagnostics
"""

import argparse
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DB_DIR = SCRIPT_DIR / "chroma_db"


def test_retrieval(db_dir: Path):
    if not db_dir.exists():
        logger.error(f"Database not found: {db_dir}. Run build_vector_db.py first.")
        return

    client = chromadb.PersistentClient(path=str(db_dir))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            name="udb_parameters",
            embedding_function=emb_fn
        )
    except:
        logger.error("Collection 'udb_parameters' not found.")
        return
    
    # TEST SUITES 
    tests = [
        # VALUE / ENUM
        ("legal values of mtvec mode field", "MTVEC_MODES"),
        ("valid values of mstatus fs field", "MSTATUS_FS_LEGAL_VALUES"),

        # RANGE
        ("maximum ASID width", "ASID_WIDTH"),
        ("range of mtval width register", "MTVAL_WIDTH"),

        # DEPENDENCY
        ("parameters depending on mxlen register width", "MXLEN"),
        ("what depends on misaligned load store support", "MISALIGNED_LDST"),

        # CSR CONTEXT
        ("mtvec base alignment constraint", "MTVEC_BASE_ALIGNMENT_DIRECT"),
        ("mtvec register access read only or read write", "MTVEC_ACCESS"),

        # SEMANTIC
        ("how hardware handles misaligned memory access", "MISALIGNED_LDST"),
        ("machine register width 32 or 64 bit", "MXLEN"),

        # TRAP / BEHAVIOR
        ("does illegal wlrl field cause trap", "TRAP_ON_ILLEGAL_WLRL"),
        ("illegal instruction encoding stored in mtval", "REPORT_ENCODING_IN_MTVAL_ON_ILLEGAL_INSTRUCTION"),

        # ARCHITECTURAL
        ("number of physical memory protection entries", "NUM_PMP_ENTRIES"),
        ("physical address width of processor", "PHYS_ADDR_WIDTH"),

        # LRSC
        ("reservation strategy for load reserved store conditional", "LRSC_RESERVATION_STRATEGY"),

        # HARDWARE UPDATE
        ("hardware update of floating point dirty bit fs", "HW_MSTATUS_FS_DIRTY_UPDATE"),
    ]

    logger.info("\n── Enhanced Retrieval Evaluation ───────────────────────────────")

    top1_pass = 0
    top3_pass = 0
    top5_pass = 0

    for query, expected in tests:
        results = collection.query(query_texts=[query], n_results=5)
        top_ids = results["ids"][0] if results["ids"] else []

        hit_top1 = expected == top_ids[0] if top_ids else False
        hit_top3 = expected in top_ids[:3]
        hit_top5 = expected in top_ids[:5]

        if hit_top1:
            top1_pass += 1
        if hit_top3:
            top3_pass += 1
        if hit_top5:
            top5_pass += 1

        if hit_top3:
            logger.info(f"✓ PASS  {query[:50]} -> {expected}")
        else:
            logger.warning(
                f"✗ FAIL  {query[:50]}\n"
                f"   Expected: {expected}\n"
                f"   Got Top5: {top_ids}"
            )

    total = len(tests)

    logger.info("\n────────── FINAL METRICS ──────────")
    logger.info(f"Top-1 Accuracy: {top1_pass}/{total} ({int((top1_pass/total)*100)}%)")
    logger.info(f"Top-3 Accuracy: {top3_pass}/{total} ({int((top3_pass/total)*100)}%)")
    logger.info(f"Top-5 Accuracy: {top5_pass}/{total} ({int((top5_pass/total)*100)}%)")

    # acceptance criteria
    if top3_pass / total >= 0.9:
        logger.info("\nPASS: Retrieval system meets 90% Top-3 accuracy")
    else:
        logger.error("\nFAIL: Retrieval system below required threshold")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Parameter Retrieval")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    args = parser.parse_args()

    test_retrieval(args.db_dir)