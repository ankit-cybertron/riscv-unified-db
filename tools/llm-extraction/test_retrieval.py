"""
test_retrieval.py
Verifies the ChromaDB vector database maintains required Top-3 retrieval accuracy
across different semantic categories.
"""
import argparse

from config import logger, DB_DIR
import chromadb
from chromadb.utils import embedding_functions


TESTS = [
    ("legal values of mtvec mode field",                          "MTVEC_MODES"),
    ("valid values of mstatus fs field",                          "MSTATUS_FS_LEGAL_VALUES"),
    ("maximum ASID width",                                        "ASID_WIDTH"),
    ("range of mtval width register",                             "MTVAL_WIDTH"),
    ("parameters depending on mxlen register width",              "MXLEN"),
    ("what depends on misaligned load store support",             "MISALIGNED_LDST"),
    ("mtvec base alignment constraint",                           "MTVEC_BASE_ALIGNMENT_DIRECT"),
    ("mtvec register access read only or read write",             "MTVEC_ACCESS"),
    ("how hardware handles misaligned memory access",             "MISALIGNED_LDST"),
    ("machine register width 32 or 64 bit",                      "MXLEN"),
    ("does illegal wlrl field cause trap",                        "TRAP_ON_ILLEGAL_WLRL"),
    ("illegal instruction encoding stored in mtval",              "REPORT_ENCODING_IN_MTVAL_ON_ILLEGAL_INSTRUCTION"),
    ("number of physical memory protection entries",              "NUM_PMP_ENTRIES"),
    ("physical address width of processor",                       "PHYS_ADDR_WIDTH"),
    ("reservation strategy for load reserved store conditional",  "LRSC_RESERVATION_STRATEGY"),
    ("hardware update of floating point dirty bit fs",            "HW_MSTATUS_FS_DIRTY_UPDATE"),
]


def run(db_dir=None):
    db_dir = db_dir or DB_DIR
    if not db_dir.exists():
        logger.error(f"Database not found at {db_dir}. Run: python pipeline.py params")
        return

    client = chromadb.PersistentClient(path=str(db_dir))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        collection = client.get_collection(name="udb_parameters", embedding_function=emb_fn)
    except Exception:
        logger.error("Collection 'udb_parameters' not found. Run: python pipeline.py params")
        return

    top1 = top3 = top5 = 0
    for query, expected in TESTS:
        results = collection.query(query_texts=[query], n_results=5)
        ids = results["ids"][0] if results["ids"] else []

        if ids and ids[0] == expected: top1 += 1
        if expected in ids[:3]:        top3 += 1
        if expected in ids[:5]:        top5 += 1

        if expected in ids[:3]:
            logger.info(f"✓  {query[:55]} → {expected}")
        else:
            logger.warning(f"✗  {query[:55]}\n   expected={expected}  got={ids}")

    total = len(TESTS)
    logger.info(f"\nTop-1: {top1}/{total} ({top1*100//total}%)")
    logger.info(f"Top-3: {top3}/{total} ({top3*100//total}%)")
    logger.info(f"Top-5: {top5}/{total} ({top5*100//total}%)")

    if top3 / total >= 0.9:
        logger.info("PASS — meets ≥90% Top-3 accuracy target")
    else:
        logger.error("FAIL — below required 90% Top-3 threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test parameter retrieval accuracy.")
    parser.add_argument("--db-dir", type=lambda p: __import__("pathlib").Path(p), default=None,
                        help="Path to ChromaDB directory (default: chroma_db/)")
    args = parser.parse_args()
    run(db_dir=args.db_dir)