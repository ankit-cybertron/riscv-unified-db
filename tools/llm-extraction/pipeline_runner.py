"""
Purpose:
    Orchestration logic for the RISC-V UDB extraction pipeline.
    Receives a subcommand from main.py and dispatches to the correct stage.

Pipeline Stage:
    all (orchestrates ingest → process → embed → export)

Inputs:
    - (delegates to per-stage modules; see each _run_* function)

Outputs:
    - (delegates to per-stage modules; see each _run_* function)

Core Responsibilities:
    - Set up sys.path so chunk/ and configs/ are importable from any cwd
    - Implement _run_ingest, _run_process, _run_embed, _run_export
    - Dispatch run(subcommand, mode) → correct stage(s)

Key Assumptions:
    - Called by main.py; never run directly
    - chunk/ modules are imported lazily inside each _run_* to avoid circular deps

Failure Modes:
    - sys.exit(1) on AsciiDocChunker failure or unknown subcommand
    - ModuleNotFoundError if .venv is not activated before running

Notes:
    - Lazy imports (inside _run_*) keep startup time fast for --dry-run
"""

import subprocess
import sys
from pathlib import Path

# sys.path setup: both chunk/ and configs/ must be importable from any cwd.
_TOOL_DIR  = Path(__file__).parent.resolve()            # llm-extraction/
_CHUNK_DIR = _TOOL_DIR / "chunk"

if str(_CHUNK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHUNK_DIR))
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

from configs.config import (          # noqa: E402  (after sys.path setup)
    logger,
    ISA_MANUAL_DIR,
    ISA_MANUAL_REPO_URL,
    OUTPUT_DIR,
    DATA_DIR,
    CHUNKS_PATH,
)



def _run_ingest() -> None:
    """Clone riscv-isa-manual on first run; fetch + reset to HEAD on subsequent runs."""
    logger.info("=== ingest: riscv-isa-manual ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ISA_MANUAL_DIR.exists():
        logger.info(f"Cloning ISA manual into {ISA_MANUAL_DIR} …")
        subprocess.run(
            ["git", "clone", "--depth=1", ISA_MANUAL_REPO_URL, str(ISA_MANUAL_DIR)],
            check=True,
        )
    else:
        logger.info(f"ISA manual already present at {ISA_MANUAL_DIR}; pulling latest …")
        subprocess.run(
            ["git", "-C", str(ISA_MANUAL_DIR), "fetch", "--depth=1", "origin", "main"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(ISA_MANUAL_DIR), "reset", "--hard", "origin/main"],
            check=True,
        )

    logger.info("ingest: complete.")


def _run_process() -> None:
    """Run AsciiDocChunker: raw + filtered chunks, CSV, and filter-stats report."""
    logger.info("=== process: spec chunking ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from pipeline.ingest.chunker_adoc import AsciiDocChunker  # noqa: PLC0415

    chunker = AsciiDocChunker(ISA_MANUAL_DIR)
    success = chunker.run()
    if success:
        import shutil
        if ISA_MANUAL_DIR.exists():
            shutil.rmtree(ISA_MANUAL_DIR)
            logger.info(f"Cleaned up {ISA_MANUAL_DIR}")
        logger.info("process: complete.")
    else:
        logger.error("process: AsciiDocChunker reported a failure.")
        sys.exit(1)


def _run_embed(mode: str = "rag") -> None:
    """Build ChromaDB vector index. mode='analysis' also writes full corpus JSON."""
    logger.info(f"=== embed: vector database (mode={mode}) ===")
    logger.warning("run_embed is not fully complete and build_vector_db was removed.")
    # TODO: integrate with new RAG embedder once implemented
    logger.info("embed: stub complete.")


def _run_export(mode: str = "rag") -> None:
    """
    Standalone hook to refresh reports and corpus without a full re-embed.
    """
    logger.info(f"=== export (mode={mode}) ===")
    logger.warning("run_export is incomplete pending new report logic.")
    logger.info("export: stub complete.")




def run(subcommand: str, mode: str = "rag") -> None:
    """Dispatch subcommand to the correct pipeline stage(s)."""
    dispatch = {
        "ingest":  lambda: _run_ingest(),
        "process": lambda: _run_process(),
        "embed":   lambda: _run_embed(mode=mode),
        "export":  lambda: _run_export(mode=mode),
        "all": lambda: (
            _run_ingest(),
            _run_process(),
            _run_embed(mode=mode),
        ),
    }

    handler = dispatch.get(subcommand)
    if handler is None:
        logger.error(f"Unknown subcommand: {subcommand!r}")
        sys.exit(1)

    handler()
    logger.info(f"Pipeline '{subcommand}' finished successfully.")
   
    