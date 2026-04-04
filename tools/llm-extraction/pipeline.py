"""
pipeline.py — run the RISC-V UDB extraction pipeline.

Commands:
    python pipeline.py              # run everything (params + chunk)
    python pipeline.py params       # build parameter database + search index only
    python pipeline.py chunk        # extract and filter spec text chunks only
    python pipeline.py --full       # run everything + write deep analysis corpus
"""
import argparse

from config import logger, CHUNKS_PATH


def build_params(full: bool = False):
    from build_vector_db import Pipeline
    logger.info("Building parameter database...")
    Pipeline(mode="analysis" if full else "rag").run()
    logger.info("Parameter database ready.")


def chunk_spec():
    from spec_chunker import SpecChunker
    logger.info("Chunking spec text...")
    SpecChunker(CHUNKS_PATH.parent.parent.parent.parent).run()
    logger.info("Spec chunking complete.")


def main():
    parser = argparse.ArgumentParser(
        description="RISC-V UDB LLM Extraction Pipeline",
        epilog=(
            "Examples:\n"
            "  python pipeline.py            # run everything\n"
            "  python pipeline.py params     # parameter database only\n"
            "  python pipeline.py chunk      # spec text chunking only\n"
            "  python pipeline.py --full     # run everything + full analysis corpus"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command", nargs="?", choices=["params", "chunk"],
        help="'params' = build search database  |  'chunk' = extract spec text  |  (omit to run both)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Also write the complete analysis corpus (larger output, used for deep inspection)",
    )
    args = parser.parse_args()

    if args.command == "params":
        build_params(full=args.full)
    elif args.command == "chunk":
        chunk_spec()
    else:
        build_params(full=args.full)
        chunk_spec()


if __name__ == "__main__":
    main()
