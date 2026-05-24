import argparse
from pathlib import Path
from pprint import pprint
import sys

from dotenv import load_dotenv

from .config import SelfRagConfig
from .graph import run_self_rag
from .ingestion import discover_ingest_candidates, format_batch_report, run_batch_ingestion


def main(argv=None) -> None:
    load_dotenv()
    configure_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ingest":
        run_ingest_cli(argv[1:])
        return
    run_query_cli(argv)


def run_query_cli(argv) -> None:
    parser = argparse.ArgumentParser(description="Run the independent LangGraph Self-RAG flow.")
    parser.add_argument("question", help="Question to ask against the indexed document store.")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE retrieval expansion.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking.")
    args = parser.parse_args(argv)

    config = SelfRagConfig()
    if args.no_hyde:
        config.use_hyde = False
    if args.no_rerank:
        config.use_rerank = False
    result = run_self_rag(args.question, config=config)

    print("\nAnswer:\n")
    print(result.get("generation") or result.get("error") or "No answer generated.")
    print("\nTrace summary:\n")
    pprint(
        {
            "documents": len(result.get("documents", [])),
            "attempt_count": result.get("attempt_count", 0),
            "retrieval_attempt_count": result.get("retrieval_attempt_count", 0),
            "generation_attempt_count": result.get("generation_attempt_count", 0),
            "error": result.get("error"),
        }
    )


def run_ingest_cli(argv) -> None:
    parser = argparse.ArgumentParser(description="Ingest files into the Self-RAG parent-child index.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset directory to scan. Can be repeated.")
    parser.add_argument("--file", action="append", default=[], help="Specific file to ingest. Can be repeated.")
    parser.add_argument("--include-glob", action="append", default=[], help="Only include paths matching this glob.")
    parser.add_argument("--exclude-glob", action="append", default=[], help="Exclude paths matching this glob.")
    parser.add_argument("--no-recursive", action="store_true", help="Scan dataset directories non-recursively.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and inspect chunks without embedding or writing indexes.")
    parser.add_argument("--reset", action="store_true", help="Reset SQLite, Chroma collection, and BM25 files before ingesting.")
    parser.add_argument(
        "--skip-quality-failures",
        action="store_true",
        help="Skip files with empty chunks, oversize chunks, hard-limit chunks, or mojibake warnings before embedding.",
    )
    parser.add_argument("--report-output", help="Optional path for the ingestion report.")
    parser.add_argument("--no-progress", action="store_true", help="Do not print per-file progress.")
    args = parser.parse_args(argv)

    dataset_dirs = args.dataset or ([] if args.file else ["dataset"])
    candidates = discover_ingest_candidates(
        dataset_dirs=dataset_dirs,
        files=args.file,
        include_globs=args.include_glob,
        exclude_globs=args.exclude_glob,
        recursive=not args.no_recursive,
    )
    config = SelfRagConfig()

    def progress(index, total, path, action):
        if args.no_progress:
            return
        print(f"[{index}/{total}] {action}: {path}")

    summary = run_batch_ingestion(
        candidates,
        config=config,
        dry_run=args.dry_run,
        reset=args.reset,
        skip_quality_failures=args.skip_quality_failures,
        progress=progress,
    )
    report = format_batch_report(summary)
    print()
    print(report)
    if args.report_output:
        output = Path(args.report_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote ingestion report to {output}")


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
