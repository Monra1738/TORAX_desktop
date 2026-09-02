"""Build and manage compact TORAX cache products."""

from __future__ import annotations

import argparse
import json

from jaxa_torax.infrastructure import science as backend


def progress(completed: int, total: int, key: str) -> None:
    if completed == total or completed == 1 or completed % 25 == 0:
        print(f"[{completed:,}/{total:,}] {key}", flush=True)


def selected_records(args, missing_headers_only: bool = False):
    return backend.server_catalog_records(
        mission=getattr(args, "mission", ""),
        search_text=getattr(args, "search", ""),
        limit=getattr(args, "limit", None),
        missing_headers_only=missing_headers_only,
    )


def show_status(_args) -> dict:
    migration = backend.register_existing_cache()
    migration.pop("registered", None)
    return migration


def build_headers(args) -> dict:
    records = selected_records(args, missing_headers_only=not args.refresh)
    return backend.sync_server_headers(
        records,
        workers=args.workers,
        progress=progress,
    )


def build_previews(args) -> dict:
    records = selected_records(args)
    return backend.build_preview_batch(
        records,
        max_rows=args.rows,
        progress=progress,
    )


def clear_products(args) -> dict:
    kinds = {
        "raw": ("raw",),
        "derived": ("preview", "image"),
        "all": ("raw", "preview", "image"),
    }[args.kind]
    return backend.clear_cache(kinds=kinds)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mission", choices=backend.TORAX_MISSIONS)
    parser.add_argument(
        "--search",
        default="",
        help="Object name, observation ID, or catalog key substring",
    )
    parser.add_argument("--limit", type=int, help="Maximum matching observations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the 5 GB compact TORAX preview and image cache"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Show catalog and cache usage")
    status.set_defaults(handler=show_status)

    headers = commands.add_parser(
        "headers", help="Synchronize small JSON headers into DuckDB"
    )
    add_selection_arguments(headers)
    headers.add_argument("--workers", type=int, default=8)
    headers.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch matching headers again instead of only missing headers",
    )
    headers.set_defaults(handler=build_headers)

    previews = commands.add_parser(
        "previews", help="Build deterministic energy-stratified previews"
    )
    add_selection_arguments(previews)
    previews.add_argument("--rows", type=int, default=backend.DEFAULT_PREVIEW_ROWS)
    previews.set_defaults(handler=build_previews)

    clear = commands.add_parser("clear", help="Remove tracked cache products")
    clear.add_argument("--kind", choices=("raw", "derived", "all"), required=True)
    clear.set_defaults(handler=clear_products)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, default=str))
    skipped = result.get("skipped", []) if isinstance(result, dict) else []
    return 1 if skipped and not result.get("completed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
