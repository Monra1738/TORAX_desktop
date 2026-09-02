"""Download/cache Cas A data and build four side-by-side reference-band images.

Use ``--no-download`` for an offline cache/product check.  The default selection
keeps the large XRISM Xtend files optional while including both Resolve records,
ASCA GIS/SIS, and a Suzaku XIS comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "var")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-bins", type=int, default=240)
    parser.add_argument("--preview-rows", type=int, default=15_000)
    parser.add_argument("--include-xtend", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser


def _render_products(
    products: list[dict],
    output_dir: Path,
    stem: str = "cas_a_bands_side_by_side",
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(products), figsize=(16, 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    paths = []
    for axis, product in zip(axes, products):
        values = np.asarray(product["hist"], dtype=float)
        positive = values[values > 0]
        ceiling = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        display = np.sqrt(np.clip(values, 0.0, ceiling) / max(ceiling, 1.0))
        extent = (
            float(product["x_edges"][0]),
            float(product["x_edges"][-1]),
            float(product["y_edges"][0]),
            float(product["y_edges"][-1]),
        )
        axis.imshow(display, origin="lower", extent=extent, cmap="inferno", interpolation="nearest")
        axis.set_xlim(extent[1], extent[0])
        axis.set_title(
            f'{product["label"]}\n{product["event_count"]:,} events',
            fontsize=9,
        )
        axis.set_xlabel("RA (deg)")
        axis.set_ylabel("DEC (deg)")
        individual = output_dir / (
            f'{stem.removesuffix("_side_by_side")}_'
            f'{product["low_kev"]:.2f}-{product["high_kev"]:.2f}keV.png'
        )
        single, single_axis = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
        single_axis.imshow(
            display, origin="lower", extent=extent, cmap="inferno", interpolation="nearest"
        )
        single_axis.set_xlim(extent[1], extent[0])
        single_axis.set_title(f'{product["label"]}  •  {product["event_count"]:,} events')
        single_axis.set_xlabel("RA (deg)")
        single_axis.set_ylabel("DEC (deg)")
        single.savefig(individual, dpi=180)
        plt.close(single)
        paths.append(str(individual.resolve()))
    combined = output_dir / f"{stem}.png"
    figure.savefig(combined, dpi=180)
    paths.insert(0, str(combined.resolve()))
    plt.close(figure)
    return paths


def _extract_reference_images(pdf_path: Path, output_dir: Path) -> list[str]:
    """Copy the Cas A figures embedded in the Theme 2 PDF when pypdf is available."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    page = PdfReader(str(pdf_path)).pages[7]  # PDF page 8 contains the Cas A examples.
    paths = []
    for image in page.images:
        destination = output_dir / f"reference_{image.name}"
        destination.write_bytes(image.data)
        paths.append(str(destination.resolve()))
    return paths


def _render_comparison(
    generated_path: Path,
    reference_paths: list[str],
    output_dir: Path,
) -> str | None:
    if not reference_paths:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    generated = Image.open(generated_path).convert("RGB")
    references = [Image.open(path).convert("RGB") for path in reference_paths]
    width = max(generated.width, *(image.width for image in references))
    thumb_height = max(220, width * 440 // 780)
    generated = generated.resize((width, thumb_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, thumb_height * (len(references) + 1) + 42), "white")
    canvas.paste(generated, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), "TORAX generated ASCA/Suzaku/XRISM bands vs Theme 2 PDF reference", fill="black")
    for index, reference in enumerate(references, start=1):
        reference.thumbnail((width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(reference, (0, 42 + index * thumb_height))
    destination = output_dir / "cas_a_reference_comparison.png"
    canvas.save(destination)
    return str(destination.resolve())


def run(args: argparse.Namespace) -> dict:
    data_root = args.data_root.expanduser().resolve()
    os.environ["TORAX_DATA_ROOT"] = str(data_root)
    output_dir = (args.output_dir or data_root / "exports" / "cas_a").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports follow TORAX_DATA_ROOT so all paths use the normal application cache.
    from jaxa_torax.infrastructure.cas_a import (
        CAS_A_BANDS,
        CAS_A_REGION,
        cas_a_manifest,
        cas_a_record_specs,
        cas_a_records,
    )
    from jaxa_torax.infrastructure.event_sources import ensure_cached
    from jaxa_torax.infrastructure.images import exact_all_events_image, exact_energy_image
    from jaxa_torax.infrastructure.previews import read_compact_preview
    from jaxa_torax.infrastructure.science_core import DB_PATH

    specs = cas_a_record_specs(args.include_xtend)
    records = cas_a_records(include_xtend=args.include_xtend)
    cached = []
    for record in records:
        if not args.no_download:
            ensure_cached(record, db_path=DB_PATH)
        # This also verifies the compact preview cache convention used by the UI.
        frame, _metadata, total, cache_hit = read_compact_preview(
            record, max_rows=args.preview_rows, db_path=DB_PATH
        )
        cached.append(
            {
                "key": f"{record.mission}/{record.instrument}/{record.observation_id}",
                "rows": len(frame),
                "total_events": total,
                "preview_cache_hit": bool(cache_hit),
                "raw_path": str(record.parquet_path.resolve()),
                "header_path": str(record.header_path.resolve()),
            }
        )

    products = []
    for low, high, label in CAS_A_BANDS:
        result = exact_energy_image(
            records,
            low,
            high,
            bins=max(16, args.image_bins),
            db_path=DB_PATH,
            region=CAS_A_REGION,
        )
        products.append({**result, "label": label})
    artifact_paths = _render_products(products, output_dir)
    all_events = exact_all_events_image(
        records,
        bins=max(16, args.image_bins),
        db_path=DB_PATH,
        region=CAS_A_REGION,
    )
    all_event_artifacts = _render_products(
        [{**all_events, "label": "All energies — every parquet event"}],
        output_dir,
        stem="cas_a_all_events_exact_side_by_side",
    )
    artifact_paths.extend(all_event_artifacts)
    asca_records = [
        record for record in records
        if record.mission == "asca" and record.observation_id in {"50018000", "50018010"}
    ]
    asca_products = []
    for low, high, label in CAS_A_BANDS:
        result = exact_energy_image(
            asca_records,
            low,
            high,
            bins=max(16, args.image_bins),
            db_path=DB_PATH,
            region=CAS_A_REGION,
        )
        asca_products.append({**result, "label": label})
    asca_artifacts = _render_products(
        asca_products, output_dir, stem="cas_a_asca_bands_side_by_side"
    )
    reference_paths = _extract_reference_images(
        ROOT / "docs" / "8470b964-a30c-4a14-b994-56cb1beb73d6_2026_JAXA_Intern_Theme_2.pdf",
        output_dir,
    )
    manifest = cas_a_manifest(
        specs,
        image_bins=args.image_bins,
        output_dir=output_dir,
        db_path=DB_PATH,
        downloaded=not args.no_download,
    )
    manifest["cache"] = cached
    manifest["artifacts"] = artifact_paths
    manifest["reference_assets"] = reference_paths
    manifest["products"] = [
        {
            "label": item["label"],
            "low_kev": item["low_kev"],
            "high_kev": item["high_kev"],
            "event_count": int(item["event_count"]),
            "cache_hit": bool(item.get("cache_hit")),
        }
        for item in products
    ]
    manifest["all_events_product"] = {
        "low_kev": float(all_events["low_kev"]),
        "high_kev": float(all_events["high_kev"]),
        "event_count": int(all_events["event_count"]),
        "histogram_sum": int(all_events["hist"].sum()),
        "cache_hit": bool(all_events.get("cache_hit")),
        "uses_preview_sampling": False,
    }
    manifest["product_cache_hits"] = [bool(item.get("cache_hit")) for item in products]
    manifest["artifacts"].extend(asca_artifacts)
    comparison = _render_comparison(
        Path(asca_artifacts[0]), reference_paths, output_dir
    )
    if comparison:
        manifest["artifacts"].append(comparison)
    manifest["comparison_notes"] = [
        (
            "Images use 240x240 sky bins, square-root stretch, inferno palette, "
            "99.5th-percentile clipping, lower-origin pixels, and RA reversed to increase left."
        ),
        (
            "The ASCA-only row now uses both original 1993 observations 50018000 and "
            "50018010 identified in the official HEASARC ASCA master catalog."
        ),
        (
            "The literature panels are PSF-restored to about 30 arcsec FWHM. TORAX currently "
            "shows exact raw sky histograms, so detector footprint, background, exposure, PSF, "
            "and continuum-subtraction differences remain and exact pixel equality is not claimed."
        ),
    ]
    manifest["reference_reproduction"] = {
        "same_observation_ids": True,
        "same_energy_bands": True,
        "same_event_accounting": "all matching parquet events; no preview sampling",
        "same_restoration_pipeline": False,
        "verdict": (
            "Scientifically comparable raw-band maps, not an exact reproduction of the "
            "paper's proprietary/undocumented PSF-restoration result."
        ),
    }
    (output_dir / "cas_a_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Cas A check failed: {error}")
        return 1
    print(json.dumps({
        "output_dir": manifest["output_dir"],
        "artifacts": manifest["artifacts"],
        "records": [item["key"] for item in manifest["cache"]],
        "event_counts": [item["event_count"] for item in manifest.get("products", [])],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
