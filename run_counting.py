#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import json
import traceback
from typing import Any, Dict, List

from PIL import Image
import torch

from modules.rex_omni import RexOmniVisualize, RexOmniWrapper


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = "IDEA-Research/Rex-Omni"
INPUT_DIR = r"test/counting"
OUTPUT_DIR = r"output_counting"
CATEGORIES = ["flower"]
#CATEGORIES = ["flower", "tree"]


RECURSIVE_SEARCH = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# FONT_SIZE = 13
# DRAW_WIDTH = 5
SHOW_LABELS = True

MAX_TOKENS = 2048
REPETITION_PENALTY = 1.05

ATTN_IMPLEMENTATION = "sdpa"

TORCH_DTYPE = torch.float16

SAVE_RAW_JSON = True


# =============================================================================
# Utility functions
# =============================================================================

def find_image_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    return sorted(files)

def get_dynamic_vis_style(image: Image.Image):
    width, height = image.size
    short_side = min(width, height)

    font_size = max(8, min(28, int(short_side * 0.012)))
    draw_width = max(4, min(18, int(short_side * 0.008)))

    return font_size, draw_width

def ensure_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [ensure_json_serializable(v) for v in obj]

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    return str(obj)


def count_points_for_one_category(category_prediction: Any) -> int:
    if category_prediction is None:
        return 0

    # Case 1: directly a list of points
    if isinstance(category_prediction, list):
        return len(category_prediction)

    # Case 2: dictionary structure with a known field
    if isinstance(category_prediction, dict):
        for key in ["points", "point", "coordinates", "coords", "instances"]:
            value = category_prediction.get(key)
            if isinstance(value, list):
                return len(value)
        if len(category_prediction) > 0:
            all_values_are_structured = all(
                isinstance(v, (dict, list, tuple)) for v in category_prediction.values()
            )
            if all_values_are_structured:
                return len(category_prediction)

    # Case 3: generic iterable with a length
    try:
        return len(category_prediction)
    except Exception:
        return 0


def normalize_counts(predictions: Any, categories: List[str]) -> Dict[str, int]:
    counts = {cat: 0 for cat in categories}

    if predictions is None:
        return counts

    # -------------------------------------------------------------------------
    # Case A: predictions is a dict, possibly keyed by category name
    # -------------------------------------------------------------------------
    if isinstance(predictions, dict):
        for cat in categories:
            if cat in predictions:
                counts[cat] = count_points_for_one_category(predictions[cat])

        lower_key_map = {str(k).strip().lower(): k for k in predictions.keys()}
        for cat in categories:
            if counts[cat] == 0:
                matched_key = lower_key_map.get(cat.strip().lower())
                if matched_key is not None:
                    counts[cat] = count_points_for_one_category(predictions[matched_key])

        return counts

    # -------------------------------------------------------------------------
    # Case B: predictions is a list of dicts
    # -------------------------------------------------------------------------
    if isinstance(predictions, list):
        aggregated = {cat: 0 for cat in categories}

        for item in predictions:
            if not isinstance(item, dict):
                continue

            item_category = item.get("category") or item.get("label") or item.get("name")
            if item_category is None:
                continue

            item_category_lower = str(item_category).strip().lower()

            for cat in categories:
                if item_category_lower == cat.strip().lower():
                    aggregated[cat] += count_points_for_one_category(item)

        return aggregated

    return counts


def build_visualization(
    image: Image.Image,
    predictions: Any,
) -> Image.Image:

    font_size, draw_width = get_dynamic_vis_style(image)

    vis_image = RexOmniVisualize(
        image=image,
        predictions=predictions,
        font_size=font_size,
        draw_width=draw_width,
        show_labels=SHOW_LABELS,
    )
    return vis_image


def write_long_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["filename", "relative_path", "category", "point_count", "status", "error_message"]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_wide_csv(csv_path: Path, rows: List[Dict[str, Any]], categories: List[str]) -> None:
    fieldnames = ["filename", "relative_path", "status", "error_message"] + categories

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary_path: Path, summary: Dict[str, Any]) -> None:
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(summary), f, indent=2, ensure_ascii=False)


# =============================================================================
# Main processing
# =============================================================================

def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    vis_dir = output_dir / "visualizations"
    raw_dir = output_dir / "raw_predictions"

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_RAW_JSON:
        raw_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = find_image_files(input_dir, recursive=RECURSIVE_SEARCH)

    if not image_paths:
        print(f"⚠️ No images found in: {input_dir}")
        return

    print("Initializing Rex Omni model...")
    rex_model = RexOmniWrapper(
        model_path=MODEL_PATH,
        backend="transformers",
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=TORCH_DTYPE,
        max_tokens=MAX_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        temperature=0.0,
    )

    long_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []
    failed_images: List[str] = []
    category_totals = {cat: 0 for cat in CATEGORIES}

    print(f"Found {len(image_paths)} image(s) in: {input_dir}")

    for idx, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(input_dir).as_posix()
        print(f"\n[{idx}/{len(image_paths)}] Processing: {relative_path}")

        base_name = image_path.stem
        vis_output_path = vis_dir / f"{base_name}_Pointing.jpg"
        raw_output_path = raw_dir / f"{base_name}_raw.json"

        try:
            image = Image.open(image_path).convert("RGB")
            print(f"   Image size: {image.size}")
            results = rex_model.inference(
                images=image,
                task="pointing",
                categories=CATEGORIES,
            )

            result = results[0]

            if not result.get("success", False):
                error_message = str(result.get("error", "Unknown inference error"))
                print(f" ❌ Inference failed: {error_message}")

                failed_images.append(relative_path)

                for cat in CATEGORIES:
                    long_rows.append(
                        {
                            "filename": image_path.name,
                            "relative_path": relative_path,
                            "category": cat,
                            "point_count": -1,
                            "status": "failed",
                            "error_message": error_message,
                        }
                    )

                failed_row = {
                    "filename": image_path.name,
                    "relative_path": relative_path,
                    "status": "failed",
                    "error_message": error_message,
                }
                for cat in CATEGORIES:
                    failed_row[cat] = -1
                wide_rows.append(failed_row)

                continue

            predictions = result.get("extracted_predictions")
            counts = normalize_counts(predictions, CATEGORIES)

            vis_image = build_visualization(image=image, predictions=predictions)
            vis_image.save(vis_output_path)
            print(f" ✅ Visualization saved: {vis_output_path}")

            if SAVE_RAW_JSON:
                raw_payload = {
                    "image_path": str(image_path),
                    "relative_path": relative_path,
                    "categories": CATEGORIES,
                    "result": ensure_json_serializable(result),
                    "predictions": ensure_json_serializable(predictions),
                    "counts": counts,
                }
                with open(raw_output_path, "w", encoding="utf-8") as f:
                    json.dump(raw_payload, f, indent=2, ensure_ascii=False)
                print(f" ✅ Raw prediction saved: {raw_output_path}")

            for cat in CATEGORIES:
                count_value = counts.get(cat, 0)
                category_totals[cat] += count_value

                long_rows.append(
                    {
                        "filename": image_path.name,
                        "relative_path": relative_path,
                        "category": cat,
                        "point_count": count_value,
                        "status": "success",
                        "error_message": "",
                    }
                )

            wide_row = {
                "filename": image_path.name,
                "relative_path": relative_path,
                "status": "success",
                "error_message": "",
            }
            for cat in CATEGORIES:
                wide_row[cat] = counts.get(cat, 0)
            wide_rows.append(wide_row)

            for cat in CATEGORIES:
                print(f"   - {cat}: {counts.get(cat, 0)}")

        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            print(f" ❌ Error processing {relative_path}: {error_message}")
            traceback.print_exc()

            failed_images.append(relative_path)

            for cat in CATEGORIES:
                long_rows.append(
                    {
                        "filename": image_path.name,
                        "relative_path": relative_path,
                        "category": cat,
                        "point_count": -1,
                        "status": "failed",
                        "error_message": error_message,
                    }
                )

            failed_row = {
                "filename": image_path.name,
                "relative_path": relative_path,
                "status": "failed",
                "error_message": error_message,
            }
            for cat in CATEGORIES:
                failed_row[cat] = -1
            wide_rows.append(failed_row)

    long_csv_path = output_dir / "pointing_counts_long.csv"
    wide_csv_path = output_dir / "pointing_counts_wide.csv"
    summary_json_path = output_dir / "summary.json"

    write_long_csv(long_csv_path, long_rows)
    write_wide_csv(wide_csv_path, wide_rows, CATEGORIES)

    summary = {
        "model_path": MODEL_PATH,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "recursive_search": RECURSIVE_SEARCH,
        "num_images_found": len(image_paths),
        "num_images_processed": len(wide_rows),
        "num_success": sum(1 for row in wide_rows if row["status"] == "success"),
        "num_failed": sum(1 for row in wide_rows if row["status"] == "failed"),
        "failed_images": failed_images,
        "categories": CATEGORIES,
        "category_totals": category_totals,
        "long_csv": str(long_csv_path),
        "wide_csv": str(wide_csv_path),
    }
    write_summary_json(summary_json_path, summary)

    print("\n" + "=" * 80)
    print("🎉 Processing complete.")
    print(f"Long CSV : {long_csv_path}")
    print(f"Wide CSV : {wide_csv_path}")
    print(f"Summary  : {summary_json_path}")
    print(f"Success  : {summary['num_success']}")
    print(f"Failed   : {summary['num_failed']}")
    print("Category totals:")
    for cat, total in category_totals.items():
        print(f"  - {cat}: {total}")
    print("=" * 80)


if __name__ == "__main__":
    main()