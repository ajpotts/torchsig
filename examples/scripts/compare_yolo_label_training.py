"""Train matched YOLO experiments using legacy and MR2 bandwidth labels.

Only the label geometry changes between conditions. Images, train/validation
splits, model initialization, seeds, and hyperparameters remain identical.

Expected dataset layout::

    dataset_root/
      classes.txt
      images/train/*.png
      images/val/*.png
      labels_legacy/train/*.txt
      labels_legacy/val/*.txt
      labels_mr2/train/*.txt
      labels_mr2/val/*.txt

Each label is standard YOLO detection format::

    class_index x_center y_center width height

Example::

    python examples/scripts/compare_yolo_label_training.py \
        --dataset-root yolo_bandwidth_ab \
        --model yolo11n.pt \
        --epochs 50 \
        --seeds 17 29 41
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from matplotlib import colormaps
from PIL import Image


METHODS = ("legacy", "mr2")
SPLITS = ("train", "val")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class TrainingRun:
    """Summary of one trained model and its optimization history."""

    trained_on: str
    seed: int
    run_dir: str
    best_weights: str
    wall_time_seconds: float
    completed_epochs: int
    best_train_box_loss: float | None
    final_train_box_loss: float | None
    best_val_box_loss: float | None
    final_val_box_loss: float | None
    best_fitness: float | None
    final_fitness: float | None
    epoch_to_95_percent_best_fitness: int | None


@dataclass(frozen=True)
class EvaluationRun:
    """Detection metrics for one model/ground-truth pairing."""

    trained_on: str
    evaluated_on: str
    seed: int
    map50_95: float
    map50: float
    map75: float
    precision: float | None
    recall: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate the paired dataset before training it.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("yolo_label_ab_runs"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training images to use; useful for a quick smoke test.",
    )
    return parser.parse_args()


def save_spectrogram_image(spectrum: np.ndarray, output_path: Path) -> None:
    """Save a clean top-origin RGB spectrogram with no annotations or margins."""
    lower, upper = np.percentile(spectrum, [5, 99.5])
    scaled = np.clip((spectrum - lower) / max(upper - lower, np.finfo(float).eps), 0, 1)
    # spectrum rows run from negative to positive frequency; image rows run top
    # to bottom, so flip the array to put positive frequency at YOLO y=0.
    rgb = (colormaps["viridis"](np.flipud(scaled))[..., :3] * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def generate_paired_dataset(
    dataset_root: Path,
    *,
    config_path: Path | None,
    train_samples: int,
    val_samples: int,
) -> None:
    """Generate shared images and legacy/MR2 label trees deterministically."""
    if train_samples <= 0 or val_samples <= 0:
        raise ValueError("train-samples and val-samples must be positive")
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty generated dataset {dataset_root}"
        )

    # Reuse the already validated component metadata and spectrogram geometry
    # from the companion MR2 visualization script in this directory.
    import visualize_mr2_yolo_labels as visualization

    if config_path is None:
        config_path = visualization.DEFAULT_CONFIG
    dataset, seed = visualization.build_dataset(config_path)
    dataset_root.mkdir(parents=True, exist_ok=True)
    total_samples = train_samples + val_samples
    dataset_class_names = visualization.optional_metadata_value(dataset, "class_names")
    class_names = (
        None
        if dataset_class_names is None
        else [str(value) for value in dataset_class_names]
    )
    indexed_names: dict[int, str] = {}

    print(f"generating {total_samples} paired samples with seed {seed}")
    for sample_index in range(total_samples):
        sample = next(dataset)
        split = "train" if sample_index < train_samples else "val"
        split_index = sample_index if split == "train" else sample_index - train_samples
        stem = f"sample_{split_index:06d}"
        iq_data = visualization.sample_iq(sample)
        spectrum, _ = visualization.spectrogram_db(
            iq_data,
            fft_size=int(dataset.fft_size),
            fft_stride=int(dataset.fft_stride),
        )
        save_spectrogram_image(
            spectrum,
            dataset_root / "images" / split / f"{stem}.png",
        )
        boxes = [
            visualization.component_box(component)
            for component in sample.component_signals
        ]
        indexed_names.update({box.class_index: box.class_name for box in boxes})
        visualization.write_yolo_labels(
            dataset_root / "labels_legacy" / split / f"{stem}.txt",
            boxes,
            use_estimate=True,
            num_samples=iq_data.size,
            sample_rate=float(dataset.sample_rate),
        )
        visualization.write_yolo_labels(
            dataset_root / "labels_mr2" / split / f"{stem}.txt",
            boxes,
            use_estimate=False,
            num_samples=iq_data.size,
            sample_rate=float(dataset.sample_rate),
        )

        if class_names is None:
            values = visualization.optional_metadata_value(sample, "class_names")
            if values is not None:
                class_names = [str(value) for value in values]

    if class_names is None:
        maximum_index = max(indexed_names)
        if set(indexed_names) != set(range(maximum_index + 1)):
            raise RuntimeError(
                "Could not recover the complete ordered class-name list from metadata"
            )
        class_names = [indexed_names[index] for index in range(maximum_index + 1)]

    (dataset_root / "classes.txt").write_text("\n".join(class_names) + "\n")
    print(f"generated paired dataset: {dataset_root}")


def image_stems(directory: Path) -> set[str]:
    """Return stems of supported images directly beneath a split directory."""
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_label_file(path: Path) -> None:
    """Validate standard normalized YOLO detection rows."""
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id < 0:
            raise ValueError(f"{path}:{line_number}: negative class index")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{path}:{line_number}: non-finite coordinate")
        x_center, y_center, width, height = coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            raise ValueError(f"{path}:{line_number}: center lies outside [0, 1]")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid width or height")


def validate_dataset(root: Path) -> list[str]:
    """Ensure both conditions use exactly the same image population."""
    classes_path = root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Missing {classes_path}")
    class_names = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if not class_names:
        raise ValueError(f"No class names found in {classes_path}")

    for split in SPLITS:
        images_dir = root / "images" / split
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing {images_dir}")
        images = image_stems(images_dir)
        if not images:
            raise ValueError(f"No images found in {images_dir}")
        for method in METHODS:
            labels_dir = root / f"labels_{method}" / split
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"Missing {labels_dir}")
            labels = {path.stem for path in labels_dir.glob("*.txt")}
            missing = images - labels
            extra = labels - images
            if missing or extra:
                raise ValueError(
                    f"{method}/{split} image-label mismatch: "
                    f"{len(missing)} missing, {len(extra)} extra"
                )
            for label_path in labels_dir.glob("*.txt"):
                validate_label_file(label_path)
    return class_names


def ensure_hardlink(link: Path, target: Path) -> None:
    """Create an idempotent file hard link without copying image bytes."""
    target = target.resolve()
    if link.exists():
        if link.samefile(target):
            return
        raise FileExistsError(f"Refusing to replace existing file {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.hardlink_to(target)


def prepare_condition(
    dataset_root: Path,
    output_dir: Path,
    method: str,
    class_names: list[str],
) -> Path:
    """Create the directory convention Ultralytics uses to locate labels."""
    condition_root = output_dir / "prepared" / method
    condition_root.mkdir(parents=True, exist_ok=True)
    # Do not use a directory symlink for images. Ultralytics resolves it to the
    # source dataset and then searches for a sibling ``labels`` directory there,
    # bypassing the condition-specific labels. File hard links keep paths under
    # this condition root while sharing the underlying image bytes.
    for split in SPLITS:
        for image_path in (dataset_root / "images" / split).iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                ensure_hardlink(
                    condition_root / "images" / split / image_path.name,
                    image_path,
                )
        for label_path in (dataset_root / f"labels_{method}" / split).glob("*.txt"):
            ensure_hardlink(
                condition_root / "labels" / split / label_path.name,
                label_path,
            )
    yaml_path = condition_root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(condition_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            sort_keys=False,
        )
    )
    return yaml_path


def numeric(row: dict[str, str], *candidates: str) -> float | None:
    """Read a metric despite small Ultralytics column-name variations."""
    normalized = {key.strip(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate in normalized and normalized[candidate] != "":
            return float(normalized[candidate])
    return None


def summarize_history(results_csv: Path) -> dict[str, Any]:
    """Extract comparable optimization statistics from results.csv."""
    with results_csv.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"No training history found in {results_csv}")

    train_box = [numeric(row, "train/box_loss") for row in rows]
    val_box = [numeric(row, "val/box_loss") for row in rows]
    fitness = [
        numeric(row, "fitness", "metrics/mAP50-95(B)", "metrics/mAP50-95")
        for row in rows
    ]
    finite_fitness = [value for value in fitness if value is not None and math.isfinite(value)]
    epoch_95 = None
    best_fitness = max(finite_fitness) if finite_fitness else None
    if best_fitness is not None:
        threshold = 0.95 * best_fitness
        epoch_95 = next(
            (
                index
                for index, value in enumerate(fitness, start=1)
                if value is not None and value >= threshold
            ),
            None,
        )

    def minimum(values: list[float | None]) -> float | None:
        finite = [value for value in values if value is not None and math.isfinite(value)]
        return min(finite) if finite else None

    return {
        "completed_epochs": len(rows),
        "best_train_box_loss": minimum(train_box),
        "final_train_box_loss": train_box[-1],
        "best_val_box_loss": minimum(val_box),
        "final_val_box_loss": val_box[-1],
        "best_fitness": best_fitness,
        "final_fitness": fitness[-1],
        "epoch_to_95_percent_best_fitness": epoch_95,
    }


def evaluate(
    weights: Path,
    data_yaml: Path,
    *,
    imgsz: int,
    batch: int,
    device: str | None,
    workers: int,
) -> dict[str, float | None]:
    """Evaluate saved weights and normalize the metrics returned by Ultralytics."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    return {
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "precision": float(box.mp) if hasattr(box, "mp") else None,
        "recall": float(box.mr) if hasattr(box, "mr") else None,
    }


def write_csv(path: Path, records: list[Any]) -> None:
    """Write dataclass records to CSV."""
    if not records:
        return
    rows = [asdict(record) for record in records]
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run paired training and cross-evaluation experiments."""
    args = parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0:
        raise ValueError("epochs, batch, and imgsz must be positive")
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.generate:
        generate_paired_dataset(
            dataset_root,
            config_path=args.config,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
        )
    class_names = validate_dataset(dataset_root)
    data_yamls = {
        method: prepare_condition(dataset_root, output_dir, method, class_names)
        for method in METHODS
    }

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install the project's Ultralytics dependency first") from error

    training_runs: list[TrainingRun] = []
    evaluation_runs: list[EvaluationRun] = []
    weights_by_condition: dict[tuple[int, str], Path] = {}

    for seed_index, seed in enumerate(args.seeds):
        order = METHODS if seed_index % 2 == 0 else tuple(reversed(METHODS))
        for method in order:
            run_name = f"{method}_seed_{seed}"
            started = time.perf_counter()
            model = YOLO(args.model)
            model.train(
                data=str(data_yamls[method]),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                patience=args.patience,
                seed=seed,
                deterministic=True,
                fraction=args.fraction,
                project=str(output_dir / "runs"),
                name=run_name,
                exist_ok=False,
                plots=True,
            )
            wall_time = time.perf_counter() - started
            if model.trainer is None:
                raise RuntimeError("Ultralytics did not retain its trainer state")
            run_dir = Path(model.trainer.save_dir)
            best_weights = run_dir / "weights" / "best.pt"
            if not best_weights.is_file():
                raise FileNotFoundError(f"Training did not create {best_weights}")
            history = summarize_history(run_dir / "results.csv")
            training_runs.append(
                TrainingRun(
                    trained_on=method,
                    seed=seed,
                    run_dir=str(run_dir),
                    best_weights=str(best_weights),
                    wall_time_seconds=wall_time,
                    **history,
                )
            )
            weights_by_condition[(seed, method)] = best_weights

    for seed in args.seeds:
        for trained_on in METHODS:
            weights = weights_by_condition[(seed, trained_on)]
            for evaluated_on in METHODS:
                metrics = evaluate(
                    weights,
                    data_yamls[evaluated_on],
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    workers=args.workers,
                )
                evaluation_runs.append(
                    EvaluationRun(
                        trained_on=trained_on,
                        evaluated_on=evaluated_on,
                        seed=seed,
                        **metrics,
                    )
                )

    write_csv(output_dir / "training_summary.csv", training_runs)
    write_csv(output_dir / "cross_evaluation.csv", evaluation_runs)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    print(f"training summary: {output_dir / 'training_summary.csv'}")
    print(f"cross-evaluation: {output_dir / 'cross_evaluation.csv'}")


if __name__ == "__main__":
    main()