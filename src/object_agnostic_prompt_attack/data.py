"""Leakage-safe MVTec AD and VisA prompt-training datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PromptTrainingSample:
    protocol_id: str
    dataset: str
    category: str
    defect_type: str
    image_path: Path
    mask_path: Path | None
    label: int
    partition: str = "attack_train"

    def manifest_record(self, root: Path) -> dict[str, object]:
        def relative(path: Path | None) -> str:
            if path is None:
                return ""
            try:
                return path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                return str(path.resolve())

        return {
            "protocol_id": self.protocol_id,
            "dataset": self.dataset,
            "category": self.category,
            "defect_type": self.defect_type,
            "label": self.label,
            "partition": self.partition,
            "image_relative_path": relative(self.image_path),
            "mask_relative_path": relative(self.mask_path),
        }


def _image_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_mvtec(root: str | Path) -> list[PromptTrainingSample]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root not found: {root_path}")
    samples: list[PromptTrainingSample] = []
    categories = sorted(path for path in root_path.iterdir() if (path / "test").is_dir())
    for category_path in categories:
        test_root = category_path / "test"
        for defect_path in sorted(path for path in test_root.iterdir() if path.is_dir()):
            normal = defect_path.name.lower() == "good"
            for image_path in _image_files(defect_path):
                mask_path = None
                if not normal:
                    mask_root = category_path / "ground_truth" / defect_path.name
                    candidates = sorted(
                        path
                        for path in mask_root.glob(f"{image_path.stem}_mask.*")
                        if path.suffix.lower() in IMAGE_EXTENSIONS
                    )
                    if not candidates:
                        raise FileNotFoundError(f"MVTec mask missing for {image_path}")
                    mask_path = candidates[0].resolve()
                samples.append(
                    PromptTrainingSample(
                        protocol_id=f"test/{category_path.name}/{defect_path.name}/{image_path.stem}",
                        dataset="mvtec",
                        category=category_path.name,
                        defect_type=defect_path.name,
                        image_path=image_path.resolve(),
                        mask_path=mask_path,
                        label=0 if normal else 1,
                    )
                )
    if not samples:
        raise RuntimeError(f"No MVTec test samples found under {root_path}")
    return samples


def _visa_manifest(root: Path) -> Path:
    for candidate in (
        root / "split_csv" / "1cls.csv",
        root / "split_csv" / "1cls.csv.csv",
        root / "1cls.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"VisA split_csv/1cls.csv not found under {root}")


def _rooted_path(root: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    value_path = Path(text.replace("\\", "/"))
    path = value_path if value_path.is_absolute() else root / value_path
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes its configured root: {text}") from exc
    return resolved


def discover_visa(root: str | Path) -> list[PromptTrainingSample]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"VisA root not found: {root_path}")
    manifest_path = _visa_manifest(root_path)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [
            {str(key).strip().lower(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    required = {"object", "split", "label", "image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"VisA manifest must contain {sorted(required)}: {manifest_path}")
    samples: list[PromptTrainingSample] = []
    for row in rows:
        if row["split"].lower() != "test":
            continue
        label_name = row["label"].lower()
        if label_name not in {"normal", "good", "0", "anomaly", "bad", "1"}:
            raise ValueError(f"Unknown VisA label {row['label']!r}")
        normal = label_name in {"normal", "good", "0"}
        image_path = _rooted_path(root_path, row["image"])
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"VisA image missing: {image_path}")
        mask_path = None if normal else _rooted_path(root_path, row.get("mask", ""))
        if not normal and (mask_path is None or not mask_path.is_file()):
            raise FileNotFoundError(f"VisA mask missing: {mask_path}")
        category = row["object"]
        samples.append(
            PromptTrainingSample(
                protocol_id=f"test/visa/{category}/{'normal' if normal else 'anomaly'}/{image_path.stem}",
                dataset="visa",
                category=category,
                defect_type="normal" if normal else "anomaly",
                image_path=image_path,
                mask_path=mask_path,
                label=0 if normal else 1,
            )
        )
    samples.sort(key=lambda sample: sample.protocol_id)
    if not samples:
        raise RuntimeError(f"No VisA test samples found under {root_path}")
    return samples


def discover_dataset(dataset: str, root: str | Path) -> list[PromptTrainingSample]:
    if dataset == "mvtec":
        return discover_mvtec(root)
    if dataset == "visa":
        return discover_visa(root)
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def automatic_attack_train_split(
    samples: Sequence[PromptTrainingSample],
    *,
    seed: int,
    evaluation_fraction: float,
) -> list[PromptTrainingSample]:
    """Match the previous category/label-balanced deterministic protocol."""

    groups: dict[tuple[str, str, int], list[PromptTrainingSample]] = {}
    for sample in samples:
        groups.setdefault((sample.dataset, sample.category, sample.label), []).append(sample)
    category_keys = sorted({(dataset, category) for dataset, category, _ in groups})
    selected: list[PromptTrainingSample] = []
    for dataset, category in category_keys:
        normal = groups.get((dataset, category, 0), [])
        abnormal = groups.get((dataset, category, 1), [])
        if not normal or not abnormal:
            raise RuntimeError(f"Need both labels in {dataset}/{category}")
        balanced_size = min(len(normal), len(abnormal))
        if balanced_size < 2:
            raise RuntimeError(f"Need at least two samples per label in {dataset}/{category}")
        for label, group in ((0, normal), (1, abnormal)):
            shuffled = sorted(group, key=lambda sample: sample.protocol_id)
            random.Random(_stable_seed(seed, dataset, category, label)).shuffle(shuffled)
            balanced = shuffled[:balanced_size]
            n_evaluation = min(
                max(int(round(balanced_size * evaluation_fraction)), 1),
                balanced_size - 1,
            )
            selected.extend(balanced[n_evaluation:])
    selected.sort(key=lambda sample: sample.protocol_id)
    if not selected:
        raise RuntimeError("Automatic prompt-training split is empty")
    return selected


def load_attack_train_manifest(
    path: str | Path,
    *,
    dataset: str,
    root: str | Path,
    discovered: Sequence[PromptTrainingSample],
) -> list[PromptTrainingSample]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [
            {str(key).strip(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows or "protocol_id" not in rows[0]:
        raise ValueError(f"Training manifest has no protocol_id rows: {manifest_path}")
    partitions = {row.get("partition", "attack_train") for row in rows}
    if "attack_train" not in partitions:
        raise ValueError(f"Manifest contains no attack_train partition: {manifest_path}")
    rows = [row for row in rows if row.get("partition", "attack_train") == "attack_train"]
    # The earlier pipeline may store MVTec and VisA rows in one shared protocol
    # CSV. Select the requested source dataset without treating the other
    # dataset's rows as an error.
    rows = [
        row
        for row in rows
        if row.get("dataset", dataset).lower() == dataset
    ]
    if not rows:
        raise ValueError(
            f"Manifest contains no attack_train rows for dataset {dataset!r}: "
            f"{manifest_path}"
        )
    by_id = {sample.protocol_id: sample for sample in discovered}
    if len(by_id) != len(discovered):
        raise RuntimeError("Discovered dataset contains duplicate protocol IDs")
    selected: list[PromptTrainingSample] = []
    seen: set[str] = set()
    for row in rows:
        protocol_id = row["protocol_id"]
        if protocol_id in seen:
            raise ValueError(f"Duplicate protocol_id in training manifest: {protocol_id}")
        seen.add(protocol_id)
        try:
            sample = by_id[protocol_id]
        except KeyError as exc:
            raise ValueError(f"Manifest sample not found under configured root: {protocol_id}") from exc
        if row.get("label", "") and int(row["label"]) != sample.label:
            raise ValueError(f"Manifest label disagrees with dataset for {protocol_id}")
        if row.get("category", "") and row["category"] != sample.category:
            raise ValueError(f"Manifest category disagrees with dataset for {protocol_id}")
        selected.append(sample)
    labels = {sample.label for sample in selected}
    if labels != {0, 1}:
        raise ValueError("Prompt training requires both normal and abnormal manifest rows")
    return sorted(selected, key=lambda sample: sample.protocol_id)


def canonical_manifest(
    samples: Sequence[PromptTrainingSample], root: str | Path
) -> tuple[list[dict[str, object]], str]:
    root_path = Path(root).expanduser().resolve()
    records = [sample.manifest_record(root_path) for sample in samples]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return records, hashlib.sha256(encoded).hexdigest()


class PromptTrainingDataset(Dataset):
    def __init__(self, samples: Sequence[PromptTrainingSample], image_size: int) -> None:
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_array = np.array(Image.open(sample.image_path).convert("RGB"), dtype=np.float32, copy=True)
        image = torch.from_numpy(image_array / 255.0).permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0].clamp_(0, 1)
        if sample.mask_path is None:
            mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        else:
            mask_array = np.array(Image.open(sample.mask_path).convert("L"), dtype=np.uint8, copy=True)
            mask = torch.from_numpy((mask_array > 0).astype(np.float32))[None, None]
            mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="nearest")[0]
        return {
            "image": image,
            "mask": mask,
            "label": sample.label,
            "protocol_id": sample.protocol_id,
        }
