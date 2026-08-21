"""Command-line entry point for per-dataset shallow prompt fitting."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from object_agnostic_prompt_attack.config import load_experiment_config
from object_agnostic_prompt_attack.training import (
    run_training_pipeline,
    validate_dataset_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and package shallow object-agnostic prompts."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.example.yaml",
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated mvtec/visa selection, or 'all'.",
    )
    parser.add_argument("--anomalyclip-root")
    parser.add_argument("--clip-download-root")
    parser.add_argument("--mvtec-root")
    parser.add_argument("--visa-root")
    parser.add_argument("--mvtec-training-manifest")
    parser.add_argument("--visa-training-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate roots/manifests and print cohort summaries without loading CLIP.",
    )
    return parser


def _selection(raw: str, enabled: tuple[str, ...]) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return enabled
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("--datasets must contain unique mvtec/visa values")
    unknown = sorted(set(values) - set(enabled))
    if unknown:
        raise ValueError(f"Datasets are not enabled by the configuration: {unknown}")
    return values


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_experiment_config(args.config)
    model = replace(
        config.model,
        anomalyclip_root=args.anomalyclip_root or config.model.anomalyclip_root,
        clip_download_root=args.clip_download_root or config.model.clip_download_root,
        device=args.device or config.model.device,
    )
    data = replace(
        config.data,
        mvtec_root=args.mvtec_root or config.data.mvtec_root,
        visa_root=args.visa_root or config.data.visa_root,
        mvtec_training_manifest=(
            args.mvtec_training_manifest or config.data.mvtec_training_manifest
        ),
        visa_training_manifest=(
            args.visa_training_manifest or config.data.visa_training_manifest
        ),
    )
    artifacts = replace(
        config.artifacts,
        output_root=args.output_root or config.artifacts.output_root,
        overwrite=args.overwrite or config.artifacts.overwrite,
    )
    config = replace(config, model=model, data=data, artifacts=artifacts)
    datasets = _selection(args.datasets, config.data.datasets)
    if args.validate_only:
        summaries = [validate_dataset_inputs(config, dataset) for dataset in datasets]
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return
    if not config.model.anomalyclip_root:
        raise ValueError(
            "Provide model.anomalyclip_root in the config or --anomalyclip-root"
        )
    results = run_training_pipeline(config, datasets)
    for result in results:
        print(
            f"[{result.dataset}] saved {result.checkpoint} "
            f"({result.sample_count} samples)"
        )


if __name__ == "__main__":
    main()
