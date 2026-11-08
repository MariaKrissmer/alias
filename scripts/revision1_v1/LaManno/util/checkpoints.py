from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_best_checkpoint_model_source(
    *,
    model_label: str,
    configured_model_id: str,
    best_checkpoint_metadata_path: Path | str | None,
    use_configured_model_id: bool = False,
    allow_missing_best_checkpoint: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Resolve a model path from best_checkpoint.json metadata when available."""
    if use_configured_model_id:
        return configured_model_id, {"source": "configured_model_id"}

    if best_checkpoint_metadata_path is None:
        return configured_model_id, {"source": "configured_model_id", "reason": "no_checkpoint_model_id"}

    metadata_path = Path(best_checkpoint_metadata_path)
    if not metadata_path.exists():
        if allow_missing_best_checkpoint:
            return (
                configured_model_id,
                {
                    "source": "configured_model_id",
                    "reason": "missing_best_checkpoint_metadata",
                    "best_checkpoint_metadata_path": str(metadata_path),
                },
            )
        raise FileNotFoundError(
            f"Missing best checkpoint metadata for {model_label}: {metadata_path}. "
            "Run checkpoint evaluation first, or pass "
            "--allow-missing-best-checkpoint/--use-configured-model-ids."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        best_metadata = json.load(handle)
    checkpoint_path = best_metadata.get("best_checkpoint_path")
    if not checkpoint_path:
        raise ValueError(f"Missing best_checkpoint_path in {metadata_path}.")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Best checkpoint path for {model_label} does not exist: {checkpoint_path}"
        )

    return (
        str(checkpoint_path),
        {
            "source": "checkpoint_validation_score",
            "best_checkpoint_metadata_path": str(metadata_path),
            "best_checkpoint": best_metadata.get("best_checkpoint"),
            "best_checkpoint_path": str(checkpoint_path),
            "joint_validation_score": best_metadata.get("joint_validation_score"),
            "joint_triplet_accuracy": best_metadata.get("joint_triplet_accuracy"),
            "combined_pairwise_ranking_loss": best_metadata.get("combined_pairwise_ranking_loss"),
        },
    )
