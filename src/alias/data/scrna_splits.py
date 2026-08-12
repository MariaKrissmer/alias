from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class SplitIndices:
    dataset_id: str
    strategy: str
    random_state: int
    train_indices: list[str]
    test_indices: list[str]
    columns: dict[str, str | None] = field(default_factory=dict)
    heldout_values: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def write_split_indices(split: SplitIndices, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(split), handle, indent=2, sort_keys=True)
    return output_path


def load_split_indices(path: str | Path) -> SplitIndices:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitIndices(**payload)


def validate_split_indices(split: SplitIndices, obs: pd.DataFrame) -> None:
    train = set(map(str, split.train_indices))
    test = set(map(str, split.test_indices))
    obs_index = set(map(str, obs.index))

    overlap = train.intersection(test)
    if overlap:
        raise ValueError(f"Train/test index overlap detected: {sorted(overlap)[:5]}")

    missing = train.union(test).difference(obs_index)
    if missing:
        raise ValueError(f"Split indices missing from AnnData obs: {sorted(missing)[:5]}")


def validate_no_donor_leakage(
    split: SplitIndices,
    obs: pd.DataFrame,
    *,
    donor_column: str,
) -> None:
    validate_split_indices(split, obs)
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    train_donors = set(obs_indexed.loc[split.train_indices, donor_column].dropna().astype(str))
    test_donors = set(obs_indexed.loc[split.test_indices, donor_column].dropna().astype(str))
    leaked = train_donors.intersection(test_donors)
    if leaked:
        raise ValueError(f"Donor leakage detected between train and test: {sorted(leaked)}")


def validate_heldout_absent_from_train(
    split: SplitIndices,
    obs: pd.DataFrame,
    *,
    column: str,
    values: list[str],
) -> None:
    validate_split_indices(split, obs)
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    train_values = set(obs_indexed.loc[split.train_indices, column].dropna().astype(str))
    forbidden = train_values.intersection(set(map(str, values)))
    if forbidden:
        raise ValueError(f"Held-out values found in train split: {sorted(forbidden)}")


def validate_no_group_leakage(
    split: SplitIndices,
    obs: pd.DataFrame,
    *,
    group_column: str,
) -> None:
    validate_split_indices(split, obs)
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    train_groups = set(obs_indexed.loc[split.train_indices, group_column].dropna().astype(str))
    test_groups = set(obs_indexed.loc[split.test_indices, group_column].dropna().astype(str))
    leaked = train_groups.intersection(test_groups)
    if leaked:
        raise ValueError(f"Group leakage detected between train and test: {sorted(leaked)}")


def make_random_stratified_split(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    annotation_column: str,
    test_size: float,
    total_cells: int | None = None,
    random_state: int,
) -> SplitIndices:
    rng = np.random.default_rng(random_state)
    train_candidates: list[str] = []
    test_candidates: list[str] = []

    grouped = obs.groupby(annotation_column, sort=True, dropna=False, observed=False)
    for _, group in grouped:
        indices = group.index.astype(str).to_numpy()
        shuffled = rng.permutation(indices)
        n_test = int(round(len(shuffled) * test_size))
        if len(shuffled) > 1:
            n_test = max(1, min(len(shuffled) - 1, n_test))
        test_candidates.extend(shuffled[:n_test].tolist())
        train_candidates.extend(shuffled[n_test:].tolist())

    total_cells = _validate_total_cells(total_cells)
    if total_cells is not None:
        obs_indexed = obs.copy()
        obs_indexed.index = obs_indexed.index.astype(str)
        full_proportions = (
            obs_indexed[annotation_column]
            .astype(str)
            .value_counts(normalize=True, dropna=False)
            .sort_index()
        )
        train_target, test_target, selected_total = _allocate_train_test_targets(
            total_cells=total_cells,
            test_size=test_size,
            n_train_candidates=len(train_candidates),
            n_test_candidates=len(test_candidates),
        )
        train_indices = _stratified_subsample_indices(
            obs_indexed,
            train_candidates,
            annotation_column=annotation_column,
            target_count=train_target,
            full_proportions=full_proportions,
            random_state=random_state,
        )
        test_indices = _stratified_subsample_indices(
            obs_indexed,
            test_candidates,
            annotation_column=annotation_column,
            target_count=test_target,
            full_proportions=full_proportions,
            random_state=random_state + 1,
        )
        metadata = {
            "test_size": test_size,
            "total_cells": total_cells,
            "n_target_train_cells": train_target,
            "n_target_test_cells": test_target,
            "n_selected_total_cells": selected_total,
            "n_train_candidate_cells": len(train_candidates),
            "n_test_candidate_cells": len(test_candidates),
        }
    else:
        train_indices = train_candidates
        test_indices = test_candidates
        metadata = {"test_size": test_size}

    return SplitIndices(
        dataset_id=dataset_id,
        strategy="random_stratified",
        random_state=random_state,
        train_indices=sorted(train_indices),
        test_indices=sorted(test_indices),
        columns={"annotation_column": annotation_column},
        metadata=metadata,
    )


def make_proportional_heldout_group_split(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    group_column: str,
    group_key: str,
    annotation_column: str,
    test_size: float,
    group_test_size: float | None,
    random_state: int,
    total_cells: int | None = None,
    stratified_subsample: bool = True,
) -> SplitIndices:
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    if group_column not in obs_indexed.columns:
        raise ValueError(f"Missing group_column in AnnData obs: {group_column}")
    if annotation_column not in obs_indexed.columns:
        raise ValueError(f"Missing annotation_column in AnnData obs: {annotation_column}")

    effective_group_test_size = group_test_size if group_test_size is not None else test_size
    groups = np.array(sorted(obs_indexed[group_column].dropna().astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("heldout_group split requires at least two groups.")

    rng = np.random.default_rng(random_state)
    shuffled_groups = rng.permutation(groups)
    n_test_groups = _split_count(
        len(shuffled_groups),
        effective_group_test_size,
        require_both_sides=True,
    )
    test_groups = sorted(shuffled_groups[:n_test_groups].astype(str).tolist())
    group_values = obs_indexed[group_column].astype(str)
    train_candidates = obs_indexed.index[~group_values.isin(test_groups)].astype(str).tolist()
    test_candidates = obs_indexed.index[group_values.isin(test_groups)].astype(str).tolist()
    total_cells = _validate_total_cells(total_cells)

    if total_cells is not None:
        train_target, test_target, selected_total = _allocate_train_test_targets(
            total_cells=total_cells,
            test_size=test_size,
            n_train_candidates=len(train_candidates),
            n_test_candidates=len(test_candidates),
        )
        if stratified_subsample:
            full_proportions = (
                obs_indexed[annotation_column]
                .astype(str)
                .value_counts(normalize=True, dropna=False)
                .sort_index()
            )
            train_indices = _stratified_subsample_indices(
                obs_indexed,
                train_candidates,
                annotation_column=annotation_column,
                target_count=train_target,
                full_proportions=full_proportions,
                random_state=random_state,
            )
            test_indices = _stratified_subsample_indices(
                obs_indexed,
                test_candidates,
                annotation_column=annotation_column,
                target_count=test_target,
                full_proportions=full_proportions,
                random_state=random_state + 1,
            )
        else:
            train_indices = _random_subsample_indices(
                train_candidates,
                target_count=train_target,
                random_state=random_state,
            )
            test_indices = _random_subsample_indices(
                test_candidates,
                target_count=test_target,
                random_state=random_state + 1,
            )
    elif stratified_subsample:
        full_proportions = (
            obs_indexed[annotation_column]
            .astype(str)
            .value_counts(normalize=True, dropna=False)
            .sort_index()
        )
        train_indices = _stratified_subsample_indices(
            obs_indexed,
            train_candidates,
            annotation_column=annotation_column,
            target_fraction=test_size,
            full_proportions=full_proportions,
            random_state=random_state,
        )
        test_indices = _stratified_subsample_indices(
            obs_indexed,
            test_candidates,
            annotation_column=annotation_column,
            target_fraction=test_size,
            full_proportions=full_proportions,
            random_state=random_state + 1,
        )
        train_target = len(train_indices)
        test_target = len(test_indices)
        selected_total = train_target + test_target
    else:
        train_indices = sorted(train_candidates)
        test_indices = sorted(test_candidates)
        train_target = len(train_indices)
        test_target = len(test_indices)
        selected_total = len(train_indices) + len(test_indices)

    metadata = {
        "test_size": test_size,
        "group_test_size": effective_group_test_size,
        "stratified_subsample": stratified_subsample,
        "n_train_groups": len(groups) - n_test_groups,
        "n_test_groups": n_test_groups,
        "n_train_candidate_cells": len(train_candidates),
        "n_test_candidate_cells": len(test_candidates),
    }
    if total_cells is not None:
        metadata.update(
            {
                "total_cells": total_cells,
                "n_target_train_cells": train_target,
                "n_target_test_cells": test_target,
                "n_selected_total_cells": selected_total,
            }
        )

    split = SplitIndices(
        dataset_id=dataset_id,
        strategy=f"heldout_{group_column}",
        random_state=random_state,
        train_indices=sorted(train_indices),
        test_indices=sorted(test_indices),
        columns={"annotation_column": annotation_column, "group_column": group_column},
        heldout_values={group_key: test_groups},
        metadata=metadata,
    )
    validate_no_group_leakage(split, obs_indexed, group_column=group_column)
    return split


def make_heldout_value_split(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    annotation_column: str,
    heldout_column: str,
    heldout_values: list[str],
    heldout_key: str,
    random_state: int,
    total_cells: int | None = None,
    test_size: float = 0.1,
    stratified_subsample: bool = True,
) -> SplitIndices:
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    if heldout_column not in obs_indexed.columns:
        raise ValueError(f"Missing heldout_column in AnnData obs: {heldout_column}")
    if annotation_column not in obs_indexed.columns:
        raise ValueError(f"Missing annotation_column in AnnData obs: {annotation_column}")

    heldout_values = sorted(map(str, heldout_values))
    heldout_mask = obs_indexed[heldout_column].astype(str).isin(heldout_values)
    train_candidates = obs_indexed.index[~heldout_mask].astype(str).tolist()
    test_candidates = obs_indexed.index[heldout_mask].astype(str).tolist()
    total_cells = _validate_total_cells(total_cells)

    if total_cells is not None:
        train_target, test_target, selected_total = _allocate_train_test_targets(
            total_cells=total_cells,
            test_size=test_size,
            n_train_candidates=len(train_candidates),
            n_test_candidates=len(test_candidates),
        )
        if stratified_subsample:
            full_proportions = (
                obs_indexed[annotation_column]
                .astype(str)
                .value_counts(normalize=True, dropna=False)
                .sort_index()
            )
            train_indices = _stratified_subsample_indices(
                obs_indexed,
                train_candidates,
                annotation_column=annotation_column,
                target_count=train_target,
                full_proportions=full_proportions,
                random_state=random_state,
            )
            test_indices = _stratified_subsample_indices(
                obs_indexed,
                test_candidates,
                annotation_column=annotation_column,
                target_count=test_target,
                full_proportions=full_proportions,
                random_state=random_state + 1,
            )
        else:
            train_indices = _random_subsample_indices(
                train_candidates,
                target_count=train_target,
                random_state=random_state,
            )
            test_indices = _random_subsample_indices(
                test_candidates,
                target_count=test_target,
                random_state=random_state + 1,
            )
    else:
        train_indices = sorted(train_candidates)
        test_indices = sorted(test_candidates)
        train_target = len(train_indices)
        test_target = len(test_indices)
        selected_total = len(train_indices) + len(test_indices)

    split = SplitIndices(
        dataset_id=dataset_id,
        strategy=f"heldout_{heldout_column}",
        random_state=random_state,
        train_indices=sorted(train_indices),
        test_indices=sorted(test_indices),
        columns={"annotation_column": annotation_column, "heldout_column": heldout_column},
        heldout_values={heldout_key: heldout_values},
        metadata={
            "test_size": test_size,
            "stratified_subsample": stratified_subsample,
            "total_cells": total_cells,
            "n_train_candidate_cells": len(train_candidates),
            "n_test_candidate_cells": len(test_candidates),
            "n_target_train_cells": train_target,
            "n_target_test_cells": test_target,
            "n_selected_total_cells": selected_total,
        },
    )
    validate_heldout_absent_from_train(
        split,
        obs_indexed,
        column=heldout_column,
        values=heldout_values,
    )
    return split


def make_heldout_donor_split(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    donor_column: str,
    annotation_column: str,
    heldout_donors: list[str],
    random_state: int,
) -> SplitIndices:
    heldout_donors = sorted(map(str, heldout_donors))
    donor_values = obs[donor_column].astype(str)
    test_mask = donor_values.isin(heldout_donors)
    split = SplitIndices(
        dataset_id=dataset_id,
        strategy="heldout_donor",
        random_state=random_state,
        train_indices=sorted(obs.index[~test_mask].astype(str).tolist()),
        test_indices=sorted(obs.index[test_mask].astype(str).tolist()),
        columns={"annotation_column": annotation_column, "donor_column": donor_column},
        heldout_values={"donors": heldout_donors},
    )
    validate_no_donor_leakage(split, obs, donor_column=donor_column)
    return split


def _split_count(total: int, fraction: float, *, require_both_sides: bool = False) -> int:
    if not 0 < fraction < 1:
        raise ValueError(f"Split fraction must be between 0 and 1, got {fraction}.")
    count = int(round(total * fraction))
    if total > 0:
        count = max(1, min(total, count))
    if require_both_sides and total > 1:
        count = min(total - 1, count)
    return count


def _validate_total_cells(total_cells: int | None) -> int | None:
    if total_cells is None:
        return None
    total_cells = int(total_cells)
    if total_cells <= 0:
        raise ValueError(f"total_cells must be a positive integer, got {total_cells}.")
    return total_cells


def _allocate_train_test_targets(
    *,
    total_cells: int,
    test_size: float,
    n_train_candidates: int,
    n_test_candidates: int,
) -> tuple[int, int, int]:
    if not 0 < test_size < 1:
        raise ValueError(f"Split fraction must be between 0 and 1, got {test_size}.")

    available_total = n_train_candidates + n_test_candidates
    target_total = min(total_cells, available_total)
    if target_total == 0:
        return 0, 0, 0

    test_target = int(round(target_total * test_size))
    if n_train_candidates > 0 and n_test_candidates > 0 and target_total > 1:
        test_target = max(1, min(target_total - 1, test_target))
    test_target = min(test_target, n_test_candidates)
    train_target = min(target_total - test_target, n_train_candidates)

    remaining = target_total - train_target - test_target
    if remaining > 0:
        train_fill = min(remaining, n_train_candidates - train_target)
        train_target += train_fill
        remaining -= train_fill
    if remaining > 0:
        test_fill = min(remaining, n_test_candidates - test_target)
        test_target += test_fill

    return train_target, test_target, train_target + test_target


def _random_subsample_indices(
    candidate_indices: list[str],
    *,
    target_count: int,
    random_state: int,
) -> list[str]:
    candidate_indices = list(map(str, candidate_indices))
    if target_count >= len(candidate_indices):
        return sorted(candidate_indices)
    rng = np.random.default_rng(random_state)
    return sorted(rng.permutation(np.array(sorted(candidate_indices)))[:target_count].tolist())


def _stratified_subsample_indices(
    obs: pd.DataFrame,
    candidate_indices: list[str],
    *,
    annotation_column: str,
    target_fraction: float | None = None,
    target_count: int | None = None,
    full_proportions: pd.Series,
    random_state: int,
) -> list[str]:
    if not candidate_indices:
        return []

    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    candidate_indices = list(map(str, candidate_indices))
    candidate_obs = obs_indexed.loc[candidate_indices]
    if target_count is None:
        if target_fraction is None:
            raise ValueError("Either target_fraction or target_count is required.")
        target_total = _split_count(len(candidate_obs), target_fraction)
    else:
        target_total = min(int(target_count), len(candidate_obs))
    available = candidate_obs[annotation_column].astype(str).value_counts(dropna=False)

    raw_targets = full_proportions * target_total
    base_targets = np.floor(raw_targets).astype(int)
    target_counts: dict[str, int] = {}
    for label, base_count in base_targets.items():
        target_counts[str(label)] = min(int(base_count), int(available.get(str(label), 0)))

    remaining = target_total - sum(target_counts.values())
    ordered_labels = (
        (raw_targets - base_targets)
        .sort_values(ascending=False, kind="mergesort")
        .index.astype(str)
        .tolist()
    )
    while remaining > 0:
        changed = False
        for label in ordered_labels:
            if remaining == 0:
                break
            if target_counts.get(label, 0) < int(available.get(label, 0)):
                target_counts[label] = target_counts.get(label, 0) + 1
                remaining -= 1
                changed = True
        if not changed:
            break

    rng = np.random.default_rng(random_state)
    selected: list[str] = []
    labels = candidate_obs[annotation_column].astype(str)
    for label in sorted(target_counts):
        count = target_counts[label]
        if count <= 0:
            continue
        label_indices = np.array(sorted(candidate_obs.index[labels == label].astype(str)))
        selected.extend(rng.permutation(label_indices)[:count].tolist())

    if len(selected) < target_total:
        selected_set = set(selected)
        remaining_indices = np.array(
            sorted(index for index in candidate_obs.index.astype(str) if index not in selected_set)
        )
        fill_count = min(target_total - len(selected), len(remaining_indices))
        selected.extend(rng.permutation(remaining_indices)[:fill_count].tolist())

    return sorted(selected)


def make_proportional_heldout_donor_split(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    donor_column: str,
    annotation_column: str,
    test_size: float,
    donor_test_size: float | None,
    random_state: int,
    total_cells: int | None = None,
    stratified_subsample: bool = True,
) -> SplitIndices:
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    if donor_column not in obs_indexed.columns:
        raise ValueError(f"Missing donor_column in AnnData obs: {donor_column}")
    if annotation_column not in obs_indexed.columns:
        raise ValueError(f"Missing annotation_column in AnnData obs: {annotation_column}")

    effective_donor_test_size = donor_test_size if donor_test_size is not None else test_size
    donors = np.array(sorted(obs_indexed[donor_column].dropna().astype(str).unique()))
    if len(donors) < 2:
        raise ValueError("heldout_donor split requires at least two donors.")

    rng = np.random.default_rng(random_state)
    shuffled_donors = rng.permutation(donors)
    n_test_donors = _split_count(
        len(shuffled_donors),
        effective_donor_test_size,
        require_both_sides=True,
    )
    test_donors = sorted(shuffled_donors[:n_test_donors].astype(str).tolist())
    donor_values = obs_indexed[donor_column].astype(str)
    train_candidates = obs_indexed.index[~donor_values.isin(test_donors)].astype(str).tolist()
    test_candidates = obs_indexed.index[donor_values.isin(test_donors)].astype(str).tolist()
    total_cells = _validate_total_cells(total_cells)

    if total_cells is not None:
        train_target, test_target, selected_total = _allocate_train_test_targets(
            total_cells=total_cells,
            test_size=test_size,
            n_train_candidates=len(train_candidates),
            n_test_candidates=len(test_candidates),
        )
        if stratified_subsample:
            full_proportions = (
                obs_indexed[annotation_column]
                .astype(str)
                .value_counts(normalize=True, dropna=False)
                .sort_index()
            )
            train_indices = _stratified_subsample_indices(
                obs_indexed,
                train_candidates,
                annotation_column=annotation_column,
                target_count=train_target,
                full_proportions=full_proportions,
                random_state=random_state,
            )
            test_indices = _stratified_subsample_indices(
                obs_indexed,
                test_candidates,
                annotation_column=annotation_column,
                target_count=test_target,
                full_proportions=full_proportions,
                random_state=random_state + 1,
            )
        else:
            train_indices = _random_subsample_indices(
                train_candidates,
                target_count=train_target,
                random_state=random_state,
            )
            test_indices = _random_subsample_indices(
                test_candidates,
                target_count=test_target,
                random_state=random_state + 1,
            )
    elif stratified_subsample:
        full_proportions = (
            obs_indexed[annotation_column]
            .astype(str)
            .value_counts(normalize=True, dropna=False)
            .sort_index()
        )
        train_indices = _stratified_subsample_indices(
            obs_indexed,
            train_candidates,
            annotation_column=annotation_column,
            target_fraction=test_size,
            full_proportions=full_proportions,
            random_state=random_state,
        )
        test_indices = _stratified_subsample_indices(
            obs_indexed,
            test_candidates,
            annotation_column=annotation_column,
            target_fraction=test_size,
            full_proportions=full_proportions,
            random_state=random_state + 1,
        )
    else:
        train_indices = sorted(train_candidates)
        test_indices = sorted(test_candidates)
        train_target = len(train_indices)
        test_target = len(test_indices)
        selected_total = len(train_indices) + len(test_indices)

    metadata = {
        "test_size": test_size,
        "donor_test_size": effective_donor_test_size,
        "stratified_subsample": stratified_subsample,
        "n_train_donors": len(donors) - n_test_donors,
        "n_test_donors": n_test_donors,
        "n_train_candidate_cells": len(train_candidates),
        "n_test_candidate_cells": len(test_candidates),
    }
    if total_cells is not None:
        metadata.update(
            {
                "total_cells": total_cells,
                "n_target_train_cells": train_target,
                "n_target_test_cells": test_target,
                "n_selected_total_cells": selected_total,
            }
        )

    split = SplitIndices(
        dataset_id=dataset_id,
        strategy="heldout_donor",
        random_state=random_state,
        train_indices=sorted(train_indices),
        test_indices=sorted(test_indices),
        columns={"annotation_column": annotation_column, "donor_column": donor_column},
        heldout_values={"donors": test_donors},
        metadata=metadata,
    )
    validate_no_donor_leakage(split, obs_indexed, donor_column=donor_column)
    return split


def make_proportional_heldout_donor_and_value_split(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    donor_column: str,
    annotation_column: str,
    heldout_column: str,
    heldout_values: list[str],
    heldout_key: str,
    test_size: float,
    donor_test_size: float | None,
    random_state: int,
    total_cells: int | None = None,
    stratified_subsample: bool = True,
) -> SplitIndices:
    total_cells = _validate_total_cells(total_cells)
    if total_cells is not None:
        heldout_values = sorted(map(str, heldout_values))
        obs_indexed = obs.copy()
        obs_indexed.index = obs_indexed.index.astype(str)
        if donor_column not in obs_indexed.columns:
            raise ValueError(f"Missing donor_column in AnnData obs: {donor_column}")
        if annotation_column not in obs_indexed.columns:
            raise ValueError(f"Missing annotation_column in AnnData obs: {annotation_column}")
        if heldout_column not in obs_indexed.columns:
            raise ValueError(f"Missing heldout_column in AnnData obs: {heldout_column}")

        effective_donor_test_size = donor_test_size if donor_test_size is not None else test_size
        donors = np.array(sorted(obs_indexed[donor_column].dropna().astype(str).unique()))
        if len(donors) < 2:
            raise ValueError("heldout_donor split requires at least two donors.")

        rng = np.random.default_rng(random_state)
        shuffled_donors = rng.permutation(donors)
        n_test_donors = _split_count(
            len(shuffled_donors),
            effective_donor_test_size,
            require_both_sides=True,
        )
        test_donors = sorted(shuffled_donors[:n_test_donors].astype(str).tolist())
        donor_values = obs_indexed[donor_column].astype(str)
        train_pool_mask = ~donor_values.isin(test_donors)
        raw_train_candidates = obs_indexed.index[train_pool_mask].astype(str).tolist()
        test_candidates = obs_indexed.index[donor_values.isin(test_donors)].astype(str).tolist()
        heldout_mask = obs_indexed[heldout_column].astype(str).isin(heldout_values)
        train_candidates = obs_indexed.index[train_pool_mask & ~heldout_mask].astype(str).tolist()
        removed_count = len(raw_train_candidates) - len(train_candidates)

        train_target, test_target, selected_total = _allocate_train_test_targets(
            total_cells=total_cells,
            test_size=test_size,
            n_train_candidates=len(train_candidates),
            n_test_candidates=len(test_candidates),
        )
        if stratified_subsample:
            full_proportions = (
                obs_indexed[annotation_column]
                .astype(str)
                .value_counts(normalize=True, dropna=False)
                .sort_index()
            )
            train_indices = _stratified_subsample_indices(
                obs_indexed,
                train_candidates,
                annotation_column=annotation_column,
                target_count=train_target,
                full_proportions=full_proportions,
                random_state=random_state,
            )
            test_indices = _stratified_subsample_indices(
                obs_indexed,
                test_candidates,
                annotation_column=annotation_column,
                target_count=test_target,
                full_proportions=full_proportions,
                random_state=random_state + 1,
            )
        else:
            train_indices = _random_subsample_indices(
                train_candidates,
                target_count=train_target,
                random_state=random_state,
            )
            test_indices = _random_subsample_indices(
                test_candidates,
                target_count=test_target,
                random_state=random_state + 1,
            )

        split = SplitIndices(
            dataset_id=dataset_id,
            strategy=f"heldout_donor_and_{heldout_column}",
            random_state=random_state,
            train_indices=sorted(train_indices),
            test_indices=sorted(test_indices),
            columns={
                "annotation_column": annotation_column,
                "donor_column": donor_column,
                "heldout_column": heldout_column,
            },
            heldout_values={
                "donors": test_donors,
                heldout_key: heldout_values,
            },
            metadata={
                "test_size": test_size,
                "donor_test_size": effective_donor_test_size,
                "stratified_subsample": stratified_subsample,
                "total_cells": total_cells,
                "n_train_donors": len(donors) - n_test_donors,
                "n_test_donors": n_test_donors,
                "n_train_candidate_cells": len(train_candidates),
                "n_test_candidate_cells": len(test_candidates),
                "n_train_removed_heldout_cells": removed_count,
                "n_target_train_cells": train_target,
                "n_target_test_cells": test_target,
                "n_selected_total_cells": selected_total,
            },
        )
        validate_no_donor_leakage(split, obs_indexed, donor_column=donor_column)
        validate_heldout_absent_from_train(
            split,
            obs_indexed,
            column=heldout_column,
            values=heldout_values,
        )
        return split

    base_split = make_proportional_heldout_donor_split(
        obs,
        dataset_id=dataset_id,
        donor_column=donor_column,
        annotation_column=annotation_column,
        test_size=test_size,
        donor_test_size=donor_test_size,
        random_state=random_state,
        stratified_subsample=stratified_subsample,
    )
    heldout_values = sorted(map(str, heldout_values))
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    if heldout_column not in obs_indexed.columns:
        raise ValueError(f"Missing heldout_column in AnnData obs: {heldout_column}")

    train_values = obs_indexed.loc[base_split.train_indices, heldout_column].astype(str)
    kept_train_indices = train_values.index[~train_values.isin(heldout_values)].astype(str).tolist()
    removed_count = len(base_split.train_indices) - len(kept_train_indices)
    split = SplitIndices(
        dataset_id=dataset_id,
        strategy=f"heldout_donor_and_{heldout_column}",
        random_state=random_state,
        train_indices=sorted(kept_train_indices),
        test_indices=base_split.test_indices,
        columns={
            "annotation_column": annotation_column,
            "donor_column": donor_column,
            "heldout_column": heldout_column,
        },
        heldout_values={
            "donors": base_split.heldout_values.get("donors", []),
            heldout_key: heldout_values,
        },
        metadata={
            **base_split.metadata,
            "n_train_removed_heldout_cells": removed_count,
        },
    )
    validate_no_donor_leakage(split, obs_indexed, donor_column=donor_column)
    validate_heldout_absent_from_train(
        split,
        obs_indexed,
        column=heldout_column,
        values=heldout_values,
    )
    return split


def make_heldout_donor_and_value_split(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    donor_column: str,
    value_column: str,
    annotation_column: str,
    heldout_donors: list[str],
    heldout_values: list[str],
    heldout_key: str,
    random_state: int,
) -> SplitIndices:
    heldout_donors = sorted(map(str, heldout_donors))
    heldout_values = sorted(map(str, heldout_values))
    donor_values = obs[donor_column].astype(str)
    target_values = obs[value_column].astype(str)

    test_mask = donor_values.isin(heldout_donors)
    train_mask = ~test_mask & ~target_values.isin(heldout_values)
    split = SplitIndices(
        dataset_id=dataset_id,
        strategy=f"heldout_donor_and_{value_column}",
        random_state=random_state,
        train_indices=sorted(obs.index[train_mask].astype(str).tolist()),
        test_indices=sorted(obs.index[test_mask].astype(str).tolist()),
        columns={
            "annotation_column": annotation_column,
            "donor_column": donor_column,
            "heldout_column": value_column,
        },
        heldout_values={"donors": heldout_donors, heldout_key: heldout_values},
    )
    validate_no_donor_leakage(split, obs, donor_column=donor_column)
    validate_heldout_absent_from_train(
        split,
        obs,
        column=value_column,
        values=heldout_values,
    )
    return split


def summarize_split_composition(
    obs: pd.DataFrame,
    split: SplitIndices,
    *,
    columns: list[str],
) -> pd.DataFrame:
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    frames = []
    split_frames = {
        "full": obs_indexed,
        "train": obs_indexed.loc[split.train_indices],
        "test": obs_indexed.loc[split.test_indices],
    }

    for split_name, frame in split_frames.items():
        for column in columns:
            counts = (
                frame[column]
                .astype(str)
                .value_counts(dropna=False)
                .rename_axis("value")
                .reset_index(name="count")
            )
            counts["proportion"] = counts["count"] / len(frame) if len(frame) else 0.0
            counts["split"] = split_name
            counts["column"] = column
            frames.append(counts.loc[:, ["split", "column", "value", "count", "proportion"]])

    return pd.concat(frames, ignore_index=True)


def _plot_split_proportions(summary: pd.DataFrame, *, column: str, output_path: Path) -> Path:
    plot_df = summary[summary["column"] == column].copy()
    plt.figure(figsize=(max(6, 0.35 * plot_df["value"].nunique()), 4))
    sns.barplot(data=plot_df, x="value", y="proportion", hue="split")
    plt.xticks(rotation=90)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def _write_split_value_coverage(
    summary: pd.DataFrame,
    *,
    column: str,
    output_dir: Path,
) -> dict[str, str]:
    plot_df = summary[summary["column"] == column].copy()
    counts = (
        plot_df.pivot_table(
            index="value",
            columns="split",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=["full", "train", "test"], fill_value=0)
        .reset_index()
    )
    for split_name in ["full", "train", "test"]:
        counts[f"{split_name}_present"] = counts[split_name] > 0

    csv_path = output_dir / f"{column}_coverage.csv"
    counts.to_csv(csv_path, index=False)

    heatmap_df = counts.set_index("value")[
        ["full_present", "train_present", "test_present"]
    ].astype(int)
    plt.figure(figsize=(max(6, 0.35 * len(heatmap_df)), 2.8))
    sns.heatmap(
        heatmap_df.T,
        cmap="Greens",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt="d",
    )
    plt.xlabel(column)
    plt.ylabel("split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    figure_path = output_dir / f"{column}_coverage.pdf"
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    return {
        f"{column}_coverage": str(csv_path),
        f"{column}_coverage_pdf": str(figure_path),
    }


def _write_donor_leakage_report(
    obs: pd.DataFrame,
    split: SplitIndices,
    *,
    donor_column: str,
    output_path: Path,
) -> Path:
    obs_indexed = obs.copy()
    obs_indexed.index = obs_indexed.index.astype(str)
    train_donors = set(obs_indexed.loc[split.train_indices, donor_column].dropna().astype(str))
    test_donors = set(obs_indexed.loc[split.test_indices, donor_column].dropna().astype(str))
    leaked = sorted(train_donors.intersection(test_donors))
    report = pd.DataFrame({"donor": leaked, "leakage": [True] * len(leaked)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return output_path


def write_split_report(
    obs: pd.DataFrame,
    split: SplitIndices,
    *,
    output_dir: str | Path,
    annotation_column: str,
    donor_column: str | None = None,
    extra_columns: list[str] | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    columns = [annotation_column]
    if donor_column is not None:
        columns.append(donor_column)
    columns.extend(extra_columns or [])
    columns = list(dict.fromkeys(columns))

    summary = summarize_split_composition(obs, split, columns=columns)
    summary_path = output_path / "split_composition_summary.csv"
    summary.to_csv(summary_path, index=False)

    artifacts = {
        "summary": str(summary_path),
        "celltype_proportions": str(
            _plot_split_proportions(
                summary,
                column=annotation_column,
                output_path=output_path / "celltype_proportions.pdf",
            )
        ),
    }
    if donor_column is not None:
        artifacts["donor_proportions"] = str(
            _plot_split_proportions(
                summary,
                column=donor_column,
                output_path=output_path / "donor_proportions.pdf",
            )
        )
        artifacts["donor_leakage_report"] = str(
            _write_donor_leakage_report(
                obs,
                split,
                donor_column=donor_column,
                output_path=output_path / "donor_leakage_report.csv",
            )
        )
    for column in extra_columns or []:
        artifacts[f"{column}_proportions"] = str(
            _plot_split_proportions(
                summary,
                column=column,
                output_path=output_path / f"{column}_proportions.pdf",
            )
        )
        artifacts.update(
            _write_split_value_coverage(
                summary,
                column=column,
                output_dir=output_path,
            )
        )
    return artifacts


def write_generation_metadata(
    path: str | Path,
    *,
    split: SplitIndices,
    source: str,
    scrna_config: dict[str, Any],
    dataset_artifacts: dict[str, str],
    report_artifacts: dict[str, str],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "dataset_id": split.dataset_id,
        "source": source,
        "split": asdict(split),
        "scrna_config": scrna_config,
        "dataset_artifacts": dataset_artifacts,
        "report_artifacts": report_artifacts,
    }
    if extra_metadata:
        payload.update(extra_metadata)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path
