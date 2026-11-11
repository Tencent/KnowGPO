#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-in-Group Policy Optimization (GiGPO) Process Reward Module.

This is a faithful port of the reference implementation in
    https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py
adapted to the project's data layout:

  Reference gigpo:  one DataProto row == ONE step of ONE trajectory.
  This project:     one DataProto row == ONE full trajectory (SGLang multi-turn).

To preserve algorithmic equivalence:

  1. We parse each row's response text into T steps (re-using
     ``process_reward_gig.extract_steps_from_trajectory``).
  2. We compute each step's discounted return
         G_t = r_t + gamma * G_{t+1},     r_t = 0 for t < T,
                                          r_T = outcome_reward_of_this_trajectory
     so G_0 = gamma^{T-1} * outcome.
  3. We FLATTEN every (sample_idx, step_idx) into a single "virtual step"
     vector, then run the reference grouping / z-normalization routines
     exactly as in ``core_gigpo.py``.
  4. We re-aggregate each step's A_S back into a per-sample list of
     ``{"step_advantage", "token_start", "token_end"}`` dicts, which is the
     contract expected by ``ray_trainer.py`` (same schema as V3 GiG / V4 RGC).

Hyper-parameters (``gamma``, ``step_advantage_w``, ``mode``,
``enable_similarity``, ``similarity_thresh``) follow gigpo's
``examples/gigpo_trainer/run_search.sh`` defaults.

No external LLM API is used anywhere in this file: grouping is purely
string-similarity driven, which is the defining property of GiGPO w.r.t.
C-GRPO / GiG / RGC.

Author: auto-generated for anonymous_project baseline
Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Reference helpers (byte-for-byte equivalent to core_gigpo.py)
# ============================================================

def to_hashable(x: Any) -> Any:
    """Convert an object into a hashable type (used for clustering/grouping).

    Mirrors ``gigpo.core_gigpo.to_hashable``.
    """
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return tuple(x.flatten())
    if isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    raise TypeError(f"Unsupported type: {type(x)}")


def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """Return True if ``SequenceMatcher(a, b).ratio() >= threshold``.

    Mirrors ``gigpo.core_gigpo.are_similar``. Only text observations are
    supported; raises ValueError otherwise (preserve original contract).
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError(
            "Only text-based observations are supported for similarity-based GiGPO."
        )
    return SequenceMatcher(None, a, b).ratio() >= threshold


def summarize_group_size(group_size: List[int]) -> None:
    """Print a summary table of step-level group sizes. Diagnostic only."""
    if not group_size:
        return
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)
    logger.info("[GiGPO] Summary of step-level group sizes:")
    logger.info("[GiGPO] Size | Count | Proportion")
    logger.info("[GiGPO] -------------------------")
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0.0
        if prop:
            logger.info(f"[GiGPO] {size:>4} | {cnt:>5} | {prop:>9.2%}")


# ============================================================
# Step-level grouping (Eq. 6 in the paper)
# ============================================================

def build_step_group(
    anchor_obs: np.ndarray,
    index: np.ndarray,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    summarize: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """Cluster steps within each (prompt-level) group by anchor observation.

    This is the faithful port of ``gigpo.core_gigpo.build_step_group``.
    Two modes:

      - ``enable_similarity=False``: exact match on ``to_hashable(obs)``.
      - ``enable_similarity=True``:  ``SequenceMatcher`` ratio >= thresh.
        Uses greedy "first-cluster-that-matches" assignment, identical to
        the reference implementation.

    Args:
        anchor_obs: 1-D object array of length ``N_flat`` (one entry per
            flattened step) containing string observations that serve as the
            clustering key.
        index: 1-D object array of length ``N_flat`` with the prompt-level
            ``uid`` of each step. Clustering is done independently within
            each ``uid`` group.
        enable_similarity: Whether to do fuzzy clustering.
        similarity_thresh: Threshold for the fuzzy mode. Must be in (0, 1).
        summarize: If True, log a size-distribution summary.

    Returns:
        Tuple ``(step_group_uids, group_sizes)`` where
            - ``step_group_uids`` is an ``(N_flat,)`` object array of UUID
              strings, one per step, identifying the step-level group.
            - ``group_sizes`` is a flat list of cluster sizes (for metrics).
    """
    if enable_similarity:
        assert 0.0 < similarity_thresh < 1.0, (
            "similarity_thresh must be in (0, 1) when enable_similarity=True"
        )

    step_group_uids: np.ndarray = np.empty(len(anchor_obs), dtype=object)
    unique_indices = np.unique(index)
    group_sizes: List[int] = []

    for idx in unique_indices:
        locs = np.where(index == idx)[0]
        obs_group = anchor_obs[locs]

        if not enable_similarity:
            clusters: Dict[Any, List[int]] = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(locs[i])
            for _key, original_indices in clusters.items():
                uid = str(uuid.uuid4())
                group_sizes.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            # Dynamically maintain clusters; greedy first-match assignment.
            sim_clusters: List[Dict[str, Any]] = []
            for obs, loc in zip(obs_group, locs):
                placed = False
                for cluster in sim_clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                if not placed:
                    sim_clusters.append({"rep": obs, "locs": [loc]})
            for cluster in sim_clusters:
                uid = str(uuid.uuid4())
                group_sizes.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

    # Sanity check: every slot must have been assigned.
    if np.any(step_group_uids == None):  # noqa: E711
        missing_indices = np.where(step_group_uids == None)[0]  # noqa: E711
        raise ValueError(
            f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}"
        )

    if summarize:
        summarize_group_size(group_sizes)
    if group_sizes:
        logger.info(f"[GiGPO] Avg size of step-level group: {np.mean(group_sizes):.3f}")
    return step_group_uids, group_sizes


# ============================================================
# Normalization helpers (Eq. 3 & Eq. 7)
# ============================================================

def _normalize_by_group(
    scores: np.ndarray,
    group_ids: np.ndarray,
    remove_std: bool,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Generic z-score normalization within groups.

    Args:
        scores: 1-D float array of length N.
        group_ids: 1-D object array of length N with group identifiers.
        remove_std: If True (``mean_norm``), subtract group mean only.
            If False (``mean_std_norm``), divide by group std as well.
        epsilon: Numerical safety for std.

    Returns:
        Normalized scores array of the same shape as ``scores``.
    """
    out = scores.astype(np.float64).copy()
    id2score: Dict[Any, List[float]] = defaultdict(list)
    for i in range(len(out)):
        id2score[group_ids[i]].append(out[i])
    id2mean: Dict[Any, float] = {}
    id2std: Dict[Any, float] = {}
    for gid, values in id2score.items():
        if len(values) == 1:
            # Single-member group: zero out (same as gigpo reference semantics).
            id2mean[gid] = float(values[0]) if remove_std else 0.0
            id2std[gid] = 1.0
        else:
            arr = np.asarray(values, dtype=np.float64)
            id2mean[gid] = float(arr.mean())
            # NOTE: torch.std and np.std default to different ddof; gigpo uses
            # torch.std -> unbiased (ddof=1). Keep parity.
            id2std[gid] = float(arr.std(ddof=1)) if len(arr) > 1 else 1.0
    for i in range(len(out)):
        gid = group_ids[i]
        if remove_std:
            out[i] = out[i] - id2mean[gid]
        else:
            out[i] = (out[i] - id2mean[gid]) / (id2std[gid] + epsilon)
    return out


def episode_norm_scores(
    scores: np.ndarray,
    uid: np.ndarray,
    traj_uid: np.ndarray,
    remove_std: bool,
    compute_mean_std_cross_steps: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Episode-level advantage (Eq. 3), flattened-step variant.

    Faithful port of ``gigpo.core_gigpo.episode_norm_reward`` to numpy.

    Args:
        scores: 1-D array of length N_flat; each entry is the "episode score"
            of the trajectory this step belongs to (i.e., the trajectory's
            total outcome reward, broadcast to each of its steps).
        uid: 1-D object array of length N_flat with prompt-level group id.
        traj_uid: 1-D object array of length N_flat with trajectory-level id.
        remove_std: If True, mean-only normalization; else mean/std.
        compute_mean_std_cross_steps: If True (default, more stable), mean/std
            are computed over **all flattened steps** within a prompt-level
            group; if False, one score per trajectory is used (standard
            episode-level GRPO).
        epsilon: Numerical safety.

    Returns:
        1-D float array of length N_flat with episode-level advantages.
    """
    assert len(scores) == len(uid) == len(traj_uid)
    out = scores.astype(np.float64).copy()

    id2score: Dict[Any, List[float]] = defaultdict(list)
    seen_pairs: set = set()
    for i in range(len(out)):
        pair = (uid[i], traj_uid[i])
        if not compute_mean_std_cross_steps and pair in seen_pairs:
            continue
        id2score[uid[i]].append(float(out[i]))
        if not compute_mean_std_cross_steps:
            seen_pairs.add(pair)

    id2mean: Dict[Any, float] = {}
    id2std: Dict[Any, float] = {}
    for gid, values in id2score.items():
        if len(values) <= 1:
            id2mean[gid] = float(values[0]) if values else 0.0
            id2std[gid] = 1.0
        else:
            arr = np.asarray(values, dtype=np.float64)
            id2mean[gid] = float(arr.mean())
            id2std[gid] = float(arr.std(ddof=1)) if len(arr) > 1 else 1.0

    for i in range(len(out)):
        gid = uid[i]
        if remove_std:
            out[i] = out[i] - id2mean[gid]
        else:
            out[i] = (out[i] - id2mean[gid]) / (id2std[gid] + epsilon)
    return out


def step_norm_scores(
    step_rewards: np.ndarray,
    step_group_uids: np.ndarray,
    remove_std: bool,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Step-level advantage (Eq. 7), flattened-step variant.

    Faithful port of ``gigpo.core_gigpo.step_norm_reward`` to numpy.

    Args:
        step_rewards: 1-D float array of length N_flat with per-step
            discounted returns ``G_t``.
        step_group_uids: 1-D object array with the step-level cluster id
            produced by :func:`build_step_group`.
        remove_std: If True, mean-only normalization; else mean/std.
        epsilon: Numerical safety.

    Returns:
        1-D float array of length N_flat with step-level advantages.
    """
    return _normalize_by_group(
        step_rewards, step_group_uids, remove_std=remove_std, epsilon=epsilon
    )


# ============================================================
# Project-specific adapters (trajectory -> flattened steps)
# ============================================================

def compute_step_returns_from_trajectory(
    num_steps: int,
    outcome_reward: float,
    gamma: float = 0.95,
) -> List[float]:
    """Discounted returns within a single trajectory (Eq. 5).

    In this project's setup, intermediate steps carry reward 0 and only the
    final step receives the trajectory's outcome reward, hence

        G_T-1 = outcome
        G_t   = gamma * G_{t+1},    t = T-2, ..., 0

    which is exactly equivalent to gigpo's
    ``compute_step_discounted_returns`` applied to a single-trajectory
    reward vector ``[0, 0, ..., 0, outcome]``.

    Args:
        num_steps: Number of parsed steps in the trajectory (T).
        outcome_reward: Scalar outcome reward of the whole trajectory.
        gamma: Discount factor (gigpo default: 0.95).

    Returns:
        Python list of length ``num_steps`` with ``G_t`` for ``t = 0..T-1``.
    """
    if num_steps <= 0:
        return []
    returns = [0.0] * num_steps
    running = 0.0
    for t in range(num_steps - 1, -1, -1):
        r_t = float(outcome_reward) if t == num_steps - 1 else 0.0
        running = r_t + float(gamma) * running
        returns[t] = running
    return returns


def _pick_anchor_text(step: Dict[str, Any]) -> str:
    """Pick the "anchor observation" string used for step-level clustering.

    GiGPO's anchor_obs is the environment-returned observation with NO
    per-trajectory history mixed in. In this project the closest analogue is
    the ``<tool_response>`` content of that step (i.e., the search results
    returned by the external tool). If a step has no ``tool_response``
    (e.g., the final ``<answer>`` step, or a parse failure), we fall back
    to the action string so that final-answer steps are never silently
    merged across trajectories.
    """
    resp = step.get("response", "") or ""
    if resp.strip():
        return resp
    action = step.get("action", "") or ""
    if action.strip():
        return action
    # Last-resort: unique sentinel so singletons never collide.
    return f"__empty_step_{uuid.uuid4()}__"


def compute_gigpo_step_advantage_for_batch(
    steps_per_sample: List[List[Dict[str, Any]]],
    outcome_rewards: List[float],
    uid_list: List[str],
    traj_uid_list: Optional[List[str]] = None,
    gamma: float = 0.95,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = True,
    similarity_thresh: float = 0.9,
    compute_mean_std_cross_steps: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, float]]:
    """End-to-end GiGPO step-advantage computation for a training batch.

    This is the project-side counterpart of gigpo's
    ``compute_gigpo_outcome_advantage`` but operates on parsed-step lists
    instead of step-wise DataProto rows. It returns the same schema as V3
    GiG / V4 RGC so that downstream ``ray_trainer.py`` logic can reuse the
    existing ``gig_step_advantages`` hand-off path.

    Pipeline:

        Step 1. Flatten ``(sample_idx, step_idx)`` -> ``N_flat`` virtual steps.
        Step 2. For each virtual step, compute the per-trajectory discounted
                return ``G_t`` and broadcast the trajectory's outcome to
                ``scores_flat`` (for optional episode-norm ablation).
        Step 3. Cluster steps via ``build_step_group`` on anchor_obs strings.
        Step 4. Normalize ``G_t`` within each step-level cluster -> ``A_S``.
        Step 5. Re-aggregate ``A_S`` per (sample, step) with token boundaries.

    Args:
        steps_per_sample: ``steps_per_sample[i][t]`` is the step dict for
            trajectory ``i`` step ``t`` produced by
            ``extract_steps_from_trajectory``. MUST already carry
            ``char_start`` / ``char_end`` fields for token mapping.
        outcome_rewards: List of length B, the scalar outcome reward of
            each trajectory (what the reward manager writes to the EOS of
            ``token_level_scores``).
        uid_list: List of length B with prompt-level ``uid`` (same value
            repeats ``rollout.n`` times).
        traj_uid_list: List of length B with trajectory-level id. If
            ``None``, use ``f"{uid}__{i}"`` which is sufficient because
            each list row corresponds to one full trajectory in this
            project's data layout.
        gamma: Discount factor (gigpo reference default: 0.95).
        step_advantage_w: Weight of ``A_S`` in the final fusion ``A = A_E +
            step_advantage_w * A_S``. This function only returns ``A_S``;
            the actual fusion is applied downstream by ray_trainer.
        mode: ``"mean_std_norm"`` (default, gigpo search config) or
            ``"mean_norm"``. Controls both ``A_E`` and ``A_S`` normalization.
        enable_similarity: If True, use SequenceMatcher clustering
            (gigpo run_search.sh default for text tasks).
        similarity_thresh: Similarity threshold for the fuzzy mode
            (gigpo run_search.sh: 0.9).
        compute_mean_std_cross_steps: Episode-norm variance source; True is
            more stable and matches gigpo's ``episode_norm_reward`` default.
        epsilon: Numerical safety for std.

    Returns:
        Tuple ``(step_advantages_per_sample, metrics)`` where:

          - ``step_advantages_per_sample`` is a ``len(B)`` list; each element
            is a list of step dicts with keys:
              * ``step_advantage``: scalar ``A_S`` for this step (float).
              * ``char_start`` / ``char_end``: forwarded from the parser for
                downstream token-level mapping.
              * ``step_return``: raw ``G_t`` (for logging / dump).
              * ``step_group_uid``: the cluster id this step belongs to.
          - ``metrics`` contains GiGPO-specific summary numbers (group size
            distribution, episode/step adv stats, etc.).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown GiGPO mode: {mode!r}")

    batch_size = len(steps_per_sample)
    assert len(outcome_rewards) == batch_size
    assert len(uid_list) == batch_size
    if traj_uid_list is None:
        # One row == one trajectory in this project: use a deterministic id
        # that is unique across the batch while scoped inside a single uid.
        traj_uid_list = [f"{uid_list[i]}__{i}" for i in range(batch_size)]
    else:
        assert len(traj_uid_list) == batch_size

    # --- Step 1: flatten ---
    flat_anchor: List[str] = []
    flat_uid: List[str] = []
    flat_traj_uid: List[str] = []
    flat_return: List[float] = []
    flat_outcome: List[float] = []
    flat_locator: List[Tuple[int, int]] = []  # (sample_idx, step_idx)

    for i in range(batch_size):
        steps_i = steps_per_sample[i]
        if not steps_i:
            continue
        returns_i = compute_step_returns_from_trajectory(
            num_steps=len(steps_i),
            outcome_reward=float(outcome_rewards[i]),
            gamma=gamma,
        )
        for t, step in enumerate(steps_i):
            flat_anchor.append(_pick_anchor_text(step))
            flat_uid.append(str(uid_list[i]))
            flat_traj_uid.append(str(traj_uid_list[i]))
            flat_return.append(returns_i[t])
            flat_outcome.append(float(outcome_rewards[i]))
            flat_locator.append((i, t))

    n_flat = len(flat_anchor)
    if n_flat == 0:
        empty_out: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        return empty_out, {
            "gigpo/num_steps_total": 0.0,
            "gigpo/num_samples_with_steps": 0.0,
            "gigpo/avg_steps_per_trajectory": 0.0,
            "gigpo/step_advantage_abs_mean": 0.0,
            "gigpo/step_advantage_std": 0.0,
            "gigpo/avg_step_group_size": 0.0,
            "gigpo/singleton_group_ratio": 0.0,
        }

    flat_anchor_np = np.array(flat_anchor, dtype=object)
    flat_uid_np = np.array(flat_uid, dtype=object)
    flat_traj_uid_np = np.array(flat_traj_uid, dtype=object)
    flat_return_np = np.array(flat_return, dtype=np.float64)
    flat_outcome_np = np.array(flat_outcome, dtype=np.float64)

    # --- Step 2 (optional): episode-level norm (for dump / ablation only).
    # The production path uses GRPO's own A_E from ray_trainer. We still
    # compute a reference A_E_flat here for logging parity with gigpo.
    try:
        _ = episode_norm_scores(
            scores=flat_outcome_np,
            uid=flat_uid_np,
            traj_uid=flat_traj_uid_np,
            remove_std=remove_std,
            compute_mean_std_cross_steps=compute_mean_std_cross_steps,
            epsilon=epsilon,
        )
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning(f"[GiGPO] episode_norm_scores for diagnostics failed: {exc}")

    # --- Step 3: anchor-state clustering within each uid group.
    step_group_uids, group_sizes = build_step_group(
        anchor_obs=flat_anchor_np,
        index=flat_uid_np,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
        summarize=False,
    )

    # --- Step 4: step-level z-norm -> A_S per flattened step.
    step_advantages_flat = step_norm_scores(
        step_rewards=flat_return_np,
        step_group_uids=step_group_uids,
        remove_std=remove_std,
        epsilon=epsilon,
    )

    # --- Step 5: re-aggregate per (sample, step) with char boundaries.
    per_sample: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
    for k, (i, t) in enumerate(flat_locator):
        step = steps_per_sample[i][t]
        per_sample[i].append(
            {
                "step_advantage": float(step_advantages_flat[k]),
                "step_return": float(flat_return_np[k]),
                "step_group_uid": str(step_group_uids[k]),
                "char_start": int(step.get("char_start", 0)),
                "char_end": int(step.get("char_end", 0)),
                "is_final": bool(step.get("is_final", t == len(steps_per_sample[i]) - 1)),
            }
        )

    # --- metrics ---
    num_samples_with_steps = sum(1 for s in steps_per_sample if s)
    avg_steps = (
        float(np.mean([len(s) for s in steps_per_sample if s]))
        if num_samples_with_steps
        else 0.0
    )
    if group_sizes:
        avg_group_size = float(np.mean(group_sizes))
        singleton_ratio = float(
            sum(1 for g in group_sizes if g == 1) / max(len(group_sizes), 1)
        )
    else:
        avg_group_size = 0.0
        singleton_ratio = 0.0

    metrics = {
        "gigpo/num_steps_total": float(n_flat),
        "gigpo/num_samples_with_steps": float(num_samples_with_steps),
        "gigpo/avg_steps_per_trajectory": avg_steps,
        "gigpo/step_advantage_abs_mean": float(np.mean(np.abs(step_advantages_flat))),
        "gigpo/step_advantage_std": float(np.std(step_advantages_flat)),
        "gigpo/avg_step_group_size": avg_group_size,
        "gigpo/singleton_group_ratio": singleton_ratio,
        "gigpo/num_step_groups": float(len(group_sizes)),
        "gigpo/step_advantage_w": float(step_advantage_w),
        "gigpo/gamma": float(gamma),
        "gigpo/similarity_thresh": float(similarity_thresh) if enable_similarity else 0.0,
        "gigpo/mode_is_mean_std_norm": 1.0 if not remove_std else 0.0,
    }
    return per_sample, metrics


# ============================================================
# Public API re-exports
# ============================================================

__all__ = [
    "to_hashable",
    "are_similar",
    "summarize_group_size",
    "build_step_group",
    "episode_norm_scores",
    "step_norm_scores",
    "compute_step_returns_from_trajectory",
    "compute_gigpo_step_advantage_for_batch",
]
