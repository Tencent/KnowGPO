#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGC (Rubric-Guided Grouping with Counterfactual Reward) Process Reward Module

Core idea: Use rubric states ONLY for grouping, NOT for reward computation.

Two length-penalty modes supported:

 - Mode "none":
      Step value v = R_final (pure RGC baseline).
      Advantage A_S = (v - group_mean) / (group_std + eps).

 - Mode "group_zscore"  (Plan B, recommended):
      For each (uid, rubric_state) group, compute a z-score of the
      *trajectory total length* across group members, and mix it into the
      joint value:
         v_joint_i = R_final_i - alpha_length * (L_i - L_mean) / (L_std + eps)
      Then normalize within the group:
         A_S_i = (v_joint_i - V_group_mean) / (V_group_std + eps).

      Short trajectories (L < L_mean) -> v_joint bigger -> positive A_S;
      long trajectories (L > L_mean) -> v_joint smaller -> negative A_S,
      independent of correctness.

 - Mode "saturating":
      Trajectory-level shrink of R_final before grouping:
         v_i = R_final_i * (1 - alpha_length * saturate(L_i, target))
      where saturate() is 0 below `length_penalty_target` and ramps to 1.
      Then same within-group normalization as above.

For ALL modes the outputs are structurally identical to GiG's Phase B so
that downstream `align_and_fuse_advantages(A = A_E + omega * A_S)`
in `ray_trainer.py` works with zero changes.

Downstream contract (each step dict):
    {
        "trajectory_index": int,
        "step_index": int,
        "corrected_state": List[int],      # pre-action rubric state
        "post_action_state": List[int],
        "immediate_reward": float,         # kept for logging (0.0 in RGC)
        "is_final": bool,
        "uid": str,
        "step_text": str,
        "char_start": int, "char_end": int,
        "normalized_reward": float,
        "step_advantage": float,           # <- the A_S value used by trainer
        "group_id": Optional[str],
        # diagnostic fields:
        "rgc_value": float,                # v_joint before normalization
        "rgc_length_offset": float,        # (L_i - L_mean)/(L_std+eps)
        "rgc_traj_length": int,            # total non-final steps in the trajectory
    }

Author: anonymous
Version: 4.1.0 (Plan B: intra-group length penalty)
Date: 2026-04-21
"""

import logging
import math
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse step extraction / state evaluation / token mapping / dump helpers
from verl.utils.reward_score.process_reward_gig import (
    extract_steps_from_trajectory,
    compute_process_reward_phase_a,
    map_step_advantages_to_tokens,
    dump_grouping_details,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================


def _saturating_length_factor(length: int, target: int) -> float:
    """Return 0.0 when length <= target, linearly ramping to 1.0 at 2*target."""
    if target <= 0:
        return 0.0
    excess = max(0, length - target)
    return min(1.0, excess / float(target))


def _safe_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _safe_mean_std_unbiased(values: List[float]) -> Tuple[float, float]:
    """Unbiased variant: matches torch.std (ddof=1), used by V6 RGC-GiGPO.

    We keep the biased version above for backwards compatibility with V4 RGC
    (Plan B) so its training runs are byte-compatible with prior experiments.
    """
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ============================================================
# Module 2: RGC Step-Level Advantage Computation (Plan B inside)
# ============================================================


def compute_rgc_step_advantages(
    all_steps_per_sample: List[List[Dict[str, Any]]],
    corrected_state_vectors_per_sample: List[List[List[int]]],
    outcome_rewards: List[float],
    uid_list: List[str],
    *,
    # length penalty configuration (Plan B)
    length_penalty_mode: str = "group_zscore",      # "none" | "group_zscore" | "saturating"
    length_penalty_alpha: float = 0.3,
    length_penalty_target: int = 8,
    length_min_group_size: int = 2,
    length_excl_correct_only: bool = False,
    # numerical
    epsilon: float = 1e-8,
    # optional (deprecated, kept for backward compat)
    efficiency_penalty_alpha: float = 0.0,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any], List[Dict[str, Any]]]:
    """Compute step-level advantages using the RGC rule, with Plan B length
    penalty applied *inside* each rubric-state group.

    This function mirrors GiG's `compute_step_level_advantages` signature and
    return shape so that the rest of the pipeline (token mapping, advantage
    fusion) can remain unchanged.

    Args:
        all_steps_per_sample: Per-sample list of step dicts from Phase A,
            each containing: 'step_index', 'is_final', 'cumulative_text',
            'think'/'action'/'response' (optional), 'formatted_cumulative_text'.
        corrected_state_vectors_per_sample: Per-sample monotonicity-corrected
            post-action state vectors (List[List[int]]).
        outcome_rewards: R_final for each trajectory (0 or 1).
        uid_list: Query UID for each trajectory (for per-query grouping).
        length_penalty_mode: See module docstring.
        length_penalty_alpha: Strength of the length signal.
        length_penalty_target: Target length in "saturating" mode.
        length_min_group_size: Require >= N members per group to apply length
            penalty; otherwise fall back to pure R_final.
        length_excl_correct_only: If True, apply length penalty only to
            correct trajectories (R=1). Prevents worsening "search avoidance".
        epsilon: Numerical stability epsilon.
        efficiency_penalty_alpha: Legacy per-step-position shrink factor.

    Returns:
        (step_data_per_sample, metrics, group_details)
        The first element is a list (one entry per sample) of lists of step
        dicts containing `step_advantage` and related fields, compatible with
        downstream tooling.
    """
    batch_size = len(all_steps_per_sample)
    assert len(corrected_state_vectors_per_sample) == batch_size
    assert len(outcome_rewards) == batch_size
    assert len(uid_list) == batch_size

    # ---------- Build per-step records and trajectory-level stats ----------
    all_step_data: List[Dict[str, Any]] = []
    step_data_per_sample: List[List[Dict[str, Any]]] = []

    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        corrected_svs = corrected_state_vectors_per_sample[i]
        outcome_r = float(outcome_rewards[i]) if i < len(outcome_rewards) else 0.0

        sample_records: List[Dict[str, Any]] = []
        num_rubrics = len(corrected_svs[0]) if corrected_svs and corrected_svs[0] else 0

        for j, step in enumerate(steps):
            pre_action_state = (
                [0] * num_rubrics if j == 0
                else (corrected_svs[j - 1] if (j - 1) < len(corrected_svs) else [0] * num_rubrics)
            )
            post_action_state = (
                corrected_svs[j] if j < len(corrected_svs) else pre_action_state
            )

            # Keep a compact step_text for dumping only
            step_text_parts: List[str] = []
            if step.get("think"):
                step_text_parts.append(f"<think>{step['think']}</think>")
            if step.get("action"):
                step_text_parts.append(step["action"])
            if step.get("response"):
                step_text_parts.append(step["response"])
            step_text = "\n".join(step_text_parts)

            if j == 0:
                step_input = ""
            else:
                prev_step = steps[j - 1]
                step_input = prev_step.get("formatted_cumulative_text", "")

            record = {
                "trajectory_index": i,
                "step_index": step.get("step_index", j),
                "corrected_state": pre_action_state,
                "post_action_state": post_action_state,
                "state_transition": "rgc",  # placeholder, not used by RGC
                "immediate_reward": 0.0,    # RGC does NOT use rubric_delta
                "is_final": bool(step.get("is_final", j == len(steps) - 1)),
                "uid": uid_list[i] if i < len(uid_list) else "",
                "step_text": step_text,
                "step_input": step_input,
                # filled later
                "normalized_reward": 0.0,
                "step_advantage": 0.0,
                "group_id": None,
                "rgc_value": outcome_r,
                "rgc_length_offset": 0.0,
                "rgc_traj_length": 0,
                # outcome bookkeeping
                "_outcome": outcome_r,
            }
            all_step_data.append(record)
            sample_records.append(record)

        step_data_per_sample.append(sample_records)

    # ---------- Split final vs non-final ----------
    non_final_items = [it for it in all_step_data if not it["is_final"]]
    final_items = [it for it in all_step_data if it["is_final"]]

    # Trajectory length = number of non-final steps (= "search/reason" budget used)
    traj_length_by_idx: Dict[int, int] = defaultdict(int)
    for it in non_final_items:
        traj_length_by_idx[it["trajectory_index"]] += 1
    for it in all_step_data:
        it["rgc_traj_length"] = traj_length_by_idx.get(it["trajectory_index"], 0)

    # ---------- Group by (uid, pre-action state) ----------
    groups: Dict[Tuple[str, Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for it in non_final_items:
        key = (it.get("uid", ""), tuple(it["corrected_state"]))
        groups[key].append(it)

    # ---------- Within-group advantage computation ----------
    group_details: List[Dict[str, Any]] = []
    effective_groups = 0
    singleton_groups = 0
    length_active_groups = 0
    total_steps_with_signal = 0

    # stats for logging
    length_offset_pool: List[float] = []
    joint_value_pool: List[float] = []
    adv_pool: List[float] = []

    # --- Distribution counters (P0: per-batch group distribution) ---
    # Track every group's size, outcome-diversity and length-std into the
    # three dimensions that `effective_groups` alone hides.
    group_size_hist = {  # Bin A: group-size histogram
        "1": 0, "2": 0, "3_4": 0, "5_8": 0, "9_16": 0, "17plus": 0,
    }
    group_size_list: List[int] = []              # for percentiles

    outcome_all_correct = 0                       # Bin B: outcome-diversity
    outcome_all_wrong = 0
    outcome_mixed = 0
    mixed_group_steps = 0                         # steps inside mixed groups

    length_std_hist = {                           # Bin C: L_std histogram
        "zero": 0, "0_1": 0, "1_3": 0, "3plus": 0,
    }
    length_std_list: List[float] = []             # for percentiles
    length_cv_list: List[float] = []              # CV = L_std/L_mean
    # ------------------------------------------------------------------

    mode = (length_penalty_mode or "none").lower()
    alpha_len = float(length_penalty_alpha)
    target_len = int(length_penalty_target)
    min_group_for_length = max(2, int(length_min_group_size))

    for gid_idx, (group_key, members) in enumerate(groups.items()):
        gid = f"g{gid_idx}"
        uid, state_tuple = group_key
        group_size = len(members)

        # Collect group-level stats we may need
        outcomes = [float(m["_outcome"]) for m in members]
        lengths = [int(m["rgc_traj_length"]) for m in members]

        L_mean, L_std = _safe_mean_std(lengths)

        # --- Distribution counters update (Bin A: group size) ---
        group_size_list.append(group_size)
        if group_size == 1:
            group_size_hist["1"] += 1
        elif group_size == 2:
            group_size_hist["2"] += 1
        elif group_size <= 4:
            group_size_hist["3_4"] += 1
        elif group_size <= 8:
            group_size_hist["5_8"] += 1
        elif group_size <= 16:
            group_size_hist["9_16"] += 1
        else:
            group_size_hist["17plus"] += 1

        # --- Distribution counters update (Bin B: outcome diversity) ---
        # Classify group by whether R values are uniform or mixed.
        # A group of size 1 is trivially "uniform" but contributes no gradient.
        n_correct = sum(1 for o in outcomes if o > 0.5)
        n_wrong = group_size - n_correct
        if group_size >= 2:
            if n_correct == group_size:
                outcome_all_correct += 1
            elif n_wrong == group_size:
                outcome_all_wrong += 1
            else:
                outcome_mixed += 1
                mixed_group_steps += group_size

        # --- Distribution counters update (Bin C: L_std) ---
        length_std_list.append(L_std)
        if L_std < epsilon:
            length_std_hist["zero"] += 1
        elif L_std <= 1.0:
            length_std_hist["0_1"] += 1
        elif L_std <= 3.0:
            length_std_hist["1_3"] += 1
        else:
            length_std_hist["3plus"] += 1
        if L_mean > epsilon:
            length_cv_list.append(L_std / L_mean)
        # -------------------------------------------------------------

        # Decide whether to apply length penalty for this group
        apply_length_penalty = (
            mode != "none"
            and alpha_len > 0.0
            and group_size >= min_group_for_length
            and L_std > epsilon  # zero std = all same length, nothing to learn
        )
        if apply_length_penalty:
            length_active_groups += 1

        # Compute joint values v_joint_i per member
        joint_values: List[float] = []
        length_offsets: List[float] = []
        for m in members:
            R_i = float(m["_outcome"])
            L_i = int(m["rgc_traj_length"])

            # base value
            v_i = R_i
            if efficiency_penalty_alpha > 0 and L_i > 1:
                # legacy per-step-position penalty (OFF by default)
                step_pos = int(m["step_index"]) / max(1, L_i - 1)
                v_i = R_i - float(efficiency_penalty_alpha) * step_pos

            # length penalty
            length_offset_i = 0.0
            if apply_length_penalty:
                if length_excl_correct_only and R_i <= 0.5:
                    # do not penalize wrong trajectories by length
                    length_offset_i = 0.0
                else:
                    if mode == "group_zscore":
                        length_offset_i = (L_i - L_mean) / max(L_std, epsilon)
                        v_i = v_i - alpha_len * length_offset_i
                    elif mode == "saturating":
                        sat = _saturating_length_factor(L_i, target_len)
                        length_offset_i = sat
                        # shrink positive rewards for long trajectories.
                        # Wrong trajectories keep R=0 so the factor vanishes;
                        # this naturally avoids aggravating "search avoidance".
                        v_i = R_i * (1.0 - alpha_len * sat)

            m["rgc_value"] = v_i
            m["rgc_length_offset"] = length_offset_i
            joint_values.append(v_i)
            length_offsets.append(length_offset_i)

        v_mean, v_std = _safe_mean_std(joint_values)

        # singleton / all-equal group: no gradient signal
        is_effective = group_size > 1 and v_std > epsilon
        if not is_effective:
            if group_size == 1:
                singleton_groups += 1
            # normalized_reward / step_advantage already defaulted to 0
            for m in members:
                m["normalized_reward"] = 0.0
                m["step_advantage"] = 0.0
                m["group_id"] = gid
        else:
            effective_groups += 1
            for m, v_i in zip(members, joint_values):
                norm = (v_i - v_mean) / max(v_std, epsilon)
                m["normalized_reward"] = norm
                m["step_advantage"] = norm  # RGC uses no GAE discount: A_S_t = norm
                m["group_id"] = gid
                total_steps_with_signal += 1
                adv_pool.append(norm)

        # bookkeeping pools
        length_offset_pool.extend(length_offsets)
        joint_value_pool.extend(joint_values)

        # group detail dump
        reward_dist: Dict[str, int] = {}
        for o in outcomes:
            k = f"{float(o):.1f}"
            reward_dist[k] = reward_dist.get(k, 0) + 1
        state_list = list(state_tuple)
        num_rubrics = len(state_list)
        satisfied_count = sum(state_list)
        satisfied_indices = [f"C{i + 1}" for i, v in enumerate(state_list) if v == 1]
        if satisfied_indices:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied ({', '.join(satisfied_indices)})"
        else:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied"

        group_details.append({
            "group_id": gid,
            "uid": uid,
            "pre_action_state": state_list,
            "pre_action_state_summary": state_summary,
            "group_size": group_size,
            "reward_mean": float(np.mean(outcomes)) if outcomes else 0.0,
            "reward_std": float(np.std(outcomes)) if outcomes else 0.0,
            "length_mean": L_mean,
            "length_std": L_std,
            "length_penalty_active": apply_length_penalty,
            "joint_value_mean": v_mean,
            "joint_value_std": v_std,
            "is_effective": is_effective,
            "reward_distribution": reward_dist,
            "members": [
                {
                    "trajectory_index": m.get("trajectory_index", -1),
                    "step_index": m["step_index"],
                    "outcome_reward": float(m["_outcome"]),
                    "traj_length": int(m["rgc_traj_length"]),
                    "length_offset": float(m.get("rgc_length_offset", 0.0)),
                    "joint_value": float(m.get("rgc_value", m["_outcome"])),
                    "step_advantage": float(m.get("step_advantage", 0.0)),
                }
                for m in members
            ],
        })

    # ---------- Final steps ----------
    # As in GiG: outcome is the A_E signal domain, not A_S.
    for it in final_items:
        it["normalized_reward"] = 0.0
        it["step_advantage"] = 0.0
        it["group_id"] = None

    # ---------- Char boundaries for token mapping ----------
    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        sample_records = step_data_per_sample[i]
        for j, (step, sd) in enumerate(zip(steps, sample_records)):
            cum_len = len(step.get("cumulative_text", ""))
            if j == 0:
                sd["char_start"] = 0
            else:
                prev_cum_len = len(steps[j - 1].get("cumulative_text", ""))
                sd["char_start"] = prev_cum_len
            sd["char_end"] = cum_len

    # ---------- Strip internal fields that should not leak downstream ----------
    for it in all_step_data:
        it.pop("_outcome", None)

    # ---------- Metrics ----------
    non_zero_adv = [a for a in adv_pool if abs(a) > epsilon]

    # Percentile helper (safe for empty lists)
    def _pct(arr, q):
        if not arr:
            return 0.0
        return float(np.percentile(arr, q))

    total_groups_n = max(1, len(groups))

    metrics: Dict[str, Any] = {
        "rgc/num_state_groups": len(groups),
        "rgc/effective_groups": effective_groups,
        "rgc/singleton_groups": singleton_groups,
        "rgc/length_active_groups": length_active_groups,
        "rgc/num_non_final_steps": len(non_final_items),
        "rgc/num_final_steps": len(final_items),
        "rgc/num_total_steps": len(all_step_data),
        "rgc/non_zero_advantage_count": len(non_zero_adv),
        "rgc/mean_step_advantage": float(np.mean(non_zero_adv)) if non_zero_adv else 0.0,
        "rgc/std_step_advantage": float(np.std(non_zero_adv)) if non_zero_adv else 0.0,
        "rgc/mean_joint_value": float(np.mean(joint_value_pool)) if joint_value_pool else 0.0,
        "rgc/mean_length_offset_abs": float(np.mean(np.abs(length_offset_pool))) if length_offset_pool else 0.0,
        "rgc/length_penalty_mode": mode,
        "rgc/length_penalty_alpha": alpha_len,
        "rgc/signal_ratio": (
            total_steps_with_signal / max(1, len(non_final_items))
        ),
        # ========================================================
        # Per-batch group distribution histograms (P0)
        # ========================================================
        # Bin A: group-size histogram
        "rgc_dist/group_size_bin_1":       group_size_hist["1"],
        "rgc_dist/group_size_bin_2":       group_size_hist["2"],
        "rgc_dist/group_size_bin_3_4":     group_size_hist["3_4"],
        "rgc_dist/group_size_bin_5_8":     group_size_hist["5_8"],
        "rgc_dist/group_size_bin_9_16":    group_size_hist["9_16"],
        "rgc_dist/group_size_bin_17plus":  group_size_hist["17plus"],
        "rgc_dist/group_size_p25":         _pct(group_size_list, 25),
        "rgc_dist/group_size_p50":         _pct(group_size_list, 50),
        "rgc_dist/group_size_p75":         _pct(group_size_list, 75),
        "rgc_dist/group_size_p90":         _pct(group_size_list, 90),
        # Bin B: outcome-diversity — the REAL "has-gradient" signal
        "rgc_dist/groups_all_correct":     outcome_all_correct,
        "rgc_dist/groups_all_wrong":       outcome_all_wrong,
        "rgc_dist/groups_mixed":           outcome_mixed,
        "rgc_dist/mixed_group_ratio":      outcome_mixed / total_groups_n,
        "rgc_dist/outcome_diverse_steps":  mixed_group_steps,
        # Bin C: L_std distribution — Plan B activation granularity
        "rgc_dist/groups_L_std_zero":      length_std_hist["zero"],
        "rgc_dist/groups_L_std_0_1":       length_std_hist["0_1"],
        "rgc_dist/groups_L_std_1_3":       length_std_hist["1_3"],
        "rgc_dist/groups_L_std_3plus":     length_std_hist["3plus"],
        "rgc_dist/length_std_p50":         _pct(length_std_list, 50),
        "rgc_dist/length_std_p90":         _pct(length_std_list, 90),
        "rgc_dist/length_cv_p50":          _pct(length_cv_list, 50) if length_cv_list else 0.0,
        # Convenience ratios (derived)
        "rgc_dist/singleton_ratio":        group_size_hist["1"] / total_groups_n,
        "rgc_dist/effective_ratio":        effective_groups / total_groups_n,
        "rgc_dist/length_active_ratio":    length_active_groups / total_groups_n,
        # ========================================================
        # GiG-compatible keys (for dashboards/back-compat)
        "gig/num_state_groups": len(groups),
        "gig/avg_group_size": float(np.mean([len(v) for v in groups.values()])) if groups else 0.0,
        "gig/max_group_size": int(max([len(v) for v in groups.values()], default=0)),
        "gig/num_non_final_steps": len(non_final_items),
        "gig/num_final_steps": len(final_items),
        "gig/num_total_steps": len(all_step_data),
    }

    logger.info(
        f"[RGC Phase B] groups={len(groups)} effective={effective_groups} "
        f"singletons={singleton_groups} length_active={length_active_groups} "
        f"signal_ratio={metrics['rgc/signal_ratio']:.3f} "
        f"mean|A_S|={metrics['rgc/mean_step_advantage']:.3f} mode={mode} alpha={alpha_len}"
    )
    logger.info(
        f"[RGC Dist] size_bins=(1:{group_size_hist['1']}, 2:{group_size_hist['2']}, "
        f"3-4:{group_size_hist['3_4']}, 5-8:{group_size_hist['5_8']}, "
        f"9-16:{group_size_hist['9_16']}, 17+:{group_size_hist['17plus']}) | "
        f"outcome=(all-correct:{outcome_all_correct}, all-wrong:{outcome_all_wrong}, "
        f"mixed:{outcome_mixed}) mixed_ratio={metrics['rgc_dist/mixed_group_ratio']:.2f} | "
        f"L_std_bins=(0:{length_std_hist['zero']}, 0-1:{length_std_hist['0_1']}, "
        f"1-3:{length_std_hist['1_3']}, 3+:{length_std_hist['3plus']}) "
        f"L_std_p50={metrics['rgc_dist/length_std_p50']:.2f}"
    )

    return step_data_per_sample, metrics, group_details


# ============================================================
# Module 3: Phase B wrapper (GiG-compatible signature)
# ============================================================


def compute_rgc_phase_b(
    phase_a_result: Dict[str, Any],
    trajectories: List[str],
    questions: List[str],
    rubrics_list: List[List[str]],
    outcome_rewards: List[float],
    uid_list: List[str],
    config: Optional[Dict[str, Any]] = None,
    *,
    # mirror GiG's phase B kwargs for drop-in replacement
    gamma: float = 0.95,  # ignored by RGC, kept for compatibility
    global_step: int = 0,
    dump_grouping: bool = False,
    dump_dir: str = "",
    dump_max_queries: int = 0,
    dump_step_text_max_chars: int = 0,
    omega: float = 0.5,
    rank: int = 0,
) -> Dict[str, Any]:
    """RGC Phase B with a signature compatible with GiG's
    `compute_process_reward_phase_b`. Safe drop-in replacement.

    `gamma` is accepted but unused (RGC does not use discounted returns).
    All length-penalty configuration is read from `config`.
    """
    config = config or {}

    all_steps_per_sample = phase_a_result.get("all_steps_per_sample", []) or []
    corrected_state_vectors_per_sample = phase_a_result.get(
        "corrected_state_vectors_per_sample", []
    ) or []
    batch_size = phase_a_result.get("batch_size", len(all_steps_per_sample))

    if batch_size == 0:
        return {
            "step_advantages_per_sample": [],
            "process_reward_metrics": {},
            "group_details": [],
        }

    step_data_per_sample, metrics, group_details = compute_rgc_step_advantages(
        all_steps_per_sample=all_steps_per_sample,
        corrected_state_vectors_per_sample=corrected_state_vectors_per_sample,
        outcome_rewards=outcome_rewards,
        uid_list=uid_list,
        length_penalty_mode=str(config.get("length_penalty_mode", "group_zscore")),
        length_penalty_alpha=float(config.get("length_penalty_alpha", 0.3)),
        length_penalty_target=int(config.get("length_penalty_target", 8)),
        length_min_group_size=int(config.get("length_min_group_size", 2)),
        length_excl_correct_only=bool(config.get("length_excl_correct_only", False)),
        efficiency_penalty_alpha=float(config.get("efficiency_penalty_alpha", 0.0)),
    )

    # Dump grouping details (optional)
    if dump_grouping and group_details:
        try:
            dump_grouping_details(
                group_details=group_details,
                step_data_per_sample=step_data_per_sample,
                questions=questions,
                rubrics_list=rubrics_list,
                outcome_rewards=outcome_rewards,
                uid_list=uid_list,
                global_step=global_step,
                rank=rank,
                dump_dir=dump_dir,
                dump_max_queries=dump_max_queries,
                gamma=gamma,
                batch_size=batch_size,
                dump_step_text_max_chars=dump_step_text_max_chars,
                omega=omega,
            )
        except Exception as e:
            logger.warning(f"[RGC] dump_grouping_details failed: {e}")

    return {
        "step_advantages_per_sample": step_data_per_sample,
        "process_reward_metrics": metrics,
        "group_details": group_details,
    }


# ============================================================
# Module 4: Full Batch Pipeline (standalone entry; not used in training path)
# ============================================================


def compute_rgc_for_batch(
    sequences_strs: List[str],
    outcome_rewards: List[float],
    uid_list: List[str],
    rubrics_list: List[List[str]],
    query_list: List[str],
    config: Dict[str, Any],
    response_token_lengths: Optional[List[int]] = None,
    global_step: int = 0,
) -> Dict[str, Any]:
    """Run Phase A + Phase B end-to-end. Useful for offline tests.

    Note: In the training path, `api_prime_dapo_trajectory_v4_rgc`
    calls `compute_rgc_phase_b` directly after Phase A.
    """
    _start = time.time()
    n_trajs = len(sequences_strs)

    # Phase A
    logger.info(f"[RGC] Phase A on {n_trajs} trajectories...")
    _pa = time.time()
    try:
        phase_a_result = compute_process_reward_phase_a(
            trajectories=sequences_strs,
            questions=query_list,
            rubrics_list=rubrics_list,
            config=config,
        )
    except Exception as e:
        logger.error(f"[RGC] Phase A failed: {e}\n{traceback.format_exc()}")
        # Degenerate fallback
        phase_a_result = {
            "all_steps_per_sample": [extract_steps_from_trajectory(t) for t in sequences_strs],
            "corrected_state_vectors_per_sample": [[[0]]] * n_trajs,
            "batch_size": n_trajs,
        }
    _pa_time = time.time() - _pa

    # Phase B
    _pb = time.time()
    phase_b_result = compute_rgc_phase_b(
        phase_a_result=phase_a_result,
        trajectories=sequences_strs,
        questions=query_list,
        rubrics_list=rubrics_list,
        outcome_rewards=outcome_rewards,
        uid_list=uid_list,
        config=config,
        global_step=global_step,
    )
    _pb_time = time.time() - _pb

    # Optional token mapping
    token_advantages = None
    if response_token_lengths is not None:
        try:
            response_length = max(response_token_lengths)
            step_data_per_sample = phase_b_result["step_advantages_per_sample"]
            step_data_per_sample = map_step_advantages_to_tokens(
                step_data_per_sample, sequences_strs, response_length
            )
            phase_b_result["step_advantages_per_sample"] = step_data_per_sample
        except Exception as e:
            logger.error(f"[RGC] token mapping failed: {e}")

    return {
        "phase_a_result": phase_a_result,
        "phase_b_result": phase_b_result,
        "step_advantages_per_sample": phase_b_result["step_advantages_per_sample"],
        "metrics": phase_b_result["process_reward_metrics"],
        "group_details": phase_b_result["group_details"],
        "phase_a_time_s": _pa_time,
        "phase_b_time_s": _pb_time,
        "total_time_s": time.time() - _start,
    }


# ============================================================
# Module 5: V6 RGC-GiGPO Step-Level Advantage Computation
# ============================================================
# V6 keeps RGC's rubric-state grouping (group_key = (uid, rubric_state))
# but REPLACES the step value with GiGPO's discounted return:
#
#     G_t = gamma^(T-1-t) * R_final
#
# where T is the number of non-final steps in that trajectory and t is the
# step index within that trajectory. This matches gigpo Eq.5 and
# `core_gigpo.compute_step_discounted_returns`.
#
# The downstream fusion in ray_trainer is forced to "gigpo_additive":
#
#     A_final = A_E + step_advantage_w * A_S         (gigpo Eq.8)
#
# so that, numerically, the A_S produced here plays the exact same role as
# gigpo's step_advantages: a per-group-z-normed discounted return, added
# directly on top of the GRPO per-query advantage, with no extra z-norm
# on top.
# ============================================================


def compute_rgc_gigpo_step_advantages(
    all_steps_per_sample: List[List[Dict[str, Any]]],
    corrected_state_vectors_per_sample: List[List[List[int]]],
    outcome_rewards: List[float],
    uid_list: List[str],
    *,
    gamma: float = 0.95,
    epsilon: float = 1e-6,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any], List[Dict[str, Any]]]:
    """RGC-GiGPO (V6) step-level advantage.

    Semantics (BYTE-LEVEL ALIGNED with gigpo reference, `core_gigpo.py`):
      - Grouping: (uid, pre-action rubric_state) — RGC grouping kept.
      - Value:    G_t = gamma^(T - 1 - t) * R_final    [gigpo Eq.5]
                  where T is the TOTAL number of parsed steps in the
                  trajectory, INCLUDING the final `<answer>` step, and
                  t = step_index ∈ [0, T-1].
      - Final step participates in its rubric group just like any other
        step (matches gigpo's view that the final action is a regular
        env step where r_T = outcome).
      - Within-group z-norm: (v - mean) / (std + eps), std uses ddof=1
        (unbiased, equivalent to torch.std default). Matches gigpo's
        step_norm_reward exactly.
      - Singleton or zero-variance group -> A_S = 0 (same as gigpo).

    Output is structurally identical to `compute_rgc_step_advantages` so
    downstream char/token mapping, dump, and non_tensor_batch hand-off
    remain unchanged.
    """
    batch_size = len(all_steps_per_sample)
    assert len(corrected_state_vectors_per_sample) == batch_size
    assert len(outcome_rewards) == batch_size
    assert len(uid_list) == batch_size

    gamma = float(gamma)
    if gamma < 0.0 or gamma > 1.0:
        logger.warning(
            f"[RGC-GiGPO] gamma={gamma} out of [0,1]; clipping for safety."
        )
        gamma = max(0.0, min(1.0, gamma))

    # ---------- Build per-step records ----------
    all_step_data: List[Dict[str, Any]] = []
    step_data_per_sample: List[List[Dict[str, Any]]] = []

    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        corrected_svs = corrected_state_vectors_per_sample[i]
        outcome_r = float(outcome_rewards[i]) if i < len(outcome_rewards) else 0.0

        sample_records: List[Dict[str, Any]] = []
        num_rubrics = len(corrected_svs[0]) if corrected_svs and corrected_svs[0] else 0

        for j, step in enumerate(steps):
            pre_action_state = (
                [0] * num_rubrics if j == 0
                else (corrected_svs[j - 1] if (j - 1) < len(corrected_svs) else [0] * num_rubrics)
            )
            post_action_state = (
                corrected_svs[j] if j < len(corrected_svs) else pre_action_state
            )

            step_text_parts: List[str] = []
            if step.get("think"):
                step_text_parts.append(f"<think>{step['think']}</think>")
            if step.get("action"):
                step_text_parts.append(step["action"])
            if step.get("response"):
                step_text_parts.append(step["response"])
            step_text = "\n".join(step_text_parts)

            if j == 0:
                step_input = ""
            else:
                prev_step = steps[j - 1]
                step_input = prev_step.get("formatted_cumulative_text", "")

            record = {
                "trajectory_index": i,
                "step_index": step.get("step_index", j),
                "corrected_state": pre_action_state,
                "post_action_state": post_action_state,
                "state_transition": "rgc_gigpo",
                "immediate_reward": 0.0,
                "is_final": bool(step.get("is_final", j == len(steps) - 1)),
                "uid": uid_list[i] if i < len(uid_list) else "",
                "step_text": step_text,
                "step_input": step_input,
                # filled later
                "normalized_reward": 0.0,
                "step_advantage": 0.0,
                "group_id": None,
                "rgc_value": outcome_r,     # will be overwritten with G_t
                "rgc_length_offset": 0.0,   # kept 0 for schema parity
                "rgc_traj_length": 0,
                # diagnostic
                "rgc_discount_exp": 0,
                "rgc_gigpo_return": 0.0,
                # outcome bookkeeping
                "_outcome": outcome_r,
            }
            all_step_data.append(record)
            sample_records.append(record)

        step_data_per_sample.append(sample_records)

    # ---------- Trajectory length (GiGPO T: INCLUDES final step) ----------
    # gigpo Eq.5:  G_t = gamma^(T-1-t) * R_final, where T = total parsed steps
    # in the trajectory (the final <answer> step counts as step T-1).
    # We therefore use len(all_steps_per_sample[i]) rather than the count
    # of non-final steps.
    traj_length_by_idx: Dict[int, int] = {}
    for i in range(batch_size):
        traj_length_by_idx[i] = len(all_steps_per_sample[i])
    for it in all_step_data:
        it["rgc_traj_length"] = traj_length_by_idx.get(it["trajectory_index"], 0)

    # Keep the non_final / final partitions as diagnostic pools for logging.
    non_final_items = [it for it in all_step_data if not it["is_final"]]
    final_items = [it for it in all_step_data if it["is_final"]]

    # ---------- Group by (uid, pre-action state) — ALL steps, incl. final ----
    # This is the V6's key deviation from V4 (V4 excludes final steps from
    # the rubric group). gigpo's step_norm_reward applies uniformly to every
    # step, so V6 mirrors that by including final steps here.
    groups: Dict[Tuple[str, Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for it in all_step_data:
        key = (it.get("uid", ""), tuple(it["corrected_state"]))
        groups[key].append(it)

    # ---------- Within-group advantage computation ----------
    group_details: List[Dict[str, Any]] = []
    effective_groups = 0
    singleton_groups = 0
    total_steps_with_signal = 0

    # stats for logging
    joint_value_pool: List[float] = []
    adv_pool: List[float] = []
    discount_exp_pool: List[int] = []

    # Distribution counters (mirror V4 for dashboard parity)
    group_size_hist = {
        "1": 0, "2": 0, "3_4": 0, "5_8": 0, "9_16": 0, "17plus": 0,
    }
    group_size_list: List[int] = []

    outcome_all_correct = 0
    outcome_all_wrong = 0
    outcome_mixed = 0
    mixed_group_steps = 0

    for gid_idx, (group_key, members) in enumerate(groups.items()):
        gid = f"g{gid_idx}"
        uid, state_tuple = group_key
        group_size = len(members)

        outcomes = [float(m["_outcome"]) for m in members]
        lengths = [int(m["rgc_traj_length"]) for m in members]
        L_mean, L_std = _safe_mean_std(lengths)

        # Bin A: group-size histogram
        group_size_list.append(group_size)
        if group_size == 1:
            group_size_hist["1"] += 1
        elif group_size == 2:
            group_size_hist["2"] += 1
        elif group_size <= 4:
            group_size_hist["3_4"] += 1
        elif group_size <= 8:
            group_size_hist["5_8"] += 1
        elif group_size <= 16:
            group_size_hist["9_16"] += 1
        else:
            group_size_hist["17plus"] += 1

        # Bin B: outcome diversity
        n_correct = sum(1 for o in outcomes if o > 0.5)
        n_wrong = group_size - n_correct
        if group_size >= 2:
            if n_correct == group_size:
                outcome_all_correct += 1
            elif n_wrong == group_size:
                outcome_all_wrong += 1
            else:
                outcome_mixed += 1
                mixed_group_steps += group_size

        # -------- Compute GiGPO discounted returns as step values --------
        # gigpo Eq.5:  G_t = gamma^(T-1-t) * R_final
        # where T = len(parsed steps, incl. final), t = step_index ∈ [0, T-1].
        joint_values: List[float] = []
        for m in members:
            R_i = float(m["_outcome"])
            t_i = int(m["step_index"])
            T_i = int(m["rgc_traj_length"])
            # Guard: if parsing produced an inconsistency, fall back to t_i+1
            # so that discount_exp never goes negative.
            T_effective = max(T_i, t_i + 1, 1)
            discount_exp = max(0, T_effective - 1 - t_i)
            v_i = (gamma ** discount_exp) * R_i

            m["rgc_value"] = v_i
            m["rgc_length_offset"] = 0.0           # not used in V6, keep 0
            m["rgc_discount_exp"] = discount_exp
            m["rgc_gigpo_return"] = v_i
            joint_values.append(v_i)
            discount_exp_pool.append(discount_exp)

        # Use UNBIASED std (ddof=1) to exactly match gigpo's torch.std default
        # (core_gigpo.step_norm_reward). This is the V5 / reference semantics.
        v_mean, v_std = _safe_mean_std_unbiased(joint_values)

        is_effective = group_size > 1 and v_std > epsilon
        if not is_effective:
            if group_size == 1:
                singleton_groups += 1
            for m in members:
                m["normalized_reward"] = 0.0
                m["step_advantage"] = 0.0
                m["group_id"] = gid
        else:
            effective_groups += 1
            for m, v_i in zip(members, joint_values):
                norm = (v_i - v_mean) / (v_std + epsilon)
                m["normalized_reward"] = norm
                m["step_advantage"] = norm
                m["group_id"] = gid
                total_steps_with_signal += 1
                adv_pool.append(norm)

        joint_value_pool.extend(joint_values)

        # group detail dump (schema compatible with V4 for offline tooling)
        reward_dist: Dict[str, int] = {}
        for o in outcomes:
            k = f"{float(o):.1f}"
            reward_dist[k] = reward_dist.get(k, 0) + 1
        state_list = list(state_tuple)
        num_rubrics = len(state_list)
        satisfied_count = sum(state_list)
        satisfied_indices = [f"C{i + 1}" for i, v in enumerate(state_list) if v == 1]
        if satisfied_indices:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied ({', '.join(satisfied_indices)})"
        else:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied"

        group_details.append({
            "group_id": gid,
            "uid": uid,
            "pre_action_state": state_list,
            "pre_action_state_summary": state_summary,
            "group_size": group_size,
            "reward_mean": float(np.mean(outcomes)) if outcomes else 0.0,
            "reward_std": float(np.std(outcomes)) if outcomes else 0.0,
            "length_mean": L_mean,
            "length_std": L_std,
            "length_penalty_active": False,         # V6 never uses length penalty
            "joint_value_mean": v_mean,
            "joint_value_std": v_std,
            "is_effective": is_effective,
            "reward_distribution": reward_dist,
            "value_function_mode": "gigpo_return",
            "gamma": gamma,
            "members": [
                {
                    "trajectory_index": m.get("trajectory_index", -1),
                    "step_index": m["step_index"],
                    "outcome_reward": float(m["_outcome"]),
                    "traj_length": int(m["rgc_traj_length"]),
                    "discount_exp": int(m.get("rgc_discount_exp", 0)),
                    "joint_value": float(m.get("rgc_value", m["_outcome"])),
                    "step_advantage": float(m.get("step_advantage", 0.0)),
                    "is_final": bool(m.get("is_final", False)),
                }
                for m in members
            ],
        })

    # NOTE: final steps are NO LONGER post-processed here - they already went
    # through the same grouping + z-norm loop above. This matches gigpo's
    # core_gigpo.compute_gigpo_outcome_advantage which applies step_norm to
    # every step uniformly, including the last one.
    _ = final_items  # kept for diagnostic exposure; see metrics below.

    # ---------- Char boundaries for token mapping ----------
    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        sample_records = step_data_per_sample[i]
        for j, (step, sd) in enumerate(zip(steps, sample_records)):
            cum_len = len(step.get("cumulative_text", ""))
            if j == 0:
                sd["char_start"] = 0
            else:
                prev_cum_len = len(steps[j - 1].get("cumulative_text", ""))
                sd["char_start"] = prev_cum_len
            sd["char_end"] = cum_len

    # ---------- Strip internal fields ----------
    for it in all_step_data:
        it.pop("_outcome", None)

    # ---------- Metrics ----------
    non_zero_adv = [a for a in adv_pool if abs(a) > epsilon]
    total_groups_n = max(1, len(groups))

    metrics: Dict[str, Any] = {
        # ------------ V6-specific metrics (primary) ------------
        "rgc_gigpo/gamma": gamma,
        "rgc_gigpo/num_state_groups": len(groups),
        "rgc_gigpo/effective_groups": effective_groups,
        "rgc_gigpo/singleton_groups": singleton_groups,
        "rgc_gigpo/num_non_final_steps": len(non_final_items),
        "rgc_gigpo/num_final_steps": len(final_items),
        "rgc_gigpo/num_total_steps": len(all_step_data),
        "rgc_gigpo/signal_ratio": (
            total_steps_with_signal / max(1, len(non_final_items))
        ),
        "rgc_gigpo/mean_step_advantage": (
            float(np.mean(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc_gigpo/std_step_advantage": (
            float(np.std(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc_gigpo/mean_gigpo_return": (
            float(np.mean(joint_value_pool)) if joint_value_pool else 0.0
        ),
        "rgc_gigpo/avg_discount_exp": (
            float(np.mean(discount_exp_pool)) if discount_exp_pool else 0.0
        ),
        "rgc_gigpo/max_discount_exp": (
            int(np.max(discount_exp_pool)) if discount_exp_pool else 0
        ),
        # ------------ RGC-compatible metrics (alias, for V4 dashboards) ------------
        "rgc/num_state_groups": len(groups),
        "rgc/effective_groups": effective_groups,
        "rgc/singleton_groups": singleton_groups,
        "rgc/length_active_groups": 0,             # V6 disables length penalty
        "rgc/num_non_final_steps": len(non_final_items),
        "rgc/num_final_steps": len(final_items),
        "rgc/num_total_steps": len(all_step_data),
        "rgc/non_zero_advantage_count": len(non_zero_adv),
        "rgc/mean_step_advantage": (
            float(np.mean(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc/std_step_advantage": (
            float(np.std(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc/mean_joint_value": (
            float(np.mean(joint_value_pool)) if joint_value_pool else 0.0
        ),
        "rgc/mean_length_offset_abs": 0.0,         # V6: length offset disabled
        "rgc/length_penalty_mode": "gigpo_return",
        "rgc/length_penalty_alpha": 0.0,
        "rgc/signal_ratio": (
            total_steps_with_signal / max(1, len(non_final_items))
        ),
        # ------------ Group size/outcome distribution (copy of V4 style) ------------
        "rgc_dist/group_size_bin_1":       group_size_hist["1"],
        "rgc_dist/group_size_bin_2":       group_size_hist["2"],
        "rgc_dist/group_size_bin_3_4":     group_size_hist["3_4"],
        "rgc_dist/group_size_bin_5_8":     group_size_hist["5_8"],
        "rgc_dist/group_size_bin_9_16":    group_size_hist["9_16"],
        "rgc_dist/group_size_bin_17plus":  group_size_hist["17plus"],
        "rgc_dist/groups_all_correct":     outcome_all_correct,
        "rgc_dist/groups_all_wrong":       outcome_all_wrong,
        "rgc_dist/groups_mixed":           outcome_mixed,
        "rgc_dist/mixed_group_ratio":      outcome_mixed / total_groups_n,
        "rgc_dist/outcome_diverse_steps":  mixed_group_steps,
        "rgc_dist/singleton_ratio":        group_size_hist["1"] / total_groups_n,
        "rgc_dist/effective_ratio":        effective_groups / total_groups_n,
        "rgc_dist/length_active_ratio":    0.0,
        # ------------ GiG-compatible keys (for dashboards/back-compat) ------------
        "gig/num_state_groups": len(groups),
        "gig/avg_group_size": float(np.mean([len(v) for v in groups.values()])) if groups else 0.0,
        "gig/max_group_size": int(max([len(v) for v in groups.values()], default=0)),
        "gig/num_non_final_steps": len(non_final_items),
        "gig/num_final_steps": len(final_items),
        "gig/num_total_steps": len(all_step_data),
    }

    logger.info(
        f"[RGC-GiGPO Phase B] gamma={gamma} groups={len(groups)} "
        f"effective={effective_groups} singletons={singleton_groups} "
        f"signal_ratio={metrics['rgc_gigpo/signal_ratio']:.3f} "
        f"avg_discount_exp={metrics['rgc_gigpo/avg_discount_exp']:.2f} "
        f"mean|A_S|={metrics['rgc_gigpo/mean_step_advantage']:.3f}"
    )

    return step_data_per_sample, metrics, group_details


# ============================================================
# Module 6: V6 Phase B wrapper (GiG/RGC-compatible signature)
# ============================================================


def compute_rgc_gigpo_phase_b(
    phase_a_result: Dict[str, Any],
    trajectories: List[str],
    questions: List[str],
    rubrics_list: List[List[str]],
    outcome_rewards: List[float],
    uid_list: List[str],
    config: Optional[Dict[str, Any]] = None,
    *,
    gamma: float = 0.95,
    global_step: int = 0,
    dump_grouping: bool = False,
    dump_dir: str = "",
    dump_max_queries: int = 0,
    dump_step_text_max_chars: int = 0,
    omega: float = 1.0,
    rank: int = 0,
) -> Dict[str, Any]:
    """V6 RGC-GiGPO Phase B.

    Signature mirrors `compute_rgc_phase_b` / GiG's `compute_process_reward_phase_b`
    so that the V4/V6 reward manager can plug it in as a drop-in replacement.

    Grouping:  (uid, rubric_state)          [from RGC]
    Value:     G_t = gamma^(T-1-t) * R_final [from GiGPO Eq.5]
    Norm:      within-group z-score         [from RGC]
    Fusion:    A = A_E + omega * A_S        [gigpo_additive, enforced upstream]

    The `gamma` kwarg here is the REAL parameter (used for value computation),
    unlike in `compute_rgc_phase_b` where it is ignored. Config may override
    it via `rgc_gigpo_gamma`.
    """
    config = config or {}

    # Allow per-batch override through config, falling back to kwarg default.
    effective_gamma = float(config.get("rgc_gigpo_gamma", gamma))

    all_steps_per_sample = phase_a_result.get("all_steps_per_sample", []) or []
    corrected_state_vectors_per_sample = phase_a_result.get(
        "corrected_state_vectors_per_sample", []
    ) or []
    batch_size = phase_a_result.get("batch_size", len(all_steps_per_sample))

    if batch_size == 0:
        return {
            "step_advantages_per_sample": [],
            "process_reward_metrics": {},
            "group_details": [],
        }

    step_data_per_sample, metrics, group_details = compute_rgc_gigpo_step_advantages(
        all_steps_per_sample=all_steps_per_sample,
        corrected_state_vectors_per_sample=corrected_state_vectors_per_sample,
        outcome_rewards=outcome_rewards,
        uid_list=uid_list,
        gamma=effective_gamma,
    )

    # Optional dump: reuse the RGC/GiG dump helper (schema is compatible).
    if dump_grouping and group_details:
        try:
            dump_grouping_details(
                group_details=group_details,
                step_data_per_sample=step_data_per_sample,
                questions=questions,
                rubrics_list=rubrics_list,
                outcome_rewards=outcome_rewards,
                uid_list=uid_list,
                global_step=global_step,
                rank=rank,
                dump_dir=dump_dir,
                dump_max_queries=dump_max_queries,
                gamma=effective_gamma,
                batch_size=batch_size,
                dump_step_text_max_chars=dump_step_text_max_chars,
                omega=omega,
            )
        except Exception as e:
            logger.warning(f"[RGC-GiGPO] dump_grouping_details failed: {e}")

    return {
        "step_advantages_per_sample": step_data_per_sample,
        "process_reward_metrics": metrics,
        "group_details": group_details,
    }


# ============================================================
# Module 7: V7 RGC-IR Step-Level Advantage Computation
# ============================================================
# V7 keeps RGC's rubric-state grouping (group_key = (uid, pre-action state))
# but REPLACES the step value with the **immediate rubric increment**:
#
#     r_t = ||s_t||_1 - ||s_{t-1}||_1        (non-final step t)
#
# Final step:
#     * NOT grouped;
#     * NOT assigned a process reward;
#     * A_S hard-pinned to 0  (outcome signal handled by A_E upstream).
#
# Within-group z-norm: (r_t - mean) / (std + eps), std uses ddof=1.
# Downstream fusion in ray_trainer is advertised as "convex_final_exempt":
#
#     non-final tokens:  A_final = (1 - ω) · Â_E + ω · Â_S
#     final    tokens:  A_final = A_E            (raw, unattenuated)
#
# Semantics summary:
#     * process reward = "how many new rubrics did THIS step unlock";
#     * telescoping sum: Σ_t r_t = ||s_{T-2}||_1 (trajectory's final
#       rubric coverage, independent of step count) → implicit short-path
#       preference;
#     * final step is a separate action domain (writing the answer),
#       handled exclusively by A_E so that answer-token gradient is not
#       attenuated and outcome is not double-counted.
# ============================================================


def compute_rgc_ir_step_advantages(
    all_steps_per_sample: List[List[Dict[str, Any]]],
    corrected_state_vectors_per_sample: List[List[List[int]]],
    outcome_rewards: List[float],
    uid_list: List[str],
    *,
    epsilon: float = 1e-8,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any], List[Dict[str, Any]]]:
    """RGC-IR (V7) step-level advantage.

    Produces a `step_data_per_sample` structure structurally identical to
    V4/V6 so that the rest of the pipeline (char/token mapping, dump,
    non_tensor_batch hand-off) is unchanged.
    """
    batch_size = len(all_steps_per_sample)
    assert len(corrected_state_vectors_per_sample) == batch_size
    assert len(outcome_rewards) == batch_size
    assert len(uid_list) == batch_size

    # ---------- Build per-step records ----------
    all_step_data: List[Dict[str, Any]] = []
    step_data_per_sample: List[List[Dict[str, Any]]] = []

    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        corrected_svs = corrected_state_vectors_per_sample[i]
        outcome_r = float(outcome_rewards[i]) if i < len(outcome_rewards) else 0.0

        sample_records: List[Dict[str, Any]] = []
        num_rubrics = len(corrected_svs[0]) if corrected_svs and corrected_svs[0] else 0

        for j, step in enumerate(steps):
            # pre-action state:
            #   step 0    -> all-zero virtual state (s_{-1} := 0)
            #   step t>0  -> post-action state of step t-1
            pre_action_state = (
                [0] * num_rubrics if j == 0
                else (corrected_svs[j - 1] if (j - 1) < len(corrected_svs) else [0] * num_rubrics)
            )
            post_action_state = (
                corrected_svs[j] if j < len(corrected_svs) else pre_action_state
            )

            # compact step_text for dump
            step_text_parts: List[str] = []
            if step.get("think"):
                step_text_parts.append(f"<think>{step['think']}</think>")
            if step.get("action"):
                step_text_parts.append(step["action"])
            if step.get("response"):
                step_text_parts.append(step["response"])
            step_text = "\n".join(step_text_parts)

            if j == 0:
                step_input = ""
            else:
                prev_step = steps[j - 1]
                step_input = prev_step.get("formatted_cumulative_text", "")

            is_final = bool(step.get("is_final", j == len(steps) - 1))

            # Immediate rubric increment r_t = ||s_t||_1 - ||s_{t-1}||_1.
            # For the final step we set r_t = 0.0: bookkeeping only, the
            # final step never participates in group z-score and its
            # step_advantage is hard-pinned to 0 below.
            if is_final:
                r_t = 0.0
            else:
                r_t = float(sum(post_action_state) - sum(pre_action_state))
                # Monotonicity-corrected states should already ensure r_t>=0,
                # but keep a safety clamp against degenerate parses.
                if r_t < 0.0:
                    r_t = 0.0

            record = {
                "trajectory_index": i,
                "step_index": step.get("step_index", j),
                "corrected_state": pre_action_state,
                "post_action_state": post_action_state,
                "state_transition": "rgc_ir",
                "immediate_reward": r_t,          # V7's step value (non-final)
                "is_final": is_final,
                "uid": uid_list[i] if i < len(uid_list) else "",
                "step_text": step_text,
                "step_input": step_input,
                # filled later
                "normalized_reward": 0.0,
                "step_advantage": 0.0,
                "group_id": None,
                "rgc_value": r_t,                 # alias for dashboard parity
                "rgc_length_offset": 0.0,         # not used in V7
                "rgc_traj_length": 0,
                # outcome bookkeeping
                "_outcome": outcome_r,
            }
            all_step_data.append(record)
            sample_records.append(record)

        step_data_per_sample.append(sample_records)

    # ---------- Trajectory length (non-final step count) ----------
    traj_length_by_idx: Dict[int, int] = defaultdict(int)
    for it in all_step_data:
        if not it["is_final"]:
            traj_length_by_idx[it["trajectory_index"]] += 1
    for it in all_step_data:
        it["rgc_traj_length"] = traj_length_by_idx.get(it["trajectory_index"], 0)

    # ---------- Split final vs non-final ----------
    non_final_items = [it for it in all_step_data if not it["is_final"]]
    final_items = [it for it in all_step_data if it["is_final"]]

    # ---------- Group NON-FINAL steps by (uid, pre-action state) ----------
    groups: Dict[Tuple[str, Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for it in non_final_items:
        key = (it.get("uid", ""), tuple(it["corrected_state"]))
        groups[key].append(it)

    # ---------- Within-group advantage computation ----------
    group_details: List[Dict[str, Any]] = []
    effective_groups = 0
    singleton_groups = 0
    total_steps_with_signal = 0

    joint_value_pool: List[float] = []
    adv_pool: List[float] = []
    r_t_pool: List[float] = []

    # Distribution counters (mirror V4/V6 for dashboard parity)
    group_size_hist = {
        "1": 0, "2": 0, "3_4": 0, "5_8": 0, "9_16": 0, "17plus": 0,
    }
    group_size_list: List[int] = []

    outcome_all_correct = 0
    outcome_all_wrong = 0
    outcome_mixed = 0
    mixed_group_steps = 0

    for gid_idx, (group_key, members) in enumerate(groups.items()):
        gid = f"g{gid_idx}"
        uid, state_tuple = group_key
        group_size = len(members)

        outcomes = [float(m["_outcome"]) for m in members]
        lengths = [int(m["rgc_traj_length"]) for m in members]
        L_mean, L_std = _safe_mean_std(lengths)

        # Bin A: group-size histogram
        group_size_list.append(group_size)
        if group_size == 1:
            group_size_hist["1"] += 1
        elif group_size == 2:
            group_size_hist["2"] += 1
        elif group_size <= 4:
            group_size_hist["3_4"] += 1
        elif group_size <= 8:
            group_size_hist["5_8"] += 1
        elif group_size <= 16:
            group_size_hist["9_16"] += 1
        else:
            group_size_hist["17plus"] += 1

        # Bin B: outcome diversity
        n_correct = sum(1 for o in outcomes if o > 0.5)
        n_wrong = group_size - n_correct
        if group_size >= 2:
            if n_correct == group_size:
                outcome_all_correct += 1
            elif n_wrong == group_size:
                outcome_all_wrong += 1
            else:
                outcome_mixed += 1
                mixed_group_steps += group_size

        # --- step values = immediate rubric increments r_t ---
        values: List[float] = [float(m["immediate_reward"]) for m in members]
        r_t_pool.extend(values)

        # unbiased std (ddof=1) so the z-score matches torch.std default
        v_mean, v_std = _safe_mean_std_unbiased(values)

        is_effective = group_size > 1 and v_std > epsilon
        if not is_effective:
            if group_size == 1:
                singleton_groups += 1
            for m in members:
                m["normalized_reward"] = 0.0
                m["step_advantage"] = 0.0
                m["group_id"] = gid
        else:
            effective_groups += 1
            for m, v_i in zip(members, values):
                norm = (v_i - v_mean) / (v_std + epsilon)
                m["normalized_reward"] = norm
                m["step_advantage"] = norm
                m["group_id"] = gid
                total_steps_with_signal += 1
                adv_pool.append(norm)

        joint_value_pool.extend(values)

        # group detail dump (schema compatible with V4/V6)
        reward_dist: Dict[str, int] = {}
        for o in outcomes:
            k = f"{float(o):.1f}"
            reward_dist[k] = reward_dist.get(k, 0) + 1
        state_list = list(state_tuple)
        num_rubrics = len(state_list)
        satisfied_count = sum(state_list)
        satisfied_indices = [f"C{i + 1}" for i, v in enumerate(state_list) if v == 1]
        if satisfied_indices:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied ({', '.join(satisfied_indices)})"
        else:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied"

        group_details.append({
            "group_id": gid,
            "uid": uid,
            "pre_action_state": state_list,
            "pre_action_state_summary": state_summary,
            "group_size": group_size,
            "reward_mean": float(np.mean(outcomes)) if outcomes else 0.0,
            "reward_std": float(np.std(outcomes)) if outcomes else 0.0,
            "length_mean": L_mean,
            "length_std": L_std,
            "length_penalty_active": False,          # V7 never uses length penalty
            "joint_value_mean": v_mean,
            "joint_value_std": v_std,
            "is_effective": is_effective,
            "reward_distribution": reward_dist,
            "value_function_mode": "rgc_immediate_reward",
            "members": [
                {
                    "trajectory_index": m.get("trajectory_index", -1),
                    "step_index": m["step_index"],
                    "outcome_reward": float(m["_outcome"]),
                    "traj_length": int(m["rgc_traj_length"]),
                    "immediate_reward": float(m["immediate_reward"]),
                    "joint_value": float(m["immediate_reward"]),
                    "step_advantage": float(m.get("step_advantage", 0.0)),
                    "is_final": bool(m.get("is_final", False)),
                }
                for m in members
            ],
        })

    # ---------- Final steps: hard-pin A_S = 0 ----------
    # The answer span's gradient is carried entirely by A_E upstream.
    # The `convex_final_exempt` fusion mode in ray_trainer.py will route
    # these tokens to raw A_E based on `is_final == True`.
    for it in final_items:
        it["normalized_reward"] = 0.0
        it["step_advantage"] = 0.0
        it["group_id"] = None

    # ---------- Char boundaries for token mapping ----------
    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        sample_records = step_data_per_sample[i]
        for j, (step, sd) in enumerate(zip(steps, sample_records)):
            cum_len = len(step.get("cumulative_text", ""))
            if j == 0:
                sd["char_start"] = 0
            else:
                prev_cum_len = len(steps[j - 1].get("cumulative_text", ""))
                sd["char_start"] = prev_cum_len
            sd["char_end"] = cum_len

    # ---------- Strip internal fields ----------
    for it in all_step_data:
        it.pop("_outcome", None)

    # ---------- Metrics ----------
    non_zero_adv = [a for a in adv_pool if abs(a) > epsilon]
    total_groups_n = max(1, len(groups))
    n_nonfinal = max(1, len(non_final_items))

    zero_increment_count = sum(1 for r in r_t_pool if r <= epsilon)
    # "traj_total_increment" = per-trajectory sum of r_t (= final rubric coverage).
    traj_sum_r: Dict[int, float] = defaultdict(float)
    for it in non_final_items:
        traj_sum_r[it["trajectory_index"]] += float(it["immediate_reward"])
    traj_total_increment_mean = (
        float(np.mean(list(traj_sum_r.values()))) if traj_sum_r else 0.0
    )

    metrics: Dict[str, Any] = {
        # ------------ V7-specific metrics (primary) ------------
        "rgc_ir/num_state_groups": len(groups),
        "rgc_ir/effective_groups": effective_groups,
        "rgc_ir/singleton_groups": singleton_groups,
        "rgc_ir/avg_group_size": (
            float(np.mean(group_size_list)) if group_size_list else 0.0
        ),
        "rgc_ir/max_group_size": (
            int(np.max(group_size_list)) if group_size_list else 0
        ),
        "rgc_ir/num_non_final_steps": len(non_final_items),
        "rgc_ir/num_final_steps": len(final_items),
        "rgc_ir/num_total_steps": len(all_step_data),
        "rgc_ir/mean_immediate_reward": (
            float(np.mean(r_t_pool)) if r_t_pool else 0.0
        ),
        "rgc_ir/std_immediate_reward": (
            float(np.std(r_t_pool)) if r_t_pool else 0.0
        ),
        "rgc_ir/zero_increment_ratio": (
            float(zero_increment_count) / n_nonfinal
        ),
        "rgc_ir/mean_step_advantage": (
            float(np.mean(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc_ir/std_step_advantage": (
            float(np.std(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc_ir/traj_total_increment_mean": traj_total_increment_mean,
        "rgc_ir/final_excluded_count": len(final_items),
        "rgc_ir/signal_ratio": (
            total_steps_with_signal / n_nonfinal
        ),
        # ------------ RGC-compatible metrics (alias, for V4 dashboards) ------------
        "rgc/num_state_groups": len(groups),
        "rgc/effective_groups": effective_groups,
        "rgc/singleton_groups": singleton_groups,
        "rgc/length_active_groups": 0,
        "rgc/num_non_final_steps": len(non_final_items),
        "rgc/num_final_steps": len(final_items),
        "rgc/num_total_steps": len(all_step_data),
        "rgc/non_zero_advantage_count": len(non_zero_adv),
        "rgc/mean_step_advantage": (
            float(np.mean(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc/std_step_advantage": (
            float(np.std(non_zero_adv)) if non_zero_adv else 0.0
        ),
        "rgc/mean_joint_value": (
            float(np.mean(joint_value_pool)) if joint_value_pool else 0.0
        ),
        "rgc/mean_length_offset_abs": 0.0,
        "rgc/length_penalty_mode": "rgc_immediate_reward",
        "rgc/length_penalty_alpha": 0.0,
        "rgc/signal_ratio": (
            total_steps_with_signal / n_nonfinal
        ),
        # ------------ Group size/outcome distribution (V4 style) ------------
        "rgc_dist/group_size_bin_1":       group_size_hist["1"],
        "rgc_dist/group_size_bin_2":       group_size_hist["2"],
        "rgc_dist/group_size_bin_3_4":     group_size_hist["3_4"],
        "rgc_dist/group_size_bin_5_8":     group_size_hist["5_8"],
        "rgc_dist/group_size_bin_9_16":    group_size_hist["9_16"],
        "rgc_dist/group_size_bin_17plus":  group_size_hist["17plus"],
        "rgc_dist/groups_all_correct":     outcome_all_correct,
        "rgc_dist/groups_all_wrong":       outcome_all_wrong,
        "rgc_dist/groups_mixed":           outcome_mixed,
        "rgc_dist/mixed_group_ratio":      outcome_mixed / total_groups_n,
        "rgc_dist/outcome_diverse_steps":  mixed_group_steps,
        "rgc_dist/singleton_ratio":        group_size_hist["1"] / total_groups_n,
        "rgc_dist/effective_ratio":        effective_groups / total_groups_n,
        "rgc_dist/length_active_ratio":    0.0,
        # ------------ GiG-compatible keys (dashboard parity) ------------
        "gig/num_state_groups": len(groups),
        "gig/avg_group_size": (
            float(np.mean([len(v) for v in groups.values()])) if groups else 0.0
        ),
        "gig/max_group_size": int(max([len(v) for v in groups.values()], default=0)),
        "gig/num_non_final_steps": len(non_final_items),
        "gig/num_final_steps": len(final_items),
        "gig/num_total_steps": len(all_step_data),
    }

    logger.info(
        f"[RGC-IR Phase B] groups={len(groups)} effective={effective_groups} "
        f"singletons={singleton_groups} "
        f"signal_ratio={metrics['rgc_ir/signal_ratio']:.3f} "
        f"mean_r_t={metrics['rgc_ir/mean_immediate_reward']:.2f} "
        f"zero_r_ratio={metrics['rgc_ir/zero_increment_ratio']:.2f} "
        f"mean|A_S|={metrics['rgc_ir/mean_step_advantage']:.3f} "
        f"final_excluded={len(final_items)}"
    )

    return step_data_per_sample, metrics, group_details


# ============================================================
# Module 8: V7 Phase B wrapper (GiG/RGC-compatible signature)
# ============================================================


def compute_rgc_ir_phase_b(
    phase_a_result: Dict[str, Any],
    trajectories: List[str],
    questions: List[str],
    rubrics_list: List[List[str]],
    outcome_rewards: List[float],
    uid_list: List[str],
    config: Optional[Dict[str, Any]] = None,
    *,
    # mirror GiG's phase B kwargs for drop-in replacement; gamma is accepted
    # but UNUSED (V7 does not discount).
    gamma: float = 0.0,
    global_step: int = 0,
    dump_grouping: bool = False,
    dump_dir: str = "",
    dump_max_queries: int = 0,
    dump_step_text_max_chars: int = 0,
    omega: float = 0.3,
    rank: int = 0,
) -> Dict[str, Any]:
    """V7 RGC-IR Phase B.

    Grouping:  (uid, pre-action rubric_state)   — inherited from RGC
    Value:     r_t = ||s_t||_1 - ||s_{t-1}||_1  (non-final steps only)
    Final:     NOT grouped, A_S = 0             (outcome carried by A_E)
    Norm:      within-group z-score (ddof=1)
    Fusion:    advertised upstream as "convex_final_exempt":
                 non-final tokens: (1-ω)·Â_E + ω·Â_S
                 final    tokens: A_E           (raw, unattenuated)
    """
    config = config or {}

    all_steps_per_sample = phase_a_result.get("all_steps_per_sample", []) or []
    corrected_state_vectors_per_sample = phase_a_result.get(
        "corrected_state_vectors_per_sample", []
    ) or []
    batch_size = phase_a_result.get("batch_size", len(all_steps_per_sample))

    if batch_size == 0:
        return {
            "step_advantages_per_sample": [],
            "process_reward_metrics": {},
            "group_details": [],
        }

    step_data_per_sample, metrics, group_details = compute_rgc_ir_step_advantages(
        all_steps_per_sample=all_steps_per_sample,
        corrected_state_vectors_per_sample=corrected_state_vectors_per_sample,
        outcome_rewards=outcome_rewards,
        uid_list=uid_list,
    )

    # Optional dump: reuse the RGC/GiG dump helper (schema is compatible).
    if dump_grouping and group_details:
        try:
            dump_grouping_details(
                group_details=group_details,
                step_data_per_sample=step_data_per_sample,
                questions=questions,
                rubrics_list=rubrics_list,
                outcome_rewards=outcome_rewards,
                uid_list=uid_list,
                global_step=global_step,
                rank=rank,
                dump_dir=dump_dir,
                dump_max_queries=dump_max_queries,
                gamma=0.0,            # V7 has no gamma; pass 0 for logging
                batch_size=batch_size,
                dump_step_text_max_chars=dump_step_text_max_chars,
                omega=omega,
            )
        except Exception as e:
            logger.warning(f"[RGC-IR] dump_grouping_details failed: {e}")

    return {
        "step_advantages_per_sample": step_data_per_sample,
        "process_reward_metrics": metrics,
        "group_details": group_details,
    }
