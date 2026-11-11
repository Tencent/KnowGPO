# -*- coding: utf-8 -*-
"""IGPO V8 Phase B: convert per-turn GT mean-log-probs into sparse
token-level rewards.

Inputs (from `igpo_gt_logprob.run_gt_logprob_replay`):
    - mean_logprobs: dict (sample_idx, turn_idx) -> float
    - plans:         list[GTReplayPlan] with assistant turn char spans

Inputs (from V8 reward manager):
    - r_final:       list[float]   end-to-end LLM-judge correctness
    - solution_strs: list[str]     decoded responses (for offset_mapping)
    - tokenizer:     HF tokenizer
    - response_len:  int           shape[-1] of the rollout responses

Output:
    - reward_tensor: (bs, response_len) float32 sparse tensor
        * outcome at the LAST valid response token   = R_final
        * IG at the LAST token of every non-final assistant turn
          = info_gain(turn) per `info_gain_type`
    - metrics: dict (igpo/* keys)

Notes
-----
This is the "IG writer" half of V8. It does NOT do z-score normalisation —
that happens later inside `core_algos.compute_igpo_outcome_advantage`,
which performs the GRPO joint/separate normalisation + turn-level reverse
discounting.

The sparse layout matches IGPO-main/verl/utils/reward_score/info_gain.py
exactly so that downstream advantage code is a faithful port.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from verl.utils.reward_score.igpo_gt_logprob import (
    DEFAULT_ASSISTANT_SEPARATOR,
    GTReplayPlan,
    char_span_to_token_span,
)


logger = logging.getLogger(__name__)


def _info_gain(
    cur_mean_lp: float,
    prev_mean_lp: float,
    info_gain_type: str,
) -> float:
    """Compute IG between two consecutive turns.

    Args:
        cur_mean_lp:  mean log-prob of GT at turn t
        prev_mean_lp: mean log-prob of GT at turn t-1
        info_gain_type: "log_prob_diff" or "prob_diff"

    Returns:
        scalar reward; NaN if either input is NaN/inf.
    """
    if math.isnan(cur_mean_lp) or math.isnan(prev_mean_lp):
        return float("nan")
    if math.isinf(cur_mean_lp) or math.isinf(prev_mean_lp):
        return float("nan")

    if info_gain_type == "log_prob_diff":
        return cur_mean_lp - prev_mean_lp
    elif info_gain_type == "prob_diff":
        # exp(mean_log_prob_t) - exp(mean_log_prob_{t-1})
        # Numerically: bounded by [-1, 1] since both args are in [0, 1].
        return math.exp(cur_mean_lp) - math.exp(prev_mean_lp)
    else:
        raise ValueError(
            f"[IGPO] info_gain_type must be 'log_prob_diff' or 'prob_diff', got {info_gain_type!r}"
        )


def write_sparse_token_rewards(
    bsz: int,
    response_len: int,
    solution_strs: List[str],
    valid_response_lengths: List[int],
    tokenizer,
    r_final_list: List[float],
    plans: List[GTReplayPlan],
    mean_logprobs: Dict[Tuple[int, int], float],
    info_gain_type: str = "log_prob_diff",
    alpha_outcome: float = 1.0,
    assistant_separator: str = DEFAULT_ASSISTANT_SEPARATOR,
    f1_zero_replacement: float = 1e-10,
    ig_zero_replacement: float = 1e-10,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build the (bs, response_len) sparse reward tensor for IGPO V8.

    Args:
        bsz:                       batch size.
        response_len:              shape[-1] of data.batch["responses"].
        solution_strs:             decoded responses (one per sample); used
                                   to recompute char->token offsets so that
                                   per-turn IG positions are precise even
                                   under arbitrary BPE merges.
        valid_response_lengths:    int per sample; the LAST valid response
                                   token index = valid_len - 1, where the
                                   outcome reward (R_final) is written.
                                   Pass `data.batch["attention_mask"][:,
                                   prompt_length:].sum(dim=-1).tolist()`.
        tokenizer:                 HF tokenizer.
        r_final_list:              outcome reward per sample (LLM judge).
        plans:                     per-sample GTReplayPlan from the
                                   logprob replay step. We use only their
                                   turn_spans.
        mean_logprobs:             (sample_idx, turn_idx) -> mean log-prob.
        info_gain_type:            "log_prob_diff" or "prob_diff".
        alpha_outcome:             scalar multiplier on R_final (default 1.0).
        f1_zero_replacement:       if R_final happens to be exactly 0, use
                                   this small positive value instead so the
                                   advantage code can detect the outcome
                                   position via `token_level_rewards != 0`.
                                   Mirrors IGPO-main info_gain.py.
        ig_zero_replacement:       same but for IG values.
        assistant_separator:       (currently unused at this layer; kept for
                                   API symmetry with build_pseudo_samples)

    Returns:
        reward_tensor (bs, response_len) torch.float32 sparse tensor.
        metrics dict with the following keys:
            - igpo/info_gain_count            (#non-zero IG positions written)
            - igpo/info_gain_mean             (mean over non-NaN, non-zero IG)
            - igpo/info_gain_std
            - igpo/info_gain_abs_mean
            - igpo/info_gain_nan_count        (turns where prev/cur logprob NaN)
            - igpo/r_final_mean
            - igpo/r_final_std
            - igpo/turns_per_traj_mean
            - igpo/sparse_reward_density      (#non-zero / total token slots)
            - igpo/per_turn_ig_p50/p25/p75    (rough distribution diagnostics)
    """
    if info_gain_type not in ("log_prob_diff", "prob_diff"):
        raise ValueError(
            f"[IGPO] info_gain_type must be 'log_prob_diff' or 'prob_diff', got {info_gain_type!r}"
        )

    reward_tensor = torch.zeros((bsz, response_len), dtype=torch.float32)

    # Diagnostics accumulators
    all_ig_values: List[float] = []
    nan_ig_count = 0
    total_turns = 0
    per_turn_ig_buckets: Dict[int, List[float]] = {}

    # ---- per sample: write outcome + per-turn IG ----
    for i in range(bsz):
        if i >= len(plans):
            continue
        plan = plans[i]

        # ---- outcome reward (always write at last valid response token) ----
        valid_len = int(valid_response_lengths[i]) if i < len(valid_response_lengths) else 0
        if valid_len > 0 and i < len(r_final_list):
            r_final = float(r_final_list[i]) * alpha_outcome
            if r_final == 0.0:
                r_final = f1_zero_replacement
            reward_tensor[i, valid_len - 1] = r_final

        # ---- IG rewards on each non-final turn ----
        if plan.skip_reason is not None or len(plan.turn_spans) <= 1:
            continue

        sol = solution_strs[i] if i < len(solution_strs) else ""
        if not sol:
            continue

        # Re-compute char->token offsets ON the response_ids actually used
        # in the rollout, so that the token positions we write into match
        # the real response token grid.
        try:
            sol_enc = tokenizer(
                sol,
                return_offsets_mapping=True,
                add_special_tokens=False,
                return_tensors=None,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[IGPO] sample {i}: tokenize(sol) failed: {e}")
            continue

        sol_token_ids = sol_enc["input_ids"]
        sol_offsets = sol_enc["offset_mapping"]
        n_sol_tokens = len(sol_token_ids)
        if n_sol_tokens == 0:
            continue

        # Iterate non-final turns: t in [0, len(turn_spans)-2]
        prev_mean_lp: Optional[float] = None
        for t in range(len(plan.turn_spans) - 1):
            total_turns += 1
            cur_mean_lp = mean_logprobs.get((i, t), None)
            if cur_mean_lp is None or math.isnan(cur_mean_lp):
                # No measurement -> reset the chain. Subsequent turn is
                # treated as a fresh starting point (matches IGPO-main).
                prev_mean_lp = None
                nan_ig_count += 1
                continue

            if prev_mean_lp is None:
                # First successful turn for this sample: just seed the
                # baseline; no IG reward to write at turn 0.
                prev_mean_lp = cur_mean_lp
                continue

            ig_value = _info_gain(cur_mean_lp, prev_mean_lp, info_gain_type)
            if math.isnan(ig_value) or math.isinf(ig_value):
                prev_mean_lp = cur_mean_lp
                nan_ig_count += 1
                continue

            # Map turn[t]'s last char -> token index inside the response
            ts = plan.turn_spans[t]
            tok_lo, tok_hi = char_span_to_token_span(
                ts.char_start, ts.char_end, sol_offsets, n_sol_tokens
            )
            if tok_lo >= tok_hi:
                prev_mean_lp = cur_mean_lp
                continue

            last_tok_idx = min(tok_hi - 1, response_len - 1)
            if last_tok_idx < 0 or last_tok_idx >= response_len:
                prev_mean_lp = cur_mean_lp
                continue

            # Avoid silently overwriting the outcome position (extremely
            # rare; happens only if turn_spans report the final turn's
            # last token as a non-final turn end, which split_trajectory
            # explicitly prevents).
            v = ig_value if ig_value != 0.0 else ig_zero_replacement
            reward_tensor[i, last_tok_idx] = v

            all_ig_values.append(ig_value)
            per_turn_ig_buckets.setdefault(t, []).append(ig_value)

            prev_mean_lp = cur_mean_lp

    # ---- metrics ----
    metrics: Dict[str, Any] = {}
    if all_ig_values:
        arr = np.asarray(all_ig_values, dtype=np.float64)
        metrics["igpo/info_gain_count"] = int(arr.size)
        metrics["igpo/info_gain_mean"] = float(arr.mean())
        metrics["igpo/info_gain_std"] = float(arr.std())
        metrics["igpo/info_gain_abs_mean"] = float(np.abs(arr).mean())
    else:
        metrics["igpo/info_gain_count"] = 0
        metrics["igpo/info_gain_mean"] = 0.0
        metrics["igpo/info_gain_std"] = 0.0
        metrics["igpo/info_gain_abs_mean"] = 0.0

    metrics["igpo/info_gain_nan_count"] = int(nan_ig_count)

    if r_final_list:
        rf_arr = np.asarray(r_final_list, dtype=np.float64)
        metrics["igpo/r_final_mean"] = float(rf_arr.mean())
        metrics["igpo/r_final_std"] = float(rf_arr.std())
    else:
        metrics["igpo/r_final_mean"] = 0.0
        metrics["igpo/r_final_std"] = 0.0

    metrics["igpo/turns_per_traj_mean"] = float(total_turns) / max(bsz, 1)
    nz = (reward_tensor != 0).sum().item()
    metrics["igpo/sparse_reward_density"] = float(nz) / float(bsz * response_len)

    # Per-turn-position IG distribution (helps decide whether late-turn IG
    # is worth keeping; if median ~ 0 for t > K, you can lower
    # max_turns_for_ig.)
    for t, values in sorted(per_turn_ig_buckets.items()):
        if not values:
            continue
        v_arr = np.asarray(values, dtype=np.float64)
        metrics[f"igpo/per_turn_ig_t{t}_count"] = int(v_arr.size)
        metrics[f"igpo/per_turn_ig_t{t}_mean"] = float(v_arr.mean())
        metrics[f"igpo/per_turn_ig_t{t}_abs_mean"] = float(np.abs(v_arr).mean())

    return reward_tensor, metrics
