# -*- coding: utf-8 -*-
"""IGPO GT-LogProb offline replay utilities (V8 baseline).

This module supports the V8 IGPO reward manager. After sglang multi-turn
rollout finishes, we need to compute, for every non-final assistant turn t
of every trajectory, the policy's mean log-probability over the GT answer
tokens conditioned on the prefix that includes only turns [0, t]. The
turn-to-turn delta of that mean log-prob is then the IGPO information-gain
reward (see process_reward_igpo.py).

Design constraints
==================
1. **Vectorised single forward**:
   All pseudo-samples (across all trajectories and all turns) are packed
   into a single DataProto and passed through actor_rollout_wg.compute_log_prob
   exactly once. Internally that path already does FSDP all-gather + dynamic
   batching by ppo_max_token_len_per_gpu, so we get optimal throughput.

2. **Narrow response window** (key V8 optimisation):
   verl's compute_log_prob slices `output[:, -response_length-1:-1]` to
   extract the response logits — `response_length = responses.size(1)`.
   By keeping `responses` short (just PREFIX + GT + SUFFIX, ~30-50 tokens)
   we shrink the entropy softmax buffer from `vocab × full_resp_len` to
   `vocab × ~50`, which lets us fit a much larger micro-batch within the
   same `ppo_max_token_len_per_gpu` cap. This is the single biggest
   wall-time win.

3. **Strict alignment with verl's compute_log_prob semantics**:
   We follow `pseudo_generate_sequences` from IGPO-main verbatim — the only
   change is that the "response" portion of the synthetic sample is
   constrained to PREFIX+GT+SUFFIX rather than the full original response.

4. **No special handling for entropy**:
   compute_log_prob returns both `old_log_probs` and `entropys`. We only
   read `old_log_probs`. `entropys` is computed by the worker for free
   (same forward), so leaving it alone costs nothing.

References
----------
- IGPO-main/scrl/llm_agent/generation.py : pseudo_generate_sequences,
  GT token-range location via offset_mapping, vectorized batch packing.
- IGPO-main/scrl/llm_agent/prealigned_vectorized.py : reference for the
  pre-aligned left-padding strategy this file implements.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import (
    get_response_mask,
    pad_2d_list_to_length,
)


logger = logging.getLogger(__name__)


# Strings used to wrap the GT answer into a "pseudo final turn".
# Kept identical to IGPO-main/scrl/llm_agent/generation.py so that the
# numerics match the reference implementation token-for-token (modulo
# tokenizer differences between Qwen2.5 and the model used here).
IGPO_GT_PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
IGPO_GT_SUFFIX = "\n</answer><|im_end|>"


# Default chat-template separator used by Qwen-style multi-turn rollouts.
# This MUST match `actor_rollout_ref.rollout.multi_turn.format=qwen` in
# run_train_v8_igpo.sh. If you switch to a different chat template you
# must also override `assistant_separator` when calling
# `split_trajectory_into_turns`.
DEFAULT_ASSISTANT_SEPARATOR = "\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TurnSpan:
    """Half-open character span of one assistant turn in `solution_str`.

    The span covers the assistant's reply (and any subsequent tool response)
    UP TO BUT NOT INCLUDING the next "\n<|im_start|>assistant\n" separator
    or the end of the trajectory.
    """

    char_start: int  # inclusive
    char_end: int  # exclusive
    is_final: bool  # True iff this is the last assistant turn (outcome reward goes here)


@dataclass
class GTReplayPlan:
    """Per-sample plan for the GT-LogProb forward pass.

    Attributes:
        sample_idx:        index into the original DataProto (rollout batch).
        turn_spans:        list of TurnSpan, ordered (assistant_0, assistant_1, ...).
        gt_text:           the ground-truth answer string used for IG.
        skip_reason:       if non-None, this sample is skipped entirely
                           (e.g. empty GT, tokenisation failure, no turns).
    """

    sample_idx: int
    turn_spans: List[TurnSpan] = field(default_factory=list)
    gt_text: str = ""
    skip_reason: Optional[str] = None


@dataclass
class PseudoSample:
    """One synthetic forward sample for the GT-LogProb pass.

    Attributes:
        sample_idx:    parent trajectory index (in the original batch).
        turn_idx:      which non-final assistant turn this prefix ends after
                       (0-indexed; ranges over [0, T-2] inclusive).
        prompt_ids:    tokens of [system + user + assistant_0 ... assistant_t
                       + tool_t ...] up to (but not including) PREFIX.
        response_ids:  tokens of PREFIX + GT + SUFFIX (the "narrow window").
        gt_token_lo:   index into response_ids where GT tokens start.
        gt_token_hi:   index into response_ids where GT tokens end (exclusive).
    """

    sample_idx: int
    turn_idx: int
    prompt_ids: List[int]
    response_ids: List[int]
    gt_token_lo: int
    gt_token_hi: int


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------


def split_trajectory_into_turns(
    solution_str: str,
    assistant_separator: str = DEFAULT_ASSISTANT_SEPARATOR,
) -> List[TurnSpan]:
    """Split a decoded trajectory string into assistant-turn character spans.

    The chat template that sglang emits for Qwen multi-turn looks like:

        <|im_start|>system\n...<|im_end|>
        \n<|im_start|>user\n...<|im_end|>
        \n<|im_start|>assistant\n  <-- separator #1
        ...turn 0 reply (with optional <tool_call>)...<|im_end|>
        \n<|im_start|>tool\n...<|im_end|>            (optional)
        \n<|im_start|>assistant\n  <-- separator #2
        ...turn 1 reply...<|im_end|>
        ...
        \n<|im_start|>assistant\n  <-- separator #T
        ...turn T-1 reply with <answer>...

    `solution_str` is the response string returned by tokenizer.batch_decode
    on `data.batch["responses"]`, i.e. it covers EVERYTHING after the
    initial user prompt. So separator #1 is what marks the start of turn 0.

    We return a list of TurnSpan whose `char_start` lands right after each
    separator's last character, and whose `char_end` lands right before the
    next separator (or at the end of the string). The final element has
    `is_final=True`.

    Edge cases:
      - If no separator is found the whole string is treated as a single
        (final) turn — this happens for trajectories generated by an
        engine variant that omits the first separator. We still produce
        one final turn so outcome reward has a valid attachment point.
      - Content BEFORE the first separator is discarded; it is just the
        bookkeeping suffix of the user prompt (not an assistant turn).
    """
    sep_positions: List[int] = []
    search_pos = 0
    while True:
        i = solution_str.find(assistant_separator, search_pos)
        if i == -1:
            break
        sep_positions.append(i)
        search_pos = i + 1

    if len(sep_positions) == 0:
        # Fallback: whole string is one final turn
        return [TurnSpan(char_start=0, char_end=len(solution_str), is_final=True)]

    turns: List[TurnSpan] = []
    for k, sep_pos in enumerate(sep_positions):
        turn_start = sep_pos + len(assistant_separator)
        turn_end = (
            sep_positions[k + 1] if k + 1 < len(sep_positions) else len(solution_str)
        )
        turns.append(TurnSpan(char_start=turn_start, char_end=turn_end, is_final=False))

    if turns:
        turns[-1].is_final = True
    return turns


def char_span_to_token_span(
    char_start: int,
    char_end: int,
    offset_mapping: List[Tuple[int, int]],
    tokens_size: int,
) -> Tuple[int, int]:
    """Convert a character half-open span into a token half-open span.

    Mirrors verl/utils/reward_score/info_gain.py::_char_pos_to_token_idx
    semantics — robust to subword boundaries and BPE re-tokenisation
    artefacts.
    """
    if tokens_size <= 0 or char_end <= char_start:
        return 0, 0

    # Token start: first token whose [t_start, t_end) intersects char_start
    t_lo = tokens_size
    for i, (cs, ce) in enumerate(offset_mapping):
        if ce > char_start:  # token covers a char >= char_start
            t_lo = i
            break

    # Token end: one past the last token whose [t_start, t_end) intersects
    # [char_start, char_end). Equivalent: first token whose t_start >= char_end.
    t_hi = tokens_size
    for i, (cs, ce) in enumerate(offset_mapping):
        if cs >= char_end:
            t_hi = i
            break

    if t_lo >= t_hi:
        return 0, 0
    return int(t_lo), int(t_hi)


# ---------------------------------------------------------------------------
# Pseudo-sample construction
# ---------------------------------------------------------------------------


def _normalize_gt_text(ground_truth: Any) -> str:
    """Coerce the V8 ground_truth field into a plain answer string.

    V8 reuses the same `reward_model.ground_truth` field as v3/v4/v6 — the
    field is a dict with a `target` key (see browsecomp_zh.compute_score:
    `ground_truth['target']`). For robustness we also accept a bare string
    or a list-of-dicts (common in fact-checking datasets).
    """
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, dict):
        for key in ("target", "answer", "gold_answer", "gt"):
            if key in ground_truth and ground_truth[key]:
                return str(ground_truth[key]).strip()
        return ""
    if isinstance(ground_truth, str):
        return ground_truth.strip()
    if isinstance(ground_truth, (list, tuple)):
        # multi-label format ([{"label": "true/false"}]) -> use majority
        # label as a degenerate "answer" string. This is a defensive
        # fallback; for IGPO this kind of GT yields ~0 information-gain.
        try:
            return str(ground_truth[0].get("label", "")) if ground_truth else ""
        except Exception:  # noqa: BLE001
            return str(ground_truth)
    return str(ground_truth)


def build_pseudo_samples(
    solution_strs: List[str],
    prompt_strs: List[str],
    ground_truths: List[Any],
    tokenizer,
    max_turns_for_ig: int = 20,
    assistant_separator: str = DEFAULT_ASSISTANT_SEPARATOR,
) -> Tuple[List[PseudoSample], List[GTReplayPlan]]:
    """Build per-sample pseudo-forward inputs.

    Args:
        solution_strs: list of decoded RESPONSES (one per trajectory).
        prompt_strs:   list of decoded PROMPTS (one per trajectory). The
                       full input fed to the LM is `prompt_strs[i] + prefix_t
                       + PREFIX + GT + SUFFIX`.
        ground_truths: list of GT objects (string / dict / list).
        tokenizer:     HuggingFace tokenizer (must support
                       `return_offsets_mapping=True`).
        max_turns_for_ig:
                       Hard cap on how many non-final turns per trajectory
                       contribute IG samples. Set to 20 (= effectively
                       unlimited at our `MAX_ASSISTANT_TURNS=20`) by default
                       to match IGPO paper. Lower it to trade fidelity for
                       wall-time.
        assistant_separator:
                       The chat-template separator used to delimit
                       assistant turns within `solution_strs`.

    Returns:
        (pseudo_samples, replay_plans):
            pseudo_samples: flat list of PseudoSample objects ready to be
                            packed into a DataProto.
            replay_plans:   per-trajectory metadata (turn spans, skip reason)
                            used by process_reward_igpo to map results back
                            to the sparse token-level reward tensor.
    """
    assert len(solution_strs) == len(prompt_strs) == len(ground_truths)
    pseudo_samples: List[PseudoSample] = []
    plans: List[GTReplayPlan] = []

    # Pre-tokenise GT block once per sample. It depends only on GT text.
    for sample_idx, (sol, prompt, gt_raw) in enumerate(
        zip(solution_strs, prompt_strs, ground_truths)
    ):
        plan = GTReplayPlan(sample_idx=sample_idx)
        plans.append(plan)

        # ---- 1) extract turn spans ----
        try:
            turn_spans = split_trajectory_into_turns(
                sol, assistant_separator=assistant_separator
            )
        except Exception as e:  # noqa: BLE001
            plan.skip_reason = f"split_failed: {e}"
            continue
        plan.turn_spans = turn_spans

        # ---- 2) extract GT text ----
        gt_text = _normalize_gt_text(gt_raw)
        plan.gt_text = gt_text
        if not gt_text:
            plan.skip_reason = "empty_gt"
            continue

        if len(turn_spans) <= 1:
            # Only outcome turn exists; no IG samples to compute.
            continue

        # ---- 3) tokenise GT block: PREFIX + GT + SUFFIX ----
        full_block = f"{IGPO_GT_PREFIX}{gt_text}{IGPO_GT_SUFFIX}"
        try:
            block_enc = tokenizer(
                full_block,
                return_tensors=None,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
        except Exception as e:  # noqa: BLE001
            plan.skip_reason = f"gt_tokenize_failed: {e}"
            continue

        block_ids = block_enc["input_ids"]
        block_offsets = block_enc["offset_mapping"]
        if len(block_ids) == 0:
            plan.skip_reason = "empty_block_ids"
            continue

        gt_char_start = len(IGPO_GT_PREFIX)
        gt_char_end = gt_char_start + len(gt_text)
        gt_tok_lo, gt_tok_hi = char_span_to_token_span(
            gt_char_start, gt_char_end, block_offsets, len(block_ids)
        )
        if gt_tok_lo >= gt_tok_hi:
            plan.skip_reason = "empty_gt_token_range"
            continue

        # ---- 4) tokenise prompt once + each prefix_t once ----
        # Note: we re-tokenise prefix_t (= prompt + sol[:turn_spans[t].char_end])
        # for each non-final turn, paying O(num_turns) tokenisations. This
        # is O(few ms) per trajectory and avoids subtle off-by-one issues
        # that arise from incremental tokenisation around chat-template
        # special tokens.
        try:
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors=None
            )["input_ids"]
        except Exception as e:  # noqa: BLE001
            plan.skip_reason = f"prompt_tokenize_failed: {e}"
            continue

        num_non_final = len(turn_spans) - 1  # exclude the outcome turn
        num_to_keep = min(num_non_final, int(max_turns_for_ig))
        for t in range(num_to_keep):
            # End of solution_str slice = end of turn t (turn_spans[t].char_end)
            sol_prefix = sol[: turn_spans[t].char_end]
            try:
                sol_prefix_ids = tokenizer(
                    sol_prefix,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[IGPO] sample={sample_idx} turn={t} prefix tokenise failed: {e}"
                )
                continue

            full_prompt_ids = list(prompt_ids) + list(sol_prefix_ids)
            pseudo_samples.append(
                PseudoSample(
                    sample_idx=sample_idx,
                    turn_idx=t,
                    prompt_ids=full_prompt_ids,
                    response_ids=list(block_ids),
                    gt_token_lo=int(gt_tok_lo),
                    gt_token_hi=int(gt_tok_hi),
                )
            )

    return pseudo_samples, plans


# ---------------------------------------------------------------------------
# Pack -> DataProto
# ---------------------------------------------------------------------------


def pack_pseudo_samples_to_dataproto(
    pseudo_samples: List[PseudoSample],
    pad_token_id: int,
    eos_token_id: int,
    device: torch.device | str = "cpu",
) -> DataProto:
    """Left-pad prompts, right-pad responses; emit a verl-shaped DataProto.

    Layout (per sample):
        input_ids = [PAD]*lp || prompt_ids || response_ids || [PAD]*rp
                     ^------- prompt -------^  ^----- response -----^
        prompts   = input_ids[:, :max_prompt_len]
        responses = input_ids[:, max_prompt_len:]

    verl's compute_log_prob slices `output[:, -response_length-1:-1, :]` to
    extract response logits. Since `response_length = responses.size(1) =
    max_response_len`, every sample's response window is correctly aligned
    AS LONG AS we right-pad responses (so the padding sits at the END of
    the sequence, beyond the actual GT tokens). This matches the layout
    used by `pseudo_generate_sequences` in IGPO-main and by verl's
    standard rollout output.
    """
    assert len(pseudo_samples) > 0, "pack_pseudo_samples_to_dataproto: empty input"

    max_prompt_len = max(len(s.prompt_ids) for s in pseudo_samples)
    max_response_len = max(len(s.response_ids) for s in pseudo_samples)

    bsz = len(pseudo_samples)
    input_ids = torch.full(
        (bsz, max_prompt_len + max_response_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)

    for i, s in enumerate(pseudo_samples):
        lp = len(s.prompt_ids)
        rp = len(s.response_ids)
        # left-pad prompt: prompt sits at [max_prompt_len-lp, max_prompt_len)
        prompt_start = max_prompt_len - lp
        input_ids[i, prompt_start:max_prompt_len] = torch.tensor(
            s.prompt_ids, dtype=torch.long, device=device
        )
        attention_mask[i, prompt_start:max_prompt_len] = 1
        # right-pad response: response sits at [max_prompt_len, max_prompt_len+rp)
        input_ids[i, max_prompt_len : max_prompt_len + rp] = torch.tensor(
            s.response_ids, dtype=torch.long, device=device
        )
        attention_mask[i, max_prompt_len : max_prompt_len + rp] = 1

    # position_ids: standard cumulative-mask layout.
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0).long()

    prompts = input_ids[:, :max_prompt_len].contiguous()
    responses = input_ids[:, max_prompt_len:].contiguous()

    # `prompts` field of DataProto holds prompt-only tokens (verl convention).
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=bsz,
    )
    return DataProto(batch=batch, non_tensor_batch={})


# ---------------------------------------------------------------------------
# Forward + result extraction
# ---------------------------------------------------------------------------


def extract_mean_logprobs(
    log_prob_output: DataProto,
    pseudo_samples: List[PseudoSample],
) -> List[float]:
    """For each pseudo-sample, return the mean log-prob over its GT tokens.

    `log_prob_output.batch["old_log_probs"]` is shaped
    `(bsz, max_response_len)` — log-prob per response token (excluding the
    last position, which corresponds to the next-token prediction beyond
    the EOS). Indexing into [gt_token_lo, gt_token_hi) gives the GT span.
    """
    old_log_probs = log_prob_output.batch["old_log_probs"]
    assert old_log_probs.dim() == 2, (
        f"[IGPO] expected 2D old_log_probs, got shape {tuple(old_log_probs.shape)}"
    )

    out: List[float] = []
    for i, s in enumerate(pseudo_samples):
        lo, hi = s.gt_token_lo, s.gt_token_hi
        if lo >= hi or hi > old_log_probs.shape[1]:
            out.append(float("nan"))
            continue
        slice_lp = old_log_probs[i, lo:hi]
        if slice_lp.numel() == 0:
            out.append(float("nan"))
            continue
        m = slice_lp.mean().item()
        if math.isnan(m) or math.isinf(m):
            out.append(float("nan"))
            continue
        out.append(float(m))
    return out


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def run_gt_logprob_replay(
    solution_strs: List[str],
    prompt_strs: List[str],
    ground_truths: List[Any],
    tokenizer,
    actor_rollout_wg,
    max_turns_for_ig: int = 20,
    assistant_separator: str = DEFAULT_ASSISTANT_SEPARATOR,
) -> Tuple[Dict[Tuple[int, int], float], List[GTReplayPlan], Dict[str, Any]]:
    """End-to-end GT-LogProb replay for one batch.

    Args:
        solution_strs:    decoded responses (one per trajectory).
        prompt_strs:      decoded prompts.
        ground_truths:    GT objects.
        tokenizer:        HF tokenizer.
        actor_rollout_wg: verl ActorRolloutRefWorker — must expose
                          `compute_log_prob(DataProto) -> DataProto`.
        max_turns_for_ig: cap on non-final turns per trajectory (default 20).
        assistant_separator: chat-template separator string.

    Returns:
        mean_logprobs:  dict mapping (sample_idx, turn_idx) -> mean log-prob
                        of the GT span. Keys are PRESENT for every
                        successful (sample, turn) pair and ABSENT otherwise
                        (e.g. tokenisation failure, NaN, empty GT).
        plans:          per-sample replay plans (with turn_spans).
        metrics:        diagnostics dict — see body for keys.
    """
    metrics: Dict[str, Any] = {
        "igpo/gt_logprob_total_s": 0.0,
        "igpo/gt_logprob_token_count": 0,
        "igpo/gt_logprob_num_pseudo_samples": 0,
        "igpo/gt_logprob_num_skipped_samples": 0,
        "igpo/gt_logprob_avg_turns_per_traj": 0.0,
        "igpo/gt_logprob_max_response_len": 0,
        "igpo/gt_logprob_max_prompt_len": 0,
    }

    pseudo_samples, plans = build_pseudo_samples(
        solution_strs=solution_strs,
        prompt_strs=prompt_strs,
        ground_truths=ground_truths,
        tokenizer=tokenizer,
        max_turns_for_ig=max_turns_for_ig,
        assistant_separator=assistant_separator,
    )

    # diagnostics
    skipped = sum(1 for p in plans if p.skip_reason is not None)
    total_turns = sum(len(p.turn_spans) for p in plans)
    metrics["igpo/gt_logprob_num_pseudo_samples"] = len(pseudo_samples)
    metrics["igpo/gt_logprob_num_skipped_samples"] = skipped
    metrics["igpo/gt_logprob_avg_turns_per_traj"] = (
        float(total_turns) / max(len(plans), 1)
    )

    if len(pseudo_samples) == 0:
        logger.warning(
            "[IGPO] No pseudo-samples produced (likely all trajectories "
            "are single-turn or have empty GT). Skipping GT log-prob forward."
        )
        return {}, plans, metrics

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    pseudo_dp = pack_pseudo_samples_to_dataproto(
        pseudo_samples=pseudo_samples,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        device="cpu",
    )

    metrics["igpo/gt_logprob_max_response_len"] = int(
        pseudo_dp.batch["responses"].shape[1]
    )
    metrics["igpo/gt_logprob_max_prompt_len"] = int(
        pseudo_dp.batch["prompts"].shape[1]
    )
    metrics["igpo/gt_logprob_token_count"] = int(
        pseudo_dp.batch["attention_mask"].sum().item()
    )

    # ----- forward -----
    import time as _time

    _t0 = _time.time()
    try:
        # actor_rollout_wg.compute_log_prob handles dynamic_bsz / SP / TP
        # internally based on `actor_rollout_ref.actor.use_dynamic_bsz` and
        # `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`, so we just
        # hand off the DataProto.
        lp_output = actor_rollout_wg.compute_log_prob(pseudo_dp)
    except Exception as e:
        metrics["igpo/gt_logprob_total_s"] = _time.time() - _t0
        logger.error(
            f"[IGPO] compute_log_prob FAILED: {type(e).__name__}: {e}. "
            f"Returning empty IG; outcome reward path will still work."
        )
        return {}, plans, metrics
    metrics["igpo/gt_logprob_total_s"] = _time.time() - _t0

    # ----- extract -----
    mean_lps = extract_mean_logprobs(lp_output, pseudo_samples)
    out: Dict[Tuple[int, int], float] = {}
    nan_count = 0
    for s, m in zip(pseudo_samples, mean_lps):
        if math.isnan(m):
            nan_count += 1
            continue
        out[(s.sample_idx, s.turn_idx)] = m
    metrics["igpo/gt_logprob_nan_count"] = nan_count

    return out, plans, metrics
