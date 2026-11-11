#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-in-Group GRPO Process Reward Module

Implements step-level process reward and advantage estimation for
Group-in-Group (GiG) GRPO training.

Core modules (from need_summary V3):
  Module 1:  Step state evaluation & grouping
  Module 1A: Monotonicity enforcement (V3)
  Module 2:  Process reward computation (immediate + discounted return)
  Module 3:  Step-level relative advantage (A_S)
  Module 4:  Scale alignment & GiG advantage fusion (A = A_E + omega * A_S)

Author: anonymous@example.com
Version: 3.0.0
Date: 2026-03-09
"""

import json
import logging
import os
import re
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verl.utils.reward_score.api_client_factory import get_provider_config, call_model_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# API tokens for external model calls
# ============================================================
TOKENS = [
]


# ============================================================
# External model API call
# ============================================================
def call_model_api_with_messages(
    messages: List[Dict[str, str]],
    model: str = "claude-4-sonnet-20250514",
    token_index: int = 0,
    tokens: Optional[List[str]] = None,
    max_retries: int = 20,
    api_provider: str = "vendor_a",
    api_key: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    global_timeout: float = 600.0,
    temperature: float = 0.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Generic model API call function with multi-model failover.
    
    Supports switching between vendor_a and Vendor B API providers.
    When the primary model fails (rate limit or errors), automatically
    rotates through fallback models.
    
    Args:
        messages: Chat messages.
        model: Primary model name.
        token_index: Token index for rotation.
        tokens: List of API tokens.
        max_retries: Maximum non-rate-limit retries per model.
        api_provider: API provider ('vendor_a' or 'vendor_b').
        api_key: API key override.
        fallback_models: List of fallback model names to try on failure.
        global_timeout: Global timeout in seconds across all retries.
        temperature: External model sampling temperature. MUST be 0.0 for
            step-state / answer evaluation to guarantee judgement
            idempotency — same (trajectory, rubric) input ⇒ same output,
            so the group key (uid, rubric_state) is stable across
            mini-batches. Non-zero values are accepted only for
            back-compat; do NOT rely on that in production.
    """
    return call_model_api(
        messages=messages,
        provider=api_provider,
        model=model,
        api_key=api_key,
        max_retries=max_retries,
        fallback_models=fallback_models,
        global_timeout=global_timeout,
        temperature=temperature,
    )


# ============================================================
# Module 1: Step Extraction
# ============================================================
def extract_steps_from_trajectory(trajectory_text: str) -> List[Dict[str, Any]]:
    """Extract structured steps from a rollout trajectory.

    Parses the trajectory into structured steps, each containing think,
    action (tool_call or answer), and response (tool_response) fields.
    Also builds formatted cumulative context with [Step i] labels for
    the judge model.

    Each step is identified by a <think> block, followed by either:
      - <tool_call>...</tool_call><tool_response>...</tool_response> (intermediate)
      - <answer>...</answer> (final)

    Args:
        trajectory_text: The decoded response text from rollout.

    Returns:
        List of step dicts, each containing:
          - step_index: int
          - think: str (think content for this step)
          - action: str (tool_call or answer tag with content)
          - response: str (tool_response tag with content)
          - cumulative_text: str (raw text up to and including this step,
                                  for char boundary computation)
          - formatted_cumulative_text: str (formatted with [Step i] labels,
                                            includes Step 0 ~ Step k-1 content
                                            for Step k's evaluation)
          - is_final: bool (True if this is the last step / final_answer)
    """
    if not trajectory_text or not trajectory_text.strip():
        return [{
            "step_index": 0,
            "think": "",
            "action": "",
            "response": "",
            "cumulative_text": trajectory_text or "",
            "formatted_cumulative_text": "",
            "is_final": True,
        }]

    # Parse into structured steps using regex
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tool_response_pattern = re.compile(r'<tool_response>(.*?)</tool_response>', re.DOTALL)
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

    think_matches = list(think_pattern.finditer(trajectory_text))

    if not think_matches:
        # No <think> blocks found — treat entire text as a single step
        return [{
            "step_index": 0,
            "think": "",
            "action": "",
            "response": "",
            "cumulative_text": trajectory_text,
            "formatted_cumulative_text": "",
            "is_final": True,
        }]

    # First pass: extract structured fields for each step
    parsed_steps = []
    for i, think_match in enumerate(think_matches):
        think_content = think_match.group(1).strip()

        # Determine the text range after this <think> block
        step_start = think_match.end()
        if i + 1 < len(think_matches):
            step_end = think_matches[i + 1].start()
        else:
            step_end = len(trajectory_text)

        step_text = trajectory_text[step_start:step_end]

        # Extract tool_call, tool_response, answer
        action = ""
        response = ""

        tc_match = tool_call_pattern.search(step_text)
        tr_match = tool_response_pattern.search(step_text)
        ans_match = answer_pattern.search(step_text)

        if tc_match:
            action = f"<tool_call>{tc_match.group(1)}</tool_call>"
        if tr_match:
            response = f"<tool_response>{tr_match.group(1)}</tool_response>"
        if ans_match:
            action = f"<answer>{ans_match.group(1)}</answer>"

        is_final = (i == len(think_matches) - 1)

        parsed_steps.append({
            "think": think_content,
            "action": action,
            "response": response,
            "is_final": is_final,
        })

    # Second pass: build cumulative_text (raw) and formatted_cumulative_text
    # Also compute raw cumulative boundaries using </tool_response> splitting
    # for char_start/char_end computation.
    raw_parts = re.split(r"(</tool_response>)", trajectory_text)
    raw_cumulative = ""
    raw_cum_boundaries = []  # raw_cum_boundaries[step_idx] = cumulative raw text
    raw_idx = 0
    while raw_idx < len(raw_parts):
        if raw_idx + 1 < len(raw_parts) and raw_parts[raw_idx + 1] == "</tool_response>":
            raw_cumulative += raw_parts[raw_idx] + raw_parts[raw_idx + 1]
            raw_cum_boundaries.append(raw_cumulative)
            raw_idx += 2
        else:
            raw_cumulative += raw_parts[raw_idx]
            raw_idx += 1
    # Final boundary
    if not raw_cum_boundaries or raw_cumulative != raw_cum_boundaries[-1]:
        raw_cum_boundaries.append(raw_cumulative)

    steps = []
    for step_idx, ps in enumerate(parsed_steps):
        # Raw cumulative text (for char boundary computation)
        if step_idx < len(raw_cum_boundaries):
            cum_text = raw_cum_boundaries[step_idx]
        else:
            cum_text = raw_cumulative

        # Formatted cumulative text: Step k's context = Step 0 ~ Step k (inclusive)
        # with [Step i] labels. Evaluating this gives Step k's post-action state.
        formatted_cum = _build_formatted_cumulative_text(parsed_steps, step_idx)

        steps.append({
            "step_index": step_idx,
            "think": ps["think"],
            "action": ps["action"],
            "response": ps["response"],
            "cumulative_text": cum_text,
            "formatted_cumulative_text": formatted_cum,
            "is_final": ps["is_final"],
        })

    # Ensure at least one step exists
    if not steps:
        steps.append({
            "step_index": 0,
            "think": "",
            "action": "",
            "response": "",
            "cumulative_text": trajectory_text,
            "formatted_cumulative_text": "",
            "is_final": True,
        })

    return steps


def _build_formatted_cumulative_text(
    parsed_steps: List[Dict[str, str]],
    current_step_idx: int,
) -> str:
    """Build formatted cumulative context for evaluating Step k's post-action state.

    Step k's evaluation context = Step 0 ~ Step k content (inclusive),
    formatted with [Step i] labels for the judge model to reference.

    The result of evaluating this context gives the post-action state of Step k,
    i.e., which rubrics are satisfied AFTER Step k's action has been executed.

    Rules:
      - current_step_idx = k → returns Step 0 ~ Step k content (inclusive)

    Args:
        parsed_steps: List of parsed step dicts with think/action/response.
        current_step_idx: The step index whose post-action state we want.

    Returns:
        Formatted cumulative text string with [Step i] labels.
    """
    text_parts = []
    for i in range(current_step_idx + 1):  # inclusive: Step 0 ~ Step k
        step = parsed_steps[i]
        text_parts.append(f"[Step {i}]")
        if step.get("think"):
            text_parts.append(f"<think> {step['think']} </think>\n")
        if step.get("action"):
            text_parts.append(step["action"] + "\n")
        if step.get("response"):
            text_parts.append(step["response"])
        text_parts.append("")  # blank line separator

    return "\n".join(text_parts)


# ============================================================
# Module 1: Step State Evaluation via External API
# ============================================================
STEP_STATE_EVALUATION_PROMPT = """You are an expert evaluator assessing the multi-step information retrieval and reasoning progress of an AI Assistant.

You will be provided with:
1. [Question]: The original complex question.
2. [Rubrics]: A list of factual criteria (R1, R2, etc.). Each Rubric is constructed as strict Entity-Relation-Entity statements (e.g., [Subject Entity] + [Action/Relation] + [Object Entity/Location]).
3. [Cumulative Context]: The Assistant's historical trajectory, including search queries, retrieved web documents (<tool_response>), and internal reasoning (<think>).

Your task is to act as a PEDANTIC and ULTRA-STRICT judge. Determine the satisfaction status of EACH Rubric based SOLELY on the [Cumulative Context].

### CRITICAL EVALUATION RULES:

1. **Entity-Level Granularity (Zero Semantic Drift):** You must strictly distinguish between different entity scopes and types. Do NOT substitute entities with loose semantic approximations. 
   - Example: "Chinese Community (华人社区)" is an organization/collective entity. It is strictly NOT equal to "Chinese laborers (华工)". If the rubric requires "Community" but the text only mentions "laborers", it is a MISMATCH (Status 0).

2. **Strict "AND" Logic for Triples:** Break down the Rubric into sub-constraints (Subject, Action/Relation, Object/Location, Time). If even a single sub-constraint or entity is missing or generalized, the entire Rubric MUST be marked as 0.

3. **Constrained Jigsaw Rule (No Forced Associations):**
   You can combine evidence across different steps or webpages, BUT there MUST be an explicit logical bridge in the text. You CANNOT stitch together disconnected facts based on your own assumptions.

4. **Validating <think> (Parametric Knowledge):**
   The Assistant's <think> block is a valid source of truth, provided the statement is a definitive, affirmative factual claim. Guesses or questions in <think> are invalid.

5. **Zero Inference & Anti-Hallucination:**
   Do NOT use your own external knowledge to fill in blanks. If you cannot quote the exact text proving the specific entity and relation, mark it as 0.

6. **Minimal Sufficient Evidence Chain (No Redundancy):**
   When a Rubric is marked as 1, you MUST extract the exact quotes that form the complete logical chain. DO NOT output multiple redundant quotes that prove the same point from different webpages. Only extract the MINIMAL but COMPLETE text segments necessary to verify all constraints in the Rubric, showing the logical flow.

7. **Evidence-Action Alignment (Action-Evidence Alignment):** This is the most crucial point. If a Rubric is rated as 1, its evidence must be derived from the current or historical <tool_response>. If the fact exists only in the Assistant's <think> and lacks external retrieval evidence support, it must be rated as 0, unless the fact is common knowledge. This is to prevent the model from bypassing the search step by using parameter knowledge (Shortcut).

### OUTPUT FORMAT (JSON ONLY)
You must output a strictly valid JSON object. Do not include markdown formatting like ```json. The JSON must follow this exact structure:

{{
  "evaluations": {{
    "R1": {{
      "status": 0 or 1,
      "rationale": "Verify constraints step-by-step. Example: 'The text mentions Chinese laborers (华工), which is a mismatch for the required entity Chinese Community (华人社区). Due to entity mismatch, status is 0.'",
      "evidence_chain": "Provide the minimal, non-redundant exact quotes forming the complete logical chain (e.g., '[Step 1 - Webpage 2] quote A -> [Step 3 - <think>] quote B'). Leave empty \"\" if status is 0."
    }},
    "R2": {{
      "status": 0 or 1,
      "rationale": "...",
      "evidence_chain": "..."
    }}
  }}
}}

[Question]
{question}

[Rubrics]
{rubrics}

[Cumulative Context]
{cumulative_text}

"""


def evaluate_step_state(
    question: str,
    rubrics: List[str],
    cumulative_text: str,
    model: str = "gpt-4o",
    token_index: int = 0,
    api_provider: str = "vendor_a",
    api_key: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    global_timeout: float = 600.0,
) -> List[int]:
    """Evaluate the state vector for a single step via external API.

    Args:
        question: The original query.
        rubrics: List of rubric criteria strings (from constraints_gt).
        cumulative_text: The cumulative trajectory text up to this step.
        model: The judge model to use.
        token_index: Token index for API authentication.
        api_provider: API provider ('vendor_a' or 'vendor_b').
        api_key: API key override.
        fallback_models: List of fallback model names for failover.
        global_timeout: Global timeout in seconds.

    Returns:
        State vector: list of 0/1 values, length = len(rubrics).
    """
    if not rubrics:
        return []

    # Format rubrics
    rubric_text = "\n".join(
        [f"R{i+1}. {rubric}" for i, rubric in enumerate(rubrics)]
    )

    prompt = STEP_STATE_EVALUATION_PROMPT.format(
        question=question,
        rubrics=rubric_text,
        cumulative_text=cumulative_text,  # Truncate to avoid token limits
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        result_text, _ = call_model_api_with_messages(
            messages, model=model, token_index=token_index, max_retries=3,
            api_provider=api_provider, api_key=api_key,
            fallback_models=fallback_models, global_timeout=global_timeout,
            temperature=0.0,  # HARD CONTRACT: step-state judge must be idempotent
        )

        if not result_text:
            logger.warning(
                f"API returned None for step state evaluation "
                f"(model={model}, rubrics={len(rubrics)}, "
                f"text_len={len(cumulative_text)}) after 3 retries"
            )
            return [0] * len(rubrics)

        # Parse JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find the outermost JSON object by matching balanced braces
            brace_depth = 0
            json_start = -1
            for ci, ch in enumerate(result_text):
                if ch == '{':
                    if brace_depth == 0:
                        json_start = ci
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and json_start != -1:
                        json_str = result_text[json_start:ci + 1]
                        break
            else:
                logger.warning(f"No JSON found in API response: {result_text[:200]}")
                return [0] * len(rubrics)

        try:
            import json_repair
            state_dict = json_repair.loads(json_str)
        except Exception:
            state_dict = json.loads(json_str)

        # Handle nested "evaluations" wrapper from prompt format
        # (may be a dict OR a list, both are supported below)
        if isinstance(state_dict, dict) and "evaluations" in state_dict:
            state_dict = state_dict["evaluations"]

        # ------------------------------------------------------------------
        # Robust extraction: the judge may return the rubric hits in one of
        # several shapes. We handle all of them here rather than letting a
        # single malformed response poison every Phase A downstream (which
        # silently degrades V7 r_t / group keys).
        #   (A) dict keyed by R1/C1/.../Rk :
        #         {"R1":1,"R2":0}   or   {"R1":{"status":1},...}
        #   (B) list of per-rubric entries, positionally aligned to rubrics:
        #         [1, 0, 1]   or   [{"status":1},{"status":0},...]
        #         or [{"R1":1},{"R2":0}]   (each dict holds ONE rubric)
        # Anything else falls back to all-zero and logs a warning so Phase A
        # stats surface the issue.
        # ------------------------------------------------------------------
        def _coerce_entry_to_int(entry):
            if isinstance(entry, dict):
                v = entry.get("status", entry.get("value", entry.get("hit", 0)))
            else:
                v = entry
            return 1 if (v == 1 or v is True or v == "1" or v == "true") else 0

        state_vector = [0] * len(rubrics)

        if isinstance(state_dict, dict):
            for i in range(len(rubrics)):
                entry = state_dict.get(f"R{i+1}", state_dict.get(f"C{i+1}", None))
                state_vector[i] = _coerce_entry_to_int(entry) if entry is not None else 0

        elif isinstance(state_dict, list):
            for i in range(min(len(rubrics), len(state_dict))):
                item = state_dict[i]
                if isinstance(item, dict):
                    # Could be {"R1":1} style (one-key dict) or {"status":1} style.
                    r_key = f"R{i+1}"
                    c_key = f"C{i+1}"
                    if r_key in item or c_key in item:
                        entry = item.get(r_key, item.get(c_key, None))
                        state_vector[i] = _coerce_entry_to_int(entry) if entry is not None else 0
                    else:
                        state_vector[i] = _coerce_entry_to_int(item)
                else:
                    state_vector[i] = _coerce_entry_to_int(item)

        else:
            logger.warning(
                f"[Phase A] unexpected judge payload type={type(state_dict).__name__}, "
                f"falling back to all-zero rubric state. raw={str(state_dict)[:200]}"
            )

        return state_vector

    except Exception as e:
        logger.error(f"Step state evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return [0] * len(rubrics)


# ============================================================
# Module 1A: Monotonicity Enforcement (V3)
# ============================================================
def enforce_monotonicity(
    state_vectors: List[List[int]],
) -> List[List[int]]:
    """Enforce monotonicity on state vectors for a single trajectory.

    Rule: Once a rubric is satisfied (=1) at step t, it must remain 1
    for all subsequent steps t' > t.

    Implementation: Maintain a running OR mask across steps.

    Args:
        state_vectors: List of state vectors [step_0, step_1, ...],
                       each is a list of 0/1 with length = num_rubrics.

    Returns:
        Corrected state vectors with monotonicity enforced.
    """
    if not state_vectors:
        return []

    num_rubrics = len(state_vectors[0])
    mask = [0] * num_rubrics
    corrected = []

    for sv in state_vectors:
        if len(sv) != num_rubrics:
            logger.warning(
                f"State vector length mismatch: expected {num_rubrics}, got {len(sv)}"
            )
            sv = sv[:num_rubrics] + [0] * max(0, num_rubrics - len(sv))

        # OR operation: mask = mask | sv
        mask = [max(m, s) for m, s in zip(mask, sv)]
        corrected.append(list(mask))

    return corrected


# ============================================================
# Module 2: Process Reward Computation
# ============================================================
def compute_immediate_rewards(
    corrected_state_vectors: List[List[int]],
    outcome_reward: float,
) -> List[float]:
    """Compute immediate rewards for each step.

    Each step t's reward reflects the marginal increment caused by step t's
    own action — i.e., the number of rubrics that transitioned from 0 to 1
    due to step t's execution.

    For step 0:
      r_0 = count(0→1 from initial all-zero state to svs[0])

    For non-final steps t > 0:
      r_t = count(0→1 from svs[t-1] to svs[t])

    For final step:
      r_T = outcome_reward (from end-to-end evaluation)

    Note: corrected_state_vectors[t] is the state AFTER step t executes
    (i.e., the post-action state S_{t+1}).

    Args:
        corrected_state_vectors: Monotonicity-corrected state vectors.
            corrected_state_vectors[t] = state after step t execution.
        outcome_reward: The outcome reward for this trajectory (0 or 1).

    Returns:
        List of immediate rewards, one per step.  rewards[t] is the
        marginal increment caused by step t's action.
    """
    if not corrected_state_vectors:
        return []

    num_steps = len(corrected_state_vectors)
    num_rubrics = len(corrected_state_vectors[0])
    rewards = []

    for t in range(num_steps - 1):
        # The state BEFORE step t's action:
        #   - step 0: initial all-zero state (no rubric satisfied yet)
        #   - step t>0: corrected_state_vectors[t-1] (state after step t-1)
        if t == 0:
            prev_state = [0] * num_rubrics
        else:
            prev_state = corrected_state_vectors[t - 1]

        # The state AFTER step t's action:
        current_state = corrected_state_vectors[t]

        # Count 0->1 transitions caused by step t's action
        r_t = sum(
            1 for p, c in zip(prev_state, current_state) if p == 0 and c == 1
        )
        rewards.append(float(r_t))

    # Last step uses outcome reward
    rewards.append(float(outcome_reward))

    return rewards


def compute_discounted_returns(
    immediate_rewards: List[float],
    gamma: float = 0.95,
) -> List[float]:
    """Compute discounted cumulative returns for each step.

    R_t = sum_{k=t}^{T} gamma^{k-t} * r_k

    Args:
        immediate_rewards: List of immediate rewards per step.
        gamma: Discount factor in (0, 1].

    Returns:
        List of discounted returns, one per step.
    """
    if not immediate_rewards:
        return []

    num_steps = len(immediate_rewards)
    returns = [0.0] * num_steps

    # Compute backwards
    returns[-1] = immediate_rewards[-1]
    for t in range(num_steps - 2, -1, -1):
        returns[t] = immediate_rewards[t] + gamma * returns[t + 1]

    return returns


# ============================================================
# Module 3: Step-Level Relative Advantage (A_S)
# ============================================================
def compute_step_level_advantages(
    all_trajectories_step_data: List[Dict[str, Any]],
    gamma: float = 0.95,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Compute step-level relative advantages using state-based grouping.

    Key design decisions:

    1. Pre-action state grouping: The 'corrected_state' in each step_data dict
       represents the state BEFORE the step's action (i.e., the previous step's
       post-action state). Step 0 uses all-zero vector. This ensures that steps
       facing the same state are grouped together for comparing different actions.

    2. Final step exclusion: The final (answer) step is excluded from group
       normalization because its immediate_reward is outcome_reward (0 or 1),
       which has different semantics and scale from non-final steps' marginal
       increment rewards. The final step's step_advantage is set to 0, letting
       the episode-level A_E handle the answer quality signal.

    3. Group-scoped normalization: Only non-final steps from different
       trajectories of the SAME query (uid) with identical pre-action state
       vectors are grouped. Group key = (uid, tuple(pre_action_state)).

    4. Normalized discounted returns: Within each group, immediate rewards are
       normalized first: r_t_norm = (r_t - mean) / max(std, eps). Then
       discounted returns are computed per trajectory using r_t_norm.

    Args:
        all_trajectories_step_data: List of dicts, each containing:
          - 'trajectory_index': int
          - 'step_index': int
          - 'corrected_state': list of int (pre-action state)
          - 'post_action_state': list of int (post-action state)
          - 'immediate_reward': float
          - 'is_final': bool
          - 'uid': str
          - 'step_text': str (full step reasoning text, optional)
        gamma: Discount factor for computing discounted returns.

    Returns:
        Tuple of:
          - all_trajectories_step_data: Same list with 'normalized_reward',
            'step_advantage', and 'group_id' added.
          - metrics: Dict of group statistics for wandb logging.
          - group_details: List of group dicts for dump_grouping_details.
    """
    if not all_trajectories_step_data:
        return [], {}, []

    # Separate final steps from non-final steps.
    # Final steps (answer step) have a semantically different reward
    # (outcome_reward) compared to non-final steps (marginal increment),
    # so they should NOT participate in state-based group normalization.
    non_final_items = [item for item in all_trajectories_step_data if not item.get("is_final", False)]
    final_items = [item for item in all_trajectories_step_data if item.get("is_final", False)]

    # Group non-final steps by (uid, pre-action state_vector)
    # — only same-query steps are grouped
    groups = defaultdict(list)
    for item in non_final_items:
        state_key = (item.get("uid", ""), tuple(item["corrected_state"]))
        groups[state_key].append(item)

    # Normalize immediate rewards within each group and collect group details
    eps = 1e-8
    group_details: List[Dict[str, Any]] = []
    group_id_counter = 0
    for group_key, group_items in groups.items():
        uid, state_tuple = group_key
        rewards = [item["immediate_reward"] for item in group_items]
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))

        # Assign group_id to each member item
        gid = f"g{group_id_counter}"
        group_id_counter += 1

        is_effective = len(group_items) > 1 and std_r > eps

        if len(group_items) <= 1:
            for item in group_items:
                item["normalized_reward"] = 0.0
                item["group_id"] = gid
        else:
            for item in group_items:
                item["normalized_reward"] = (
                    (item["immediate_reward"] - mean_r) / max(std_r, eps)
                )
                item["group_id"] = gid

        # Build reward distribution
        reward_dist: Dict[str, int] = {}
        for r in rewards:
            key = str(float(r))
            reward_dist[key] = reward_dist.get(key, 0) + 1

        # Build human-readable state summary
        state_list = list(state_tuple)
        num_rubrics = len(state_list)
        satisfied_count = sum(state_list)
        satisfied_indices = [f"C{i+1}" for i, v in enumerate(state_list) if v == 1]
        if satisfied_indices:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied ({', '.join(satisfied_indices)})"
        else:
            state_summary = f"{satisfied_count}/{num_rubrics} satisfied"

        # Collect group detail for dump
        group_details.append({
            "group_id": gid,
            "uid": uid,
            "pre_action_state": state_list,
            "pre_action_state_summary": state_summary,
            "group_size": len(group_items),
            "reward_mean": mean_r,
            "reward_std": std_r,
            "is_effective": is_effective,
            "reward_distribution": reward_dist,
            "members": [
                {
                    "trajectory_index": item.get("trajectory_index", -1),
                    "step_index": item["step_index"],
                    "immediate_reward": float(item["immediate_reward"]),
                    "normalized_reward": float(item.get("normalized_reward", 0.0)),
                }
                for item in group_items
            ],
        })

    # Final steps: set normalized_reward, step_advantage, and group_id to None.
    # The answer step's signal is handled by episode-level A_E.
    for item in final_items:
        item["normalized_reward"] = 0.0
        item["step_advantage"] = 0.0
        item["group_id"] = None

    # Recompute discounted returns per trajectory using normalized rewards
    # First, organize steps by trajectory_index
    traj_steps = defaultdict(list)
    for item in all_trajectories_step_data:
        traj_steps[item["trajectory_index"]].append(item)

    for traj_idx, steps in traj_steps.items():
        # Sort by step_index to ensure correct order
        steps.sort(key=lambda x: x["step_index"])
        num_steps = len(steps)
        # Compute discounted returns backwards using normalized rewards.
        # The final step already has step_advantage = 0 (set above),
        # so the backward cumulation naturally uses 0 as the terminal value.
        steps[-1]["step_advantage"] = steps[-1].get("step_advantage", steps[-1]["normalized_reward"])
        for t in range(num_steps - 2, -1, -1):
            steps[t]["step_advantage"] = (
                steps[t]["normalized_reward"] + gamma * steps[t + 1]["step_advantage"]
            )

    # Update step_advantage in group_details members (after discounted return computation)
    for gd in group_details:
        for member in gd["members"]:
            traj_idx = member["trajectory_index"]
            step_idx = member["step_index"]
            # Find the corresponding item to get the final step_advantage
            for item in all_trajectories_step_data:
                if item.get("trajectory_index") == traj_idx and item["step_index"] == step_idx:
                    member["step_advantage"] = float(item.get("step_advantage", 0.0))
                    break

    # ---------- metrics ----------
    num_groups = len(groups)
    group_sizes = [len(v) for v in groups.values()] if groups else []
    metrics: Dict[str, Any] = {
        "gig/num_state_groups":    num_groups,
        "gig/avg_group_size":      float(np.mean(group_sizes)) if group_sizes else 0.0,
        "gig/max_group_size":      int(np.max(group_sizes)) if group_sizes else 0,
        "gig/num_non_final_steps": len(non_final_items),
        "gig/num_final_steps":     len(final_items),
        "gig/num_total_steps":     len(all_trajectories_step_data),
    }

    return all_trajectories_step_data, metrics, group_details


# ============================================================
# Module 4: Scale Alignment & GiG Advantage Fusion
# ============================================================
def align_and_fuse_advantages(
    episode_advantages: np.ndarray,
    step_advantages_per_sample: List[List[Dict[str, Any]]],
    response_mask: np.ndarray,
    omega: float = 0.5,
) -> np.ndarray:
    """Align scales and fuse episode-level and step-level advantages.

    A(a_t^(i)) = A_E(tau_i) + omega * A_S(a_t^(i))

    Both A_E and A_S are independently normalized before fusion.

    The step-level advantage is mapped to the token level:
    each step's advantage is assigned to all tokens belonging to that step.

    Args:
        episode_advantages: [batch_size, seq_len] episode-level advantages (A_E).
        step_advantages_per_sample: For each sample i, list of step dicts containing:
          - 'step_index': int
          - 'step_advantage': float
          - 'token_start': int (start position in response)
          - 'token_end': int (end position in response)
        response_mask: [batch_size, seq_len] response mask.
        omega: Weight coefficient for step-level advantage.

    Returns:
        Fused advantages: [batch_size, seq_len].
    """
    batch_size, seq_len = episode_advantages.shape

    # Build token-level step advantages
    step_adv_token_level = np.zeros((batch_size, seq_len), dtype=np.float32)

    for i, steps in enumerate(step_advantages_per_sample):
        if not steps:
            continue
        for step_info in steps:
            adv = step_info.get("step_advantage", 0.0)
            t_start = step_info.get("token_start", 0)
            t_end = step_info.get("token_end", seq_len)
            t_end = min(t_end, seq_len)
            t_start = max(t_start, 0)
            step_adv_token_level[i, t_start:t_end] = adv

    # Independent normalization (scale alignment)
    # Normalize A_E
    ae_values = episode_advantages[response_mask > 0]
    if len(ae_values) > 0:
        ae_std = np.std(ae_values)
        if ae_std > 1e-8:
            episode_advantages_norm = episode_advantages / ae_std
        else:
            episode_advantages_norm = episode_advantages
    else:
        episode_advantages_norm = episode_advantages

    # Normalize A_S
    as_values = step_adv_token_level[response_mask > 0]
    if len(as_values) > 0:
        as_std = np.std(as_values)
        if as_std > 1e-8:
            step_adv_norm = step_adv_token_level / as_std
        else:
            step_adv_norm = step_adv_token_level
    else:
        step_adv_norm = step_adv_token_level

    # Fuse: A = A_E_norm + omega * A_S_norm
    fused = episode_advantages_norm + omega * step_adv_norm

    return fused


# ============================================================
# Phase A: Step extraction + state evaluation + monotonicity
# (Does NOT depend on outcome_rewards — can run in parallel with C-GRPO)
# ============================================================
def compute_process_reward_phase_a(
    trajectories: List[str],
    questions: List[str],
    rubrics_list: List[List[str]],
    uid_list: List[str],
    model: str = "gpt-4o",
    num_workers: int = 64,
    timeout: float = 600.0,
    api_provider: str = "vendor_a",
    api_key: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    global_timeout: float = 600.0,
) -> Dict[str, Any]:
    """Phase A of process reward: step extraction, state evaluation, monotonicity.

    This phase does NOT depend on outcome_rewards, so it can run in parallel
    with C-GRPO scoring to reduce wall-clock time.

    Args:
        trajectories: List of decoded trajectory strings (one per sample).
        questions: List of question strings (one per sample).
        rubrics_list: List of rubric lists per sample.
        uid_list: List of UIDs for grouping trajectories by query.
        model: Judge model for state evaluation.
        num_workers: Number of concurrent workers for API calls.
        timeout: Timeout for API calls.
        api_provider: API provider ('vendor_a' or 'vendor_b').
        api_key: API key override.
        fallback_models: List of fallback model names for API failover.
        global_timeout: Global timeout in seconds for each API call.

    Returns:
        Dict containing:
          - 'all_steps_per_sample': List of step lists per sample
          - 'corrected_state_vectors_per_sample': Monotonicity-corrected state vectors
          - 'batch_size': Number of samples
    """
    batch_size = len(trajectories)

    if batch_size == 0:
        return {
            "all_steps_per_sample": [],
            "corrected_state_vectors_per_sample": [],
            "batch_size": 0,
        }

    # ---- Step 1: Extract steps from each trajectory ----
    all_steps_per_sample = []
    for i, traj in enumerate(trajectories):
        try:
            steps = extract_steps_from_trajectory(traj)
            all_steps_per_sample.append(steps)
        except Exception as e:
            logger.error(f"Sample {i}: step extraction failed: {e}")
            all_steps_per_sample.append(
                [{"step_index": 0, "think": "", "action": "", "response": "",
                  "cumulative_text": traj, "formatted_cumulative_text": "",
                  "is_final": True}]
            )

    # ---- Step 2: Evaluate state vectors for all steps (parallel) ----
    eval_tasks = []
    for i in range(batch_size):
        rubrics = rubrics_list[i] if i < len(rubrics_list) else []
        question = questions[i] if i < len(questions) else ""
        if not rubrics:
            continue
        for step in all_steps_per_sample[i]:
            eval_tasks.append(
                {
                    "sample_idx": i,
                    "step_idx": step["step_index"],
                    "question": question,
                    "rubrics": rubrics,
                    "cumulative_text": step["formatted_cumulative_text"],
                }
            )

    raw_state_vectors = {}
    if eval_tasks:
        try:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(eval_tasks))) as executor:
                futures = {}
                for task_idx, task in enumerate(eval_tasks):
                    future = executor.submit(
                        evaluate_step_state,
                        question=task["question"],
                        rubrics=task["rubrics"],
                        cumulative_text=task["cumulative_text"],
                        model=model,
                        token_index=task_idx,
                        api_provider=api_provider,
                        api_key=api_key,
                        fallback_models=fallback_models,
                        global_timeout=global_timeout,
                    )
                    futures[future] = (task["sample_idx"], task["step_idx"])

                for future in futures:
                    sample_idx, step_idx = futures[future]
                    try:
                        sv = future.result(timeout=timeout)
                        raw_state_vectors[(sample_idx, step_idx)] = sv
                    except Exception as e:
                        logger.error(
                            f"Sample {sample_idx} step {step_idx}: API eval failed: {e}"
                        )
                        num_rubrics = len(rubrics_list[sample_idx]) if sample_idx < len(rubrics_list) else 0
                        raw_state_vectors[(sample_idx, step_idx)] = [0] * num_rubrics
        except Exception as e:
            logger.error(f"ThreadPoolExecutor failed in phase_a: {e}")
            logger.error(traceback.format_exc())

    # ---- Step 3: Monotonicity correction (per trajectory) ----
    corrected_state_vectors_per_sample = []
    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        rubrics = rubrics_list[i] if i < len(rubrics_list) else []

        if not rubrics:
            corrected_state_vectors_per_sample.append(
                [[]] * len(steps)
            )
            continue

        raw_svs = []
        for step in steps:
            sv = raw_state_vectors.get(
                (i, step["step_index"]),
                [0] * len(rubrics),
            )
            raw_svs.append(sv)

        corrected = enforce_monotonicity(raw_svs)
        corrected_state_vectors_per_sample.append(corrected)

    logger.info(
        f"[Phase A] Completed: batch_size={batch_size}, "
        f"eval_tasks={len(eval_tasks)}, "
        f"state_vectors={len(raw_state_vectors)}"
    )

    return {
        "all_steps_per_sample": all_steps_per_sample,
        "corrected_state_vectors_per_sample": corrected_state_vectors_per_sample,
        "batch_size": batch_size,
    }


# ============================================================
# Incremental Phase A Evaluator
# (Allows submitting samples one-by-one as rollout completes them,
#  instead of waiting for all samples to finish before starting.)
# ============================================================

class IncrementalPhaseAEvaluator:
    """Incremental Phase A evaluator that processes samples as they arrive.

    Instead of waiting for all rollout samples to complete before starting
    Phase A evaluation, this class allows submitting individual samples
    as soon as their rollout finishes. This overlaps Phase A computation
    with the remaining rollout generation, potentially saving 40-50 minutes
    per training step.

    Usage:
        evaluator = IncrementalPhaseAEvaluator(
            batch_size=640, model="deepseek-v3.2", num_workers=64, ...
        )
        # As each rollout request completes:
        evaluator.submit(sample_idx=i, trajectory=text, question=q, rubrics=r)
        # After all rollout requests complete:
        result = evaluator.collect()
    """

    def __init__(
        self,
        batch_size: int,
        model: str = "gpt-4o",
        num_workers: int = 64,
        timeout: float = 600.0,
        api_provider: str = "vendor_a",
        api_key: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        global_timeout: float = 600.0,
    ):
        self.batch_size = batch_size
        self.model = model
        self.num_workers = num_workers
        self.timeout = timeout
        self.api_provider = api_provider
        self.api_key = api_key
        self.fallback_models = fallback_models
        self.global_timeout = global_timeout

        # Shared thread pool for all API calls
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        # Per-sample storage (thread-safe via GIL for dict/list assignment)
        self._all_steps: Dict[int, List[Dict[str, Any]]] = {}
        self._rubrics_count: Dict[int, int] = {}  # sample_idx -> num_rubrics
        self._eval_futures: Dict[Tuple[int, int], Any] = {}  # (sample_idx, step_idx) -> Future
        self._submitted_count = 0
        self._start_time = time.time()

        logger.info(
            f"[IncrementalPhaseA] Initialized: batch_size={batch_size}, "
            f"model={model}, workers={num_workers}"
        )

    def submit(
        self,
        sample_idx: int,
        trajectory: str,
        question: str,
        rubrics: List[str],
    ) -> None:
        """Submit a single sample for Phase A evaluation.

        This method extracts steps from the trajectory and immediately
        submits API evaluation tasks for each step.

        Args:
            sample_idx: Index of this sample in the batch (0-based).
            trajectory: Decoded response text from rollout.
            question: The original question string.
            rubrics: List of rubric criteria strings.
        """
        # Step 1: Extract steps from trajectory
        try:
            steps = extract_steps_from_trajectory(trajectory)
        except Exception as e:
            logger.error(f"[IncrementalPhaseA] Sample {sample_idx}: step extraction failed: {e}")
            steps = [{
                "step_index": 0, "think": "", "action": "", "response": "",
                "cumulative_text": trajectory, "formatted_cumulative_text": "",
                "is_final": True,
            }]

        self._all_steps[sample_idx] = steps
        self._rubrics_count[sample_idx] = len(rubrics)

        # Step 2: Submit evaluation tasks for each step
        if rubrics:
            for step in steps:
                future = self._executor.submit(
                    evaluate_step_state,
                    question=question,
                    rubrics=rubrics,
                    cumulative_text=step["formatted_cumulative_text"],
                    model=self.model,
                    token_index=sample_idx,
                    api_provider=self.api_provider,
                    api_key=self.api_key,
                    fallback_models=self.fallback_models,
                    global_timeout=self.global_timeout,
                )
                self._eval_futures[(sample_idx, step["step_index"])] = future

        self._submitted_count += 1

    def collect(self) -> Dict[str, Any]:
        """Collect all Phase A results, waiting for pending evaluations.

        Returns:
            Dict with same format as compute_process_reward_phase_a():
              - 'all_steps_per_sample': List of step lists per sample
              - 'corrected_state_vectors_per_sample': Monotonicity-corrected state vectors
              - 'batch_size': Number of samples
        """
        collect_start = time.time()

        # Wait for all evaluation futures to complete
        raw_state_vectors = {}
        for (sample_idx, step_idx), future in self._eval_futures.items():
            try:
                sv = future.result(timeout=self.timeout)
                raw_state_vectors[(sample_idx, step_idx)] = sv
            except Exception as e:
                logger.error(
                    f"[IncrementalPhaseA] Sample {sample_idx} step {step_idx}: "
                    f"API eval failed: {e}"
                )
                # Match serial version: use [0]*num_rubrics on failure, not []
                num_rubrics = self._rubrics_count.get(sample_idx, 0)
                raw_state_vectors[(sample_idx, step_idx)] = [0] * num_rubrics

        # Build ordered results for all samples
        all_steps_per_sample = []
        corrected_state_vectors_per_sample = []

        for i in range(self.batch_size):
            steps = self._all_steps.get(i, [{
                "step_index": 0, "think": "", "action": "", "response": "",
                "cumulative_text": "", "formatted_cumulative_text": "",
                "is_final": True,
            }])
            all_steps_per_sample.append(steps)

            # Collect raw state vectors for this sample
            num_rubrics = self._rubrics_count.get(i, 0)
            raw_svs = []
            for step in steps:
                # Match serial version: default to [0]*num_rubrics for missing entries
                sv = raw_state_vectors.get(
                    (i, step["step_index"]),
                    [0] * num_rubrics if num_rubrics > 0 else [],
                )
                raw_svs.append(sv)

            # Apply monotonicity correction
            # Match serial version: if no rubrics, use [[]] * len(steps)
            if num_rubrics > 0 and raw_svs:
                corrected = enforce_monotonicity(raw_svs)
            else:
                corrected = [[]] * len(steps)
            corrected_state_vectors_per_sample.append(corrected)

        # Shutdown executor
        self._executor.shutdown(wait=False)

        elapsed = time.time() - self._start_time
        collect_elapsed = time.time() - collect_start
        # Use print instead of logger.info to ensure visibility in Ray worker logs
        print(
            f"[IncrementalPhaseA] Completed: batch_size={self.batch_size}, "
            f"submitted={self._submitted_count}, "
            f"eval_tasks={len(self._eval_futures)}, "
            f"total_elapsed={elapsed:.1f}s, "
            f"collect_wait={collect_elapsed:.1f}s"
        )
        # Count samples with non-empty state vectors
        _has_sv = sum(1 for svs in corrected_state_vectors_per_sample
                      if any(len(sv) > 0 for sv in svs))
        print(
            f"[IncrementalPhaseA] State vectors: "
            f"{_has_sv}/{self.batch_size} samples have non-empty state vectors"
        )

        return {
            "all_steps_per_sample": all_steps_per_sample,
            "corrected_state_vectors_per_sample": corrected_state_vectors_per_sample,
            "batch_size": self.batch_size,
        }


# ============================================================
# Phase B: Immediate rewards + step advantages + metrics
# (Depends on outcome_rewards from C-GRPO and phase_a result)
# ============================================================
def compute_process_reward_phase_b(
    phase_a_result: Dict[str, Any],
    trajectories: List[str],
    questions: List[str],
    rubrics_list: List[List[str]],
    outcome_rewards: List[float],
    uid_list: List[str],
    gamma: float = 0.95,
    global_step: int = 0,
    dump_grouping: bool = False,
    dump_dir: str = "",
    dump_max_queries: int = 0,
    dump_step_text_max_chars: int = 0,
    omega: float = 0.5,
    rank: int = 0,
) -> Dict[str, Any]:
    """Phase B of process reward: immediate rewards, step advantages, metrics.

    Depends on outcome_rewards (from C-GRPO) and phase_a_result (from phase A).

    Args:
        phase_a_result: Dict from compute_process_reward_phase_a containing
            all_steps_per_sample and corrected_state_vectors_per_sample.
        trajectories: List of decoded trajectory strings.
        questions: List of question strings.
        rubrics_list: List of rubric lists per sample.
        outcome_rewards: List of outcome rewards (0 or 1) per sample.
        uid_list: List of UIDs for grouping.
        gamma: Discount factor.
        global_step: Current training step.
        dump_grouping: Whether to dump grouping details.
        dump_dir: Directory for dump files.
        dump_max_queries: Max queries to dump.
        dump_step_text_max_chars: Max chars for step text preview.
        omega: Weight coefficient for step-level advantage.
        rank: Current process rank.

    Returns:
        Dict containing:
          - 'step_advantages_per_sample': List of step info lists per sample
          - 'process_reward_metrics': Dict of monitoring metrics
          - 'group_details': List of group dicts for offline inspection
    """
    all_steps_per_sample = phase_a_result["all_steps_per_sample"]
    corrected_state_vectors_per_sample = phase_a_result["corrected_state_vectors_per_sample"]
    batch_size = phase_a_result["batch_size"]

    if batch_size == 0:
        return {
            "step_advantages_per_sample": [],
            "process_reward_metrics": {},
        }

    # ---- Step 4: Compute immediate rewards & collect step data ----
    all_step_data = []
    step_data_per_sample = []

    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        corrected_svs = corrected_state_vectors_per_sample[i]
        outcome_r = outcome_rewards[i] if i < len(outcome_rewards) else 0.0

        immediate = compute_immediate_rewards(corrected_svs, outcome_r)

        sample_step_data = []
        num_rubrics = len(corrected_svs[0]) if corrected_svs and corrected_svs[0] else 0
        for j, step in enumerate(steps):
            if j == 0:
                pre_action_state = [0] * num_rubrics
            else:
                pre_action_state = corrected_svs[j - 1] if (j - 1) < len(corrected_svs) else []

            post_action_state = corrected_svs[j] if j < len(corrected_svs) else pre_action_state
            state_transition = _compute_state_transition(pre_action_state, post_action_state)

            step_text_parts = []
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

            sd = {
                "trajectory_index": i,
                "step_index": step["step_index"],
                "corrected_state": pre_action_state,
                "post_action_state": post_action_state,
                "state_transition": state_transition,
                "immediate_reward": immediate[j] if j < len(immediate) else 0.0,
                "is_final": step["is_final"],
                "uid": uid_list[i] if i < len(uid_list) else "",
                "step_text": step_text,
                "step_input": step_input,
            }
            all_step_data.append(sd)
            sample_step_data.append(sd)

        step_data_per_sample.append(sample_step_data)

    # ---- Step 5: Group by (uid, state), normalize, discounted returns ----
    all_step_data, step_level_metrics, group_details = compute_step_level_advantages(all_step_data, gamma=gamma)

    # ---- Step 6: Compute token boundaries for each step ----
    for i in range(batch_size):
        traj = trajectories[i]
        steps = all_steps_per_sample[i]
        sample_steps = step_data_per_sample[i]

        total_len = len(traj)
        for j, (step, sd) in enumerate(zip(steps, sample_steps)):
            cum_len = len(step["cumulative_text"])
            if j == 0:
                sd["char_start"] = 0
            else:
                prev_cum_len = len(steps[j - 1]["cumulative_text"])
                sd["char_start"] = prev_cum_len
            sd["char_end"] = cum_len

    # ---- Collect metrics ----
    total_steps = len(all_step_data)
    non_zero_advantages = [
        sd["step_advantage"]
        for sd in all_step_data
        if sd.get("step_advantage", 0.0) != 0.0
    ]

    metrics = {
        **step_level_metrics,
        "process_reward/total_steps": total_steps,
        "process_reward/avg_steps_per_trajectory": total_steps / max(batch_size, 1),
        "process_reward/non_zero_advantage_count": len(non_zero_advantages),
        "process_reward/mean_step_advantage": float(np.mean(non_zero_advantages))
        if non_zero_advantages
        else 0.0,
        "process_reward/std_step_advantage": float(np.std(non_zero_advantages))
        if non_zero_advantages
        else 0.0,
    }

    logger.info(
        f"[Phase B] batch_size={batch_size}, total_steps={total_steps}, "
        f"num_groups={step_level_metrics.get('gig/num_state_groups', 0)}, "
        f"non_zero_adv={len(non_zero_advantages)}"
    )

    # ---- Dump grouping details if enabled ----
    if dump_grouping and group_details:
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

    return {
        "step_advantages_per_sample": step_data_per_sample,
        "process_reward_metrics": metrics,
        "group_details": group_details,
    }


# ============================================================
# High-level orchestration: Process all trajectories for a batch
# ============================================================
def compute_process_reward_for_batch(
    trajectories: List[str],
    questions: List[str],
    rubrics_list: List[List[str]],
    outcome_rewards: List[float],
    uid_list: List[str],
    gamma: float = 0.95,
    model: str = "gpt-4o",
    num_workers: int = 64,
    timeout: float = 600.0,
    global_step: int = 0,
    dump_grouping: bool = False,
    dump_dir: str = "",
    dump_max_queries: int = 0,
    dump_step_text_max_chars: int = 0,
    omega: float = 0.5,
    rank: int = 0,
    api_provider: str = "vendor_a",
    api_key: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    global_timeout: float = 600.0,
) -> Dict[str, Any]:
    """Compute process rewards and step-level advantages for an entire batch.

    This is the main entry point called from the reward manager.

    Args:
        trajectories: List of decoded trajectory strings (one per sample).
        questions: List of question strings (one per sample).
        rubrics_list: List of rubric lists. Each rubric list contains
                      constraints_gt strings for the corresponding query.
        outcome_rewards: List of outcome rewards (0 or 1) per sample.
        uid_list: List of UIDs for grouping trajectories by query.
        gamma: Discount factor for process reward.
        model: Judge model for state evaluation.
        num_workers: Number of concurrent workers for API calls.
        timeout: Timeout for API calls.
        global_step: Current training step (for dump filename).
        dump_grouping: Whether to dump grouping details to file.
        dump_dir: Directory to write grouping detail files.
        dump_max_queries: Max number of unique uids to dump (0 = all).
        dump_step_text_max_chars: Max chars for step_text_preview (0 = full).
        omega: Weight coefficient for step-level advantage (for batch_meta).
        rank: Current process rank (for dump filename).
        fallback_models: List of fallback model names for API failover.
        global_timeout: Global timeout in seconds for each API call.

    Returns:
        Dict containing:
          - 'step_advantages_per_sample': List of step info lists per sample
          - 'process_reward_metrics': Dict of monitoring metrics
          - 'group_details': List of group dicts for offline inspection
    """
    # Backward-compatible entry point: delegates to phase_a + phase_b.
    # For parallel execution with C-GRPO, call phase_a and phase_b separately.
    phase_a_result = compute_process_reward_phase_a(
        trajectories=trajectories,
        questions=questions,
        rubrics_list=rubrics_list,
        uid_list=uid_list,
        model=model,
        num_workers=num_workers,
        timeout=timeout,
        api_provider=api_provider,
        api_key=api_key,
        fallback_models=fallback_models,
        global_timeout=global_timeout,
    )

    return compute_process_reward_phase_b(
        phase_a_result=phase_a_result,
        trajectories=trajectories,
        questions=questions,
        rubrics_list=rubrics_list,
        outcome_rewards=outcome_rewards,
        uid_list=uid_list,
        gamma=gamma,
        global_step=global_step,
        dump_grouping=dump_grouping,
        dump_dir=dump_dir,
        dump_max_queries=dump_max_queries,
        dump_step_text_max_chars=dump_step_text_max_chars,
        omega=omega,
        rank=rank,
    )


def map_step_advantages_to_tokens(
    step_data_per_sample: List[List[Dict[str, Any]]],
    trajectories: List[str],
    response_length: int,
    tokenizer: Any = None,
) -> List[List[Dict[str, Any]]]:
    """Map step advantages to token-level positions.

    Converts character-level boundaries to approximate token-level boundaries
    using a simple heuristic (chars / avg_chars_per_token).

    If tokenizer is provided, uses it for precise mapping.

    Args:
        step_data_per_sample: Per-sample step data with char_start/char_end.
        trajectories: Decoded trajectory strings.
        response_length: The response tensor length (number of tokens).
        tokenizer: Optional tokenizer for precise char-to-token mapping.

    Returns:
        Updated step_data_per_sample with token_start and token_end fields.
    """
    for i, (steps, traj) in enumerate(zip(step_data_per_sample, trajectories)):
        total_chars = len(traj) if traj else 1
        # Simple proportional mapping
        for sd in steps:
            char_start = sd.get("char_start", 0)
            char_end = sd.get("char_end", total_chars)
            sd["token_start"] = int(char_start / total_chars * response_length)
            sd["token_end"] = int(char_end / total_chars * response_length)
            # Clamp
            sd["token_start"] = max(0, min(sd["token_start"], response_length - 1))
            sd["token_end"] = max(sd["token_start"] + 1, min(sd["token_end"], response_length))

    return step_data_per_sample


def _compute_state_transition(
    pre_state: List[int], post_state: List[int]
) -> str:
    """Compute human-readable state transition description.

    Args:
        pre_state: State vector before the action.
        post_state: State vector after the action.

    Returns:
        Human-readable string like '+C1, +C3' or '(none)'.
    """
    if not pre_state or not post_state:
        return "(none)"
    changes = []
    for i, (p, c) in enumerate(zip(pre_state, post_state)):
        if p == 0 and c == 1:
            changes.append(f"+C{i+1}")
    return ", ".join(changes) if changes else "(none)"


def dump_grouping_details(
    group_details: List[Dict[str, Any]],
    step_data_per_sample: List[List[Dict[str, Any]]],
    questions: Optional[List[str]] = None,
    rubrics_list: Optional[List[List[str]]] = None,
    outcome_rewards: Optional[List[float]] = None,
    uid_list: Optional[List[str]] = None,
    global_step: int = 0,
    rank: int = 0,
    dump_dir: str = "",
    gamma: float = 0.95,
    dump_max_queries: int = 0,
    batch_size: int = 0,
    dump_step_text_max_chars: int = 0,
    omega: float = 0.5,
) -> None:
    """Dump query-centric grouping details to a JSONL file for offline inspection.
    New format (V2): One batch per file, query-centric structure.
    Line 1: batch_meta
    Lines 2..N+1: one JSON per sampled query, containing trajectories + groups.

    Args:
        group_details: List of group dicts from compute_step_level_advantages.
        step_data_per_sample: Per-sample step data with all computed fields.
        questions: List of question strings indexed by sample position.
        rubrics_list: List of rubric lists per sample.
        outcome_rewards: List of outcome rewards per sample.
        uid_list: List of uid strings per sample.
        global_step: Current training step (used in filename).
        rank: Current process rank (used in filename).
        dump_dir: Directory to write output files.
        dump_max_queries: Max number of unique uids to dump (0 = all).
        gamma: Discount factor used.
        batch_size: Total batch size.
        dump_step_text_max_chars: Max chars for step_text_preview (0 = full text).
        omega: Weight coefficient for step-level advantage.
    """
    if not dump_dir or not group_details:
        return

    try:
        os.makedirs(dump_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grouping_step{global_step:06d}_rank{rank}_{timestamp}.jsonl"
        filepath = os.path.join(dump_dir, filename)

        # Ensure lists are available
        questions = questions or []
        rubrics_list = rubrics_list or []
        outcome_rewards = outcome_rewards or []
        uid_list = uid_list or []

        # Build uid -> sample indices mapping
        uid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, uid in enumerate(uid_list):
            if uid:
                uid_to_indices[uid].append(idx)

        # Determine unique uids in batch
        all_unique_uids = list(dict.fromkeys(uid for uid in uid_list if uid))

        # Select uids to dump
        if dump_max_queries > 0:
            sampled_uids = all_unique_uids[:dump_max_queries]
        else:
            sampled_uids = all_unique_uids

        # Build uid -> question mapping
        uid_to_question: Dict[str, str] = {}
        uid_to_rubrics: Dict[str, List[str]] = {}
        for idx, uid in enumerate(uid_list):
            if uid and uid not in uid_to_question:
                uid_to_question[uid] = questions[idx] if idx < len(questions) else ""
                uid_to_rubrics[uid] = rubrics_list[idx] if idx < len(rubrics_list) else []

        # Build uid -> group_details mapping
        uid_to_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for gd in group_details:
            uid_to_groups[gd["uid"]].append(gd)

        # Helper: truncate step text
        def _truncate(text: str) -> str:
            if dump_step_text_max_chars <= 0 or len(text) <= dump_step_text_max_chars:
                return text
            return text[:dump_step_text_max_chars] + "...(truncated)"

        with open(filepath, "w", encoding="utf-8") as f:
            # ---- Line 1: batch_meta ----
            batch_meta = {
                "type": "batch_meta",
                "global_step": global_step,
                "rank": rank,
                "timestamp": timestamp,
                "gamma": gamma,
                "batch_size": batch_size,
                "num_sampled_queries": len(sampled_uids),
                "total_unique_queries_in_batch": len(all_unique_uids),
                "omega": omega,
                "dump_step_text_max_chars": dump_step_text_max_chars,
            }
            f.write(json.dumps(batch_meta, ensure_ascii=False) + "\n")

            # ---- Lines 2..N+1: one per sampled query ----
            for uid in sampled_uids:
                sample_indices = uid_to_indices.get(uid, [])
                question = uid_to_question.get(uid, "")
                rubrics = uid_to_rubrics.get(uid, [])

                # Build trajectories array
                trajectories_data = []
                num_correct = 0
                all_rubrics_satisfied_at_end = []

                for traj_idx in sample_indices:
                    steps = step_data_per_sample[traj_idx] if traj_idx < len(step_data_per_sample) else []
                    traj_outcome = outcome_rewards[traj_idx] if traj_idx < len(outcome_rewards) else 0.0
                    if traj_outcome >= 1.0:
                        num_correct += 1

                    # Build final state from last step's post_action_state
                    final_state = []
                    if steps:
                        final_state = steps[-1].get("post_action_state", steps[-1].get("corrected_state", []))
                    num_rubrics_val = len(final_state)
                    satisfied_at_end = sum(final_state) if final_state else 0
                    all_rubrics_satisfied_at_end.append(satisfied_at_end)

                    final_state_summary = f"{satisfied_at_end}/{num_rubrics_val} satisfied"

                    # Build step records
                    step_records = []
                    for sd in steps:
                        step_record = {
                            "step_index": sd["step_index"],
                            "is_final": sd["is_final"],
                            "pre_action_state": sd.get("corrected_state", []),
                            "post_action_state": sd.get("post_action_state", []),
                            "state_transition": sd.get("state_transition", "(none)"),
                            "immediate_reward": float(sd.get("immediate_reward", 0.0)),
                            "normalized_reward": float(sd.get("normalized_reward", 0.0)),
                            "step_advantage": float(sd.get("step_advantage", 0.0)),
                            "group_id": sd.get("group_id"),
                            "step_input": sd.get("step_input", ""),
                            "step_text_preview": _truncate(sd.get("step_text", "")),
                        }
                        step_records.append(step_record)

                    traj_record = {
                        "trajectory_index": traj_idx,
                        "num_steps": len(steps),
                        "outcome_reward": traj_outcome,
                        "final_state": final_state,
                        "final_state_summary": final_state_summary,
                        "steps": step_records,
                    }
                    trajectories_data.append(traj_record)

                # Build groups array for this uid (deep copy to avoid mutation)
                # Also enrich members with total_steps_in_trajectory context
                uid_groups_raw = uid_to_groups.get(uid, [])
                uid_groups = []
                for g in uid_groups_raw:
                    g_copy = dict(g)  # shallow copy of group dict
                    enriched_members = []
                    for member in g.get("members", []):
                        m = dict(member)
                        ti = m.get("trajectory_index", -1)
                        total_steps = len(step_data_per_sample[ti]) if 0 <= ti < len(step_data_per_sample) else 0
                        m["total_steps_in_trajectory"] = total_steps
                        enriched_members.append(m)
                    g_copy["members"] = enriched_members
                    uid_groups.append(g_copy)

                # Build query_summary
                num_trajectories = len(sample_indices)
                total_steps = sum(len(step_data_per_sample[ti]) for ti in sample_indices if ti < len(step_data_per_sample))
                non_final_steps = sum(
                    1 for ti in sample_indices if ti < len(step_data_per_sample)
                    for sd in step_data_per_sample[ti] if not sd.get("is_final", False)
                )
                num_groups = len(uid_groups)
                num_effective = sum(1 for g in uid_groups if g.get("is_effective", False))
                num_singleton = sum(1 for g in uid_groups if g.get("group_size", 0) == 1)
                group_sizes = [g.get("group_size", 0) for g in uid_groups]

                query_summary = {
                    "num_trajectories": num_trajectories,
                    "num_total_steps": total_steps,
                    "num_non_final_steps": non_final_steps,
                    "num_groups": num_groups,
                    "num_effective_groups": num_effective,
                    "effective_group_ratio": round(num_effective / max(num_groups, 1), 3),
                    "num_singleton_groups": num_singleton,
                    "avg_group_size": round(float(np.mean(group_sizes)), 2) if group_sizes else 0.0,
                    "max_group_size": int(max(group_sizes)) if group_sizes else 0,
                    "avg_steps_per_trajectory": round(total_steps / max(num_trajectories, 1), 1),
                    "avg_rubrics_satisfied_at_end": round(
                        float(np.mean(all_rubrics_satisfied_at_end)), 1
                    ) if all_rubrics_satisfied_at_end else 0.0,
                    "max_rubrics_satisfied_at_end": int(max(all_rubrics_satisfied_at_end)) if all_rubrics_satisfied_at_end else 0,
                    "num_trajectories_correct": num_correct,
                }

                # Compute outcome distribution across trajectories
                traj_outcomes = [
                    outcome_rewards[ti] if ti < len(outcome_rewards) else 0.0
                    for ti in sample_indices
                ]
                outcome_dist: Dict[str, int] = {}
                for o in traj_outcomes:
                    key = str(float(o))
                    outcome_dist[key] = outcome_dist.get(key, 0) + 1

                query_record = {
                    "type": "query",
                    "uid": uid,
                    "question": question,
                    "outcome_reward_distribution": outcome_dist,
                    "num_rubrics": len(rubrics),
                    "rubrics": rubrics,
                    "num_trajectories": num_trajectories,
                    "trajectories": trajectories_data,
                    "num_groups": num_groups,
                    "groups": uid_groups,
                    "query_summary": query_summary,
                }
                f.write(json.dumps(query_record, ensure_ascii=False) + "\n")

        logger.info(
            f"[GiG Dump V2] Wrote {len(sampled_uids)} queries to {filepath}"
        )
    except Exception as e:
        logger.error(f"[GiG Dump] Failed to dump grouping details: {e}")
        logger.error(traceback.format_exc())


def build_rubrics_from_constraints_gt(
    constraints_gt: List[str],
    entities_gt: Dict[str, str],
) -> List[str]:
    """Build rubric strings from constraints_gt and entities_gt.

    The constraints_gt already contains entity names substituted in brackets.
    This function ensures they are clean rubric strings for evaluation.

    The format is: "C1. <E1>的主要产业是[太子参]<E4>种植..."
    We replace <Ek> placeholders with actual entity names from entities_gt.

    Args:
        constraints_gt: List of constraint strings with entity placeholders
                        and bracketed real names.
        entities_gt: Dict mapping entity labels to real names.

    Returns:
        List of clean rubric strings with all entities resolved.
    """
    rubrics = []
    # Handle numpy arrays from parquet
    if hasattr(constraints_gt, 'tolist'):
        constraints_gt = constraints_gt.tolist()
    for constraint in constraints_gt:
        rubric = str(constraint)
        # Replace <Ek> placeholders with actual entity names, skip None values
        # Sort keys by length descending to avoid partial replacement (e.g. <E10> vs <E1>)
        sorted_keys = sorted(entities_gt.keys(), key=len, reverse=True)
        for entity_key in sorted_keys:
            entity_value = entities_gt[entity_key]
            if entity_value is None:
                continue
            rubric = rubric.replace(f"<{entity_key}>", str(entity_value))
        # Remove bracket markers around already-substituted names: [xxx] -> xxx
        rubric = re.sub(r"\[([^\]]+)\]", r"\1", rubric)
        rubrics.append(rubric)
    return rubrics
