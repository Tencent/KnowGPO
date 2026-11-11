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
import re
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# API tokens for external model calls
# ============================================================
TOKENS = [
    "REPLACE_WITH_YOUR_API_KEY",
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
) -> Tuple[Optional[str], Optional[str]]:
    """Generic model API call function."""
    if tokens is None:
        tokens = TOKENS

    if model in ("deepseek-v3-0324", "deepseek-r1"):
        url = "https://api.lkeap.cloud.anonymous_org.com/v1/chat/completions"
        auth_token = "REPLACE_WITH_YOUR_API_KEY"
    else:
        if tokens is None:
            raise Exception("No tokens provided")
        url = "https://api.example.com/v1/chat/completions"
        token = tokens[token_index % len(tokens)]
        auth_token = token

    # Models that support extended thinking (Claude series only)
    # NOTE: gemini-2.0-flash does NOT support extended thinking via vendor_a API,
    #       and its max output tokens is 8192.
    thinking_models = {"claude-4-sonnet-20250514","gemini-2.0-flash"}
    if model in thinking_models:
        payload = {
            "model": model,
            "messages": messages,
            # "thinking_enabled": True,
            # "thinking_tokens": 4096,
            "max_tokens": 8192,
        }
    elif model in ("gpt-4o-mini"):
        # Gemini models: vendor_a API supports max 8192 output tokens
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 16384,
        }
    else:
        # gpt-4o and other models: max_tokens capped at 4096, no thinking params
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
        }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=600
            )
            if response.status_code != 200:
                raise Exception(
                    f"Request failed, status: {response.status_code}. Response: {response.text}"
                )
            response_data = response.json()
            result_text = response_data["choices"][0]["message"]["content"]
            reason_text = response_data["choices"][0]["message"].get(
                "reasoning_content", ""
            )
            return result_text, reason_text

        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {retry_count + 1}/{max_retries} failed: TIMEOUT (model={model})")
            retry_count += 1
            sleep_time = min(2**retry_count, 60)
            time.sleep(sleep_time)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                err_type = "RATE_LIMITED"
            elif "timeout" in error_str.lower():
                err_type = "TIMEOUT"
            else:
                err_type = "ERROR"
            logger.warning(
                f"Attempt {retry_count + 1}/{max_retries} [{err_type}] "
                f"(model={model}): {error_str[:200]}"
            )
            retry_count += 1
            sleep_time = min(2**retry_count, 60)
            time.sleep(sleep_time)

    logger.warning(f"All {max_retries} retries exhausted for model={model}, returning None")
    return None, None


# ============================================================
# Module 1: Step Extraction
# ============================================================
def extract_steps_from_trajectory(trajectory_text: str) -> List[Dict[str, Any]]:
    """Extract steps (assistant turns) from a rollout trajectory.

    Each step = one assistant turn (LLM generation -> tool_call -> tool_response).

    Args:
        trajectory_text: The decoded response text from rollout.

    Returns:
        List of step dicts, each containing:
          - step_index: int
          - cumulative_text: str (all text up to and including this step)
          - is_final: bool (True if this is the last step / final_answer)
    """
    steps = []

    # Split by assistant turns using <tool_call> and <tool_response> markers
    # Pattern: assistant generates text (possibly with <think>...</think>)
    # then <tool_call>...</tool_call> followed by <tool_response>...</tool_response>
    # Final step may contain <answer>...</answer> instead

    # Strategy: split on tool_response closing tags to identify turn boundaries
    # Each segment ending with </tool_response> is one intermediate step
    # The remaining text after the last </tool_response> is the final step

    parts = re.split(r"(</tool_response>)", trajectory_text)

    cumulative = ""
    step_idx = 0

    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] == "</tool_response>":
            # This is an intermediate step
            cumulative += parts[i] + parts[i + 1]
            steps.append(
                {
                    "step_index": step_idx,
                    "cumulative_text": cumulative,
                    "is_final": False,
                }
            )
            step_idx += 1
            i += 2
        else:
            # Remaining text (final step or text without tool_response)
            cumulative += parts[i]
            i += 1

    # Add the final step if there's meaningful content after the last tool_response
    # or if there were no tool_responses at all
    if not steps or cumulative != steps[-1]["cumulative_text"]:
        steps.append(
            {
                "step_index": step_idx,
                "cumulative_text": cumulative,
                "is_final": True,
            }
        )
    else:
        # Mark the last step as final
        if steps:
            steps[-1]["is_final"] = True

    # Ensure at least one step exists
    if not steps:
        steps.append(
            {
                "step_index": 0,
                "cumulative_text": trajectory_text,
                "is_final": True,
            }
        )

    return steps


# ============================================================
# Module 1: Step State Evaluation via External API
# ============================================================
STEP_STATE_EVALUATION_PROMPT = """You are evaluating the progress of an AI assistant's multi-step reasoning process.

You will receive:
1. A complex question.
2. A list of rubric criteria (evaluation standards). Each rubric is a factual statement that the assistant should discover/verify during its research process. The rubric uses real entity names (not placeholders).
3. The AI assistant's cumulative response up to a certain point in its research process.

Your task: For each rubric criterion, determine whether the assistant's response (up to this point) has discovered or verified the factual information described in that rubric.

Output a JSON object mapping each rubric index (R1, R2, ...) to 1 (satisfied/discovered) or 0 (not yet satisfied/discovered).

IMPORTANT:
- Only mark a rubric as 1 if the information is EXPLICITLY present in the assistant's response text.
- Do NOT infer or guess. If the exact factual claim is not clearly stated, mark it as 0.
- Consider ALL text in the response, including search results and tool responses.

[Question]
{question}

[Rubric Criteria]
{rubrics}

[Assistant's Response (up to current step)]
{cumulative_text}

Output format (JSON only, no other text):
```json
{{
  "R1": 0 or 1,
  "R2": 0 or 1,
  ...
}}
```"""


def evaluate_step_state(
    question: str,
    rubrics: List[str],
    cumulative_text: str,
    model: str = "gpt-4o",
    token_index: int = 0,
) -> List[int]:
    """Evaluate the state vector for a single step via external API.

    Args:
        question: The original query.
        rubrics: List of rubric criteria strings (from constraints_gt).
        cumulative_text: The cumulative trajectory text up to this step.
        model: The judge model to use.
        token_index: Token index for API authentication.

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
        cumulative_text=cumulative_text[:8000],  # Truncate to avoid token limits
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        result_text, _ = call_model_api_with_messages(
            messages, model=model, token_index=token_index, max_retries=3
        )

        if result_text is None:
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
            # Try to find raw JSON
            json_match = re.search(r"\{[^{}]*\}", result_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning(f"No JSON found in API response: {result_text[:200]}")
                return [0] * len(rubrics)

        try:
            import json_repair
            state_dict = json_repair.loads(json_str)
        except Exception:
            state_dict = json.loads(json_str)

        # Extract state vector
        state_vector = []
        for i in range(len(rubrics)):
            key = f"R{i+1}"
            val = state_dict.get(key, 0)
            state_vector.append(1 if val == 1 or val is True else 0)

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
) -> List[Dict[str, Any]]:
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
          - 'immediate_reward': float
          - 'is_final': bool
          - 'uid': str
        gamma: Discount factor for computing discounted returns.

    Returns:
        Same list with 'normalized_reward' and 'step_advantage' added.
    """
    if not all_trajectories_step_data:
        return []

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

    # Normalize immediate rewards within each group
    eps = 1e-8
    for group_key, group_items in groups.items():
        if len(group_items) <= 1:
            for item in group_items:
                item["normalized_reward"] = 0.0
        else:
            rewards = [item["immediate_reward"] for item in group_items]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            for item in group_items:
                item["normalized_reward"] = (
                    (item["immediate_reward"] - mean_r) / max(std_r, eps)
                )

    # Final steps: set normalized_reward and step_advantage to 0.
    # The answer step's signal is handled by episode-level A_E.
    for item in final_items:
        item["normalized_reward"] = 0.0
        item["step_advantage"] = 0.0

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

    return all_trajectories_step_data


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

    Returns:
        Dict containing:
          - 'step_advantages_per_sample': List of step info lists per sample
          - 'process_reward_metrics': Dict of monitoring metrics
    """
    batch_size = len(trajectories)

    if batch_size == 0:
        return {
            "step_advantages_per_sample": [],
            "process_reward_metrics": {},
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
                [{"step_index": 0, "cumulative_text": traj, "is_final": True}]
            )

    # ---- Step 2: Evaluate state vectors for all steps (parallel) ----
    # Collect all (sample_idx, step_idx, question, rubrics, cumulative_text)
    eval_tasks = []
    for i in range(batch_size):
        rubrics = rubrics_list[i] if i < len(rubrics_list) else []
        question = questions[i] if i < len(questions) else ""
        if not rubrics:
            # No rubrics -> skip evaluation, all state vectors are empty
            continue
        for step in all_steps_per_sample[i]:
            eval_tasks.append(
                {
                    "sample_idx": i,
                    "step_idx": step["step_index"],
                    "question": question,
                    "rubrics": rubrics,
                    "cumulative_text": step["cumulative_text"],
                }
            )

    # Parallel API calls
    raw_state_vectors = {}  # (sample_idx, step_idx) -> state_vector
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
            logger.error(f"ThreadPoolExecutor failed: {e}")
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

    # ---- Step 4: Compute immediate rewards & collect step data ----
    all_step_data = []  # Flat list for grouping
    step_data_per_sample = []  # Nested list for per-sample access

    for i in range(batch_size):
        steps = all_steps_per_sample[i]
        corrected_svs = corrected_state_vectors_per_sample[i]
        outcome_r = outcome_rewards[i] if i < len(outcome_rewards) else 0.0

        # Compute immediate rewards (no discounted returns yet)
        immediate = compute_immediate_rewards(corrected_svs, outcome_r)

        sample_step_data = []
        num_rubrics = len(corrected_svs[0]) if corrected_svs and corrected_svs[0] else 0
        for j, step in enumerate(steps):
            # Use the state BEFORE this step's action (i.e., the previous step's
            # post-action state).  Step 0's pre-action state is the all-zero vector
            # because no rubric has been satisfied yet.
            if j == 0:
                pre_action_state = [0] * num_rubrics
            else:
                pre_action_state = corrected_svs[j - 1] if (j - 1) < len(corrected_svs) else []
            sd = {
                "trajectory_index": i,
                "step_index": step["step_index"],
                "corrected_state": pre_action_state,
                "immediate_reward": immediate[j] if j < len(immediate) else 0.0,
                "is_final": step["is_final"],
                "uid": uid_list[i] if i < len(uid_list) else "",
            }
            all_step_data.append(sd)
            sample_step_data.append(sd)

        step_data_per_sample.append(sample_step_data)

    # ---- Step 5: Group by (uid, state), normalize immediate rewards,
    #              then compute discounted returns as step advantages ----
    all_step_data = compute_step_level_advantages(all_step_data, gamma=gamma)

    # ---- Step 6: Compute token boundaries for each step ----
    # We need to map steps to token positions in the response
    for i in range(batch_size):
        traj = trajectories[i]
        steps = all_steps_per_sample[i]
        sample_steps = step_data_per_sample[i]

        # Compute approximate token boundaries using character positions
        # (Token-level precision is hard without tokenizer; char-level is acceptable)
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
    num_groups = len(
        set(
            (sd.get("uid", ""), tuple(sd["corrected_state"]))
            for sd in all_step_data
            if sd["corrected_state"] and not sd.get("is_final", False)
        )
    )

    metrics = {
        "process_reward/total_steps": total_steps,
        "process_reward/avg_steps_per_trajectory": total_steps / max(batch_size, 1),
        "process_reward/num_state_groups": num_groups,
        "process_reward/non_zero_advantage_count": len(non_zero_advantages),
        "process_reward/mean_step_advantage": float(np.mean(non_zero_advantages))
        if non_zero_advantages
        else 0.0,
        "process_reward/std_step_advantage": float(np.std(non_zero_advantages))
        if non_zero_advantages
        else 0.0,
    }

    logger.info(
        f"[Process Reward] batch_size={batch_size}, total_steps={total_steps}, "
        f"num_groups={num_groups}, non_zero_adv={len(non_zero_advantages)}"
    )

    return {
        "step_advantages_per_sample": step_data_per_sample,
        "process_reward_metrics": metrics,
    }


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
