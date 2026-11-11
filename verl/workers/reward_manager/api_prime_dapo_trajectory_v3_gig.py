#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Reward Manager - V3 Group-in-Group (GiG) GRPO

Extends V2 C-GRPO with step-level process rewards and Group-in-Group advantage.

Key additions over V2:
  - Step state evaluation via external API (Module 1)
  - Monotonicity enforcement on state vectors (Module 1A)
  - Process reward computation with discounted returns (Module 2)
  - Step-level relative advantage within state groups (Module 3)
  - Scale alignment and GiG advantage fusion (Module 4)

The process reward data is stored in batch.non_tensor_batch and used by
ray_trainer.py to compute the fused Group-in-Group advantage:
  A = A_E + omega * A_S

Author: anonymous
Version: 3.0.0
Date: 2026-03-09
"""

import asyncio
import os
from pathlib import Path
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import yaml
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.browsecomp_zh import (
    compute_score as browsecomp_compute_score,
)
from verl.utils.reward_score.process_reward_gig import (
    build_rubrics_from_constraints_gt,
    compute_process_reward_for_batch,
    compute_process_reward_phase_a,
    compute_process_reward_phase_b,
    dump_grouping_details,
    map_step_advantages_to_tokens,
)
from verl.utils.reward_score.trajectory_reward_v2_cgrpo import (
    RubricRewardError,
    AbnormalTrajectoryType,
    classify_abnormal_trajectory,
    extract_constraints_and_entities_from_ground_truth,
)
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Async helper: single sample C-GRPO score (reuse from V2)
# ============================================================
async def single_compute_cgrpo_score(
    config: Dict[str, Any],
    prompts_str: str,
    sequences_str: str,
    ground_truth: Dict[str, Any],
    data_source: str,
    task_extra_info: Dict[str, Any],
    executor: ThreadPoolExecutor,
    timeout: float = 6000.0,
) -> Dict[str, Any]:
    """Compute C-GRPO reward for a single sample (async)."""
    loop = asyncio.get_running_loop()

    try:
        # Concurrent task 1: End-to-end reward (browsecomp_zh)
        async def compute_end_to_end():
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        partial(
                            browsecomp_compute_score,
                            sequences_str,
                            ground_truth,
                            task_extra_info,
                            api_config={
                                "api_provider": config.get("reward_api_provider", "venus"),
                                "api_key": config.get("reward_api_key"),
                                "grader_model_name": config.get("reward_api_model", "gpt-4o"),
                                "fallback_models": config.get("reward_api_fallback_models"),
                                "global_timeout": config.get("reward_api_global_timeout", 600.0),
                            },
                        ),
                    ),
                    timeout=timeout,
                )
                return float(result) if not isinstance(result, dict) else float(result.get("score", 0.0))
            except Exception as e:
                logger.error(f"End-to-end eval failed: {e}")
                return 0.0

        # Pipeline A: Pure GRPO outcome reward only (no rubric evaluation)
        # Rubric evaluation is removed from pipeline A to use only end-to-end result.
        e2e_score = await compute_end_to_end()

        return {
            "end_to_end_score": e2e_score,
            "rubric_reward": 0.0,
            "rubric_details": {"skipped": True, "reason": "pure_outcome_reward_mode"},
        }

    except Exception as e:
        logger.error(f"single_compute_cgrpo_score failed: {e}")
        return {
            "end_to_end_score": 0.0,
            "rubric_reward": 0.0,
            "rubric_details": {"error_code": "top_level_exception"},
        }


async def parallel_compute_cgrpo_score_async(
    config: Dict[str, Any],
    prompts_strs: List[str],
    sequences_strs: List[str],
    ground_truths: List[Dict[str, Any]],
    data_sources: List[str],
    task_extra_infos: Optional[List[Dict[str, Any]]] = None,
    num_processes: int = 64,
) -> List[Dict[str, Any]]:
    """Parallel async computation of C-GRPO rewards for a batch."""
    batch_size = len(sequences_strs)

    try:
        if task_extra_infos is None:
            task_extra_infos = [{}] * batch_size

        results = []
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            tasks_async = [
                single_compute_cgrpo_score(
                    config, ps, ss, gt, ds, tei, executor, timeout=6000.0
                )
                for ps, ss, gt, ds, tei in zip(
                    prompts_strs,
                    sequences_strs,
                    ground_truths,
                    data_sources,
                    task_extra_infos,
                )
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, (Exception, BaseException)):
                logger.error(f"Sample {i} exception: {result}")
                processed.append(
                    {
                        "end_to_end_score": 0.0,
                        "rubric_reward": 0.0,
                        "rubric_details": {"error_code": "sample_exception"},
                    }
                )
            elif not isinstance(result, dict):
                processed.append(
                    {
                        "end_to_end_score": 0.0,
                        "rubric_reward": 0.0,
                        "rubric_details": {"error_code": "invalid_type"},
                    }
                )
            else:
                result.setdefault("end_to_end_score", 0.0)
                result.setdefault("rubric_reward", 0.0)
                result.setdefault("rubric_details", {})
                processed.append(result)

        return processed

    except Exception as e:
        logger.error(f"parallel_compute_cgrpo_score_async failed: {e}")
        return [
            {
                "end_to_end_score": 0.0,
                "rubric_reward": 0.0,
                "rubric_details": {"error_code": "batch_exception"},
            }
        ] * batch_size


# ============================================================
# Main Reward Manager
# ============================================================
@register("apiprimedapotrajectoryv3gig")
class ApiPrimeDapoTrajectoryV3GiGRewardManager(AbstractRewardManager):
    """
    Group-in-Group GRPO Reward Manager (V3)

    Extends V2 C-GRPO with step-level process rewards.

    Data flow:
      1. Compute C-GRPO outcome + rubric rewards (same as V2)
      2. Extract steps from trajectories
      3. Evaluate step states via external API
      4. Enforce monotonicity on state vectors
      5. Compute process rewards (immediate + discounted)
      6. Compute step-level advantages (A_S)
      7. Store step advantage data in batch for ray_trainer to use

    The actual advantage fusion (A = A_E + omega * A_S) happens in
    ray_trainer.py, not here.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        max_resp_len: Optional[int] = None,
        overlong_buffer_cfg: Optional[Any] = None,
        reward_api_provider: str = "venus",
        reward_api_key: Optional[str] = None,
        reward_api_model: str = "gpt-4o",
        step_api_provider: str = "venus",
        step_api_key: Optional[str] = None,
        step_api_model: str = "gpt-4o",
        step_api_fallback_models: Optional[str] = None,
        step_api_global_timeout: float = 600.0,
        val_api_provider: Optional[str] = None,
        val_api_key: Optional[str] = None,
        val_api_model: Optional[str] = None,
        reward_api_fallback_models: Optional[str] = None,
        reward_api_global_timeout: float = 600.0,
        val_api_fallback_models: Optional[str] = None,
        val_api_global_timeout: float = 600.0,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
        )
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        # Load GiG config
        config_path = Path(__file__).resolve().parents[2] / "trainer/config/reward_dapo_trajectory_v3_gig.yaml"
        logger.info(f"Loading GiG config: {config_path}")
        with config_path.open(encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file) or {}

        # KnowGPO V7 uses outcome-only scoring; the V2 C-GRPO baseline config is not required.

        model_from_env = os.environ.get("KNOWGPO_JUDGE_MODEL")
        if model_from_env:
            for key in ("reward_api_model", "step_api_model", "val_api_model"):
                self.config[key] = model_from_env
        if not os.environ.get("KNOWGPO_JUDGE_BASE_URL"):
            raise ValueError("KNOWGPO_JUDGE_BASE_URL is required for KnowGPO outcome and step judges")
        api_key_from_env = os.environ.get("KNOWGPO_JUDGE_API_KEY")
        if api_key_from_env:
            for key in ("reward_api_key", "step_api_key", "val_api_key"):
                self.config[key] = self.config.get(key) or api_key_from_env

        # Store API provider configuration from reward_kwargs
        # Fall back to YAML config values when not provided via command line
        _cfg = self.config  # shorthand

        self.reward_api_provider = reward_api_provider if reward_api_provider != "venus" else _cfg.get("reward_api_provider", reward_api_provider)
        self.reward_api_key = (reward_api_key if reward_api_key else _cfg.get("reward_api_key")) or None
        self.reward_api_model = reward_api_model if reward_api_model != "gpt-4o" else _cfg.get("reward_api_model", reward_api_model)
        self.step_api_provider = step_api_provider if step_api_provider != "venus" else _cfg.get("step_api_provider", step_api_provider)
        self.step_api_key = (step_api_key if step_api_key else _cfg.get("step_api_key")) or None
        self.step_api_model = step_api_model if step_api_model != "gpt-4o" else _cfg.get("step_api_model", step_api_model)

        # Parse fallback models from comma-separated string (command line > yaml > None)
        _step_fb = step_api_fallback_models or _cfg.get("step_api_fallback_models")
        if _step_fb and isinstance(_step_fb, str):
            self.step_api_fallback_models = [
                m.strip() for m in _step_fb.split(",") if m.strip()
            ]
        elif isinstance(_step_fb, list):
            self.step_api_fallback_models = _step_fb
        else:
            self.step_api_fallback_models = None

        self.step_api_global_timeout = step_api_global_timeout if step_api_global_timeout != 600.0 else float(_cfg.get("step_api_global_timeout", step_api_global_timeout))

        self.val_api_provider = val_api_provider or _cfg.get("val_api_provider") or self.reward_api_provider
        self.val_api_key = (val_api_key or _cfg.get("val_api_key") or self.reward_api_key) or None
        self.val_api_model = val_api_model or _cfg.get("val_api_model") or self.reward_api_model

        # Parse reward fallback models from comma-separated string (command line > yaml > None)
        _reward_fb = reward_api_fallback_models or _cfg.get("reward_api_fallback_models")
        if _reward_fb and isinstance(_reward_fb, str):
            self.reward_api_fallback_models = [
                m.strip() for m in _reward_fb.split(",") if m.strip()
            ]
        elif isinstance(_reward_fb, list):
            self.reward_api_fallback_models = _reward_fb
        else:
            self.reward_api_fallback_models = None
        self.reward_api_global_timeout = reward_api_global_timeout if reward_api_global_timeout != 600.0 else float(_cfg.get("reward_api_global_timeout", reward_api_global_timeout))

        # Parse val fallback models from comma-separated string (command line > yaml > reward fallback)
        _val_fb = val_api_fallback_models or _cfg.get("val_api_fallback_models")
        if _val_fb and isinstance(_val_fb, str):
            self.val_api_fallback_models = [
                m.strip() for m in _val_fb.split(",") if m.strip()
            ]
        elif isinstance(_val_fb, list):
            self.val_api_fallback_models = _val_fb
        else:
            # Default: inherit from reward fallback models
            self.val_api_fallback_models = self.reward_api_fallback_models
        self.val_api_global_timeout = val_api_global_timeout if val_api_global_timeout != 600.0 else float(_cfg.get("val_api_global_timeout", val_api_global_timeout))

        # Override config with API provider settings
        self.config['api_provider'] = self.reward_api_provider
        self.config['reward_api_provider'] = self.reward_api_provider
        self.config['reward_api_model'] = self.reward_api_model
        if self.reward_api_key:
            self.config['api_key'] = self.reward_api_key
            self.config['reward_api_key'] = self.reward_api_key
        self.config['step_api_provider'] = self.step_api_provider
        if self.step_api_key:
            self.config['step_api_key'] = self.step_api_key
        self.config['reward_api_fallback_models'] = self.reward_api_fallback_models
        self.config['reward_api_global_timeout'] = self.reward_api_global_timeout

        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        # Dump grouping details config (read from yaml)
        self.dump_grouping = self.config.get("dump_grouping_details", False)
        self.dump_max_queries = self.config.get("dump_max_queries", 0)
        self.dump_step_text_max_chars = self.config.get("dump_step_text_max_chars", 0)

        # Create a timestamped subdirectory under dump_dir so that each RL
        # training run stores all its step-sample JSONL files in one folder.
        _base_dump_dir = self.config.get("dump_dir", "")
        if _base_dump_dir and self.dump_grouping:
            from datetime import datetime as _dt
            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            self.dump_dir = os.path.join(_base_dump_dir, f"run_{_ts}")
            os.makedirs(self.dump_dir, exist_ok=True)
            logger.info(f"[GiG Dump] Created run directory: {self.dump_dir}")
        else:
            self.dump_dir = _base_dump_dir

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None
            assert self.max_resp_len >= self.overlong_buffer_cfg.len

        logger.info("✓ ApiPrimeDapoTrajectoryV3GiGRewardManager initialized")
        logger.info(
            f"  GiG params: gamma={self.config.get('process_reward_gamma', 0.95)}, "
            f"omega={self.config.get('process_reward_omega', 0.5)}, "
            f"enable_process_reward={self.config.get('enable_process_reward', True)}"
        )
        logger.info(
            f"  Step API: provider={self.step_api_provider}, "
            f"model={self.step_api_model}, "
            f"fallback_models={self.step_api_fallback_models}, "
            f"global_timeout={self.step_api_global_timeout}s"
        )
        logger.info(
            f"  Reward API: provider={self.reward_api_provider}, "
            f"model={self.reward_api_model}, "
            f"fallback_models={self.reward_api_fallback_models}, "
            f"global_timeout={self.reward_api_global_timeout}s"
        )
        logger.info(
            f"  Val API: provider={self.val_api_provider}, "
            f"model={self.val_api_model}, "
            f"fallback_models={self.val_api_fallback_models}, "
            f"global_timeout={self.val_api_global_timeout}s"
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards and process reward data.

        Returns reward_tensor (same as V2) plus stores step advantage data
        in batch.non_tensor_batch for ray_trainer to use.
        """
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        _reward_phase_timing = {}  # Sub-phase timing for monitoring
        _reward_total_start = time.time()

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Batch setup
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]
        batch_size = len(data)

        # ========================================
        # Step 1: Extract abnormal flags from rollout_extra_info
        # ========================================
        rollout_extra_info_list = data.non_tensor_batch.get(
            "rollout_extra_info", [{}] * batch_size
        )

        abnormal_flags_list = []
        for rei in rollout_extra_info_list:
            if isinstance(rei, dict):
                abnormal_flags_list.append(rei.get("abnormal_flags", {}))
            else:
                abnormal_flags_list.append({})

        # ========================================
        # Step 2: Identify normal samples
        # ========================================
        normal_indices = []
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            should_give_zero, _ = classify_abnormal_trajectory(abnormal_flags)

            has_answer_tag = False
            try:
                has_answer_tag = "<answer>" in sequences_str[i].lower()
            except Exception:
                pass

            if not should_give_zero and has_answer_tag:
                normal_indices.append(i)
            else:
                if not has_answer_tag:
                    abnormal_flags_list[i]["no_answer_tag"] = True

        logger.info(
            f"Batch: {batch_size}, normal: {len(normal_indices)}, "
            f"abnormal: {batch_size - len(normal_indices)}"
        )

        # Initialize all result arrays
        all_end_to_end_scores = [0.0] * batch_size
        all_outcome_rewards = [0.0] * batch_size
        all_rubric_rewards_raw = [0.0] * batch_size
        all_rubric_rewards_normalized = [0.0] * batch_size
        all_final_rewards = [0.0] * batch_size
        all_abnormal_types = [[] for _ in range(batch_size)]
        all_rubric_details = [{} for _ in range(batch_size)]
        all_r_identify = [0.0] * batch_size
        all_r_support = [0.0] * batch_size
        all_r_connect = [0.0] * batch_size
        all_n_connected = [0] * batch_size
        all_n_total = [0] * batch_size

        # Mark abnormal types
        for i, af in enumerate(abnormal_flags_list):
            if i not in normal_indices:
                all_abnormal_types[i] = [k for k, v in af.items() if v]

        # ========================================
        # Pre-compute: Prepare GiG data early (needed for parallel execution)
        # ========================================
        enable_process_reward = self.config.get("enable_process_reward", True)
        process_reward_gamma = self.config.get("process_reward_gamma", 0.95)
        step_eval_model = self.step_api_model or self.config.get("step_eval_model", "gpt-4o")
        num_workers = self.config.get("num_workers", 64)

        # Build rubrics and questions for GiG (moved up from Step 4)
        all_rubrics = []
        all_questions = []
        original_extra_info_list = data.non_tensor_batch.get(
            "extra_info", [{}] * batch_size
        )

        for i in range(batch_size):
            try:
                ei = original_extra_info_list[i] if i < len(original_extra_info_list) else {}
                if not isinstance(ei, dict):
                    ei = {}

                constraints_gt = ei.get("constraints_gt", [])
                entities_gt = ei.get("entities_gt", {})

                if constraints_gt and entities_gt:
                    rubrics = build_rubrics_from_constraints_gt(
                        constraints_gt, entities_gt
                    )
                else:
                    rubrics = []

                question = ei.get("question", "")
                all_rubrics.append(rubrics)
                all_questions.append(question if question else "")
            except Exception as e:
                logger.warning(f"Sample {i}: rubric extraction failed: {e}")
                all_rubrics.append([])
                all_questions.append("")

        uid_list = list(data.non_tensor_batch.get("uid", [""] * batch_size))

        # Get global_step from meta_info
        global_step = 0
        meta_info = getattr(data, "meta_info", None) or {}
        if isinstance(meta_info, dict):
            global_step = int(
                meta_info.get("global_steps",
                              meta_info.get("global_step", 0))
            )

        # Get rank for dump filename
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0

        # ========================================
        # Step 3 + GiG Phase A: PARALLEL execution
        # C-GRPO scoring and GiG step state evaluation run concurrently
        # because they use different API models and have no data dependency.
        # ========================================
        _parallel_start = time.time()

        step_advantages_per_sample = [[] for _ in range(batch_size)]
        process_metrics = {}
        phase_a_result = None

        # Check if Phase A was pre-launched by ray_trainer (parallel with rollout)
        _phase_a_future = None
        _phase_a_executor = None
        meta_info = getattr(data, "meta_info", None) or {}
        if isinstance(meta_info, dict):
            _phase_a_future = meta_info.pop("phase_a_future", None)
            _phase_a_executor = meta_info.pop("phase_a_executor", None)
        
        _phase_a_precomputed = False
        if _phase_a_future is not None:
            logger.info("[Phase A Parallel] Found pre-launched Phase A future, waiting for result...")
            try:
                phase_a_result = _phase_a_future.result(timeout=3600)
                _phase_a_precomputed = True
                logger.info(
                    f"[Phase A Parallel] Pre-computed Phase A result received: "
                    f"batch_size={phase_a_result.get('batch_size', 0)}"
                )
            except Exception as e:
                logger.error(f"[Phase A Parallel] Pre-launched Phase A failed: {e}")
                logger.error(traceback.format_exc())
                phase_a_result = None
            finally:
                if _phase_a_executor is not None:
                    _phase_a_executor.shutdown(wait=False)

        # Prepare C-GRPO data (needed before parallel launch)
        normal_prompts_str = []
        normal_sequences_str = []
        normal_ground_truth = []
        normal_data_sources = []
        task_extra_infos = []
        question_list = []

        if len(normal_indices) > 0:
            try:
                normal_prompts_str = [
                    self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                    for i in normal_indices
                ]
                normal_sequences_str = [sequences_str[i] for i in normal_indices]

                ground_truth_list = []
                for data_item in data:
                    try:
                        if "reward_model" in data_item.non_tensor_batch:
                            ground_truth_list.append(
                                data_item.non_tensor_batch["reward_model"]["ground_truth"]
                            )
                        else:
                            ground_truth_list.append({})
                    except Exception:
                        ground_truth_list.append({})

                normal_ground_truth = [ground_truth_list[i] for i in normal_indices]
                normal_data_sources = [data_sources[i] for i in normal_indices]

                for idx, data_item in enumerate(data):
                    try:
                        q = data_item.non_tensor_batch.get("question", None)
                        if q is None and isinstance(original_extra_info_list[idx], dict):
                            q = original_extra_info_list[idx].get("question", None)
                        question_list.append(q)
                    except Exception:
                        question_list.append(None)

                for i in normal_indices:
                    try:
                        tei = {}
                        if i < len(original_extra_info_list) and isinstance(
                            original_extra_info_list[i], dict
                        ):
                            tei.update(original_extra_info_list[i])
                        if i < len(question_list) and question_list[i] is not None:
                            tei["question"] = question_list[i]
                        task_extra_infos.append(tei)
                    except Exception:
                        task_extra_infos.append({})
            except Exception as e:
                logger.error(f"C-GRPO data preparation failed: {e}")
                logger.error(traceback.format_exc())

        # Launch C-GRPO and GiG Phase A in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor as _TPE

        cgrpo_results = None

        if len(normal_indices) > 0 and enable_process_reward:
            if _phase_a_precomputed:
                # Phase A already pre-computed by ray_trainer, only run C-GRPO
                logger.info(
                    f"[Parallel] Phase A pre-computed, running C-GRPO only "
                    f"({len(normal_indices)} samples)"
                )
                try:
                    cgrpo_results = self._verify_normal_samples(
                        normal_prompts_str,
                        normal_sequences_str,
                        normal_ground_truth,
                        normal_data_sources,
                        task_extra_infos,
                    )
                except Exception as e:
                    logger.error(f"C-GRPO execution failed: {e}")
                    logger.error(traceback.format_exc())
            else:
                # Both C-GRPO and GiG Phase A can run in parallel
                logger.info(
                    f"[Parallel] Launching C-GRPO ({len(normal_indices)} samples) "
                    f"and GiG Phase A ({batch_size} samples) in parallel"
                )
                with _TPE(max_workers=2) as parallel_executor:
                    # Task 1: C-GRPO scoring (uses reward API model)
                    future_cgrpo = parallel_executor.submit(
                        self._verify_normal_samples,
                        normal_prompts_str,
                        normal_sequences_str,
                        normal_ground_truth,
                        normal_data_sources,
                        task_extra_infos,
                    )

                    # Task 2: GiG Phase A - step extraction + state evaluation
                    # (uses step API model, does NOT need outcome_rewards)
                    future_gig_phase_a = parallel_executor.submit(
                        compute_process_reward_phase_a,
                        trajectories=sequences_str,
                        questions=all_questions,
                        rubrics_list=all_rubrics,
                        uid_list=uid_list,
                        model=step_eval_model,
                        num_workers=num_workers,
                        api_provider=self.step_api_provider,
                        api_key=self.step_api_key,
                        fallback_models=self.step_api_fallback_models,
                        global_timeout=self.step_api_global_timeout,
                    )

                    # Wait for both to complete
                    try:
                        cgrpo_results = future_cgrpo.result()
                    except Exception as e:
                        logger.error(f"C-GRPO parallel execution failed: {e}")
                        logger.error(traceback.format_exc())

                    try:
                        phase_a_result = future_gig_phase_a.result()
                    except Exception as e:
                        logger.error(f"GiG Phase A parallel execution failed: {e}")
                        logger.error(traceback.format_exc())

        elif len(normal_indices) > 0:
            # Only C-GRPO (process reward disabled)
            try:
                cgrpo_results = self._verify_normal_samples(
                    normal_prompts_str,
                    normal_sequences_str,
                    normal_ground_truth,
                    normal_data_sources,
                    task_extra_infos,
                )
            except Exception as e:
                logger.error(f"C-GRPO execution failed: {e}")
                logger.error(traceback.format_exc())

        elif enable_process_reward and not _phase_a_precomputed:
            # Only GiG Phase A (no normal samples for C-GRPO, not pre-computed)
            try:
                phase_a_result = compute_process_reward_phase_a(
                    trajectories=sequences_str,
                    questions=all_questions,
                    rubrics_list=all_rubrics,
                    uid_list=uid_list,
                    model=step_eval_model,
                    num_workers=num_workers,
                    api_provider=self.step_api_provider,
                    api_key=self.step_api_key,
                    fallback_models=self.step_api_fallback_models,
                    global_timeout=self.step_api_global_timeout,
                )
            except Exception as e:
                logger.error(f"GiG Phase A execution failed: {e}")
                logger.error(traceback.format_exc())

        _reward_phase_timing["reward_parallel_s"] = time.time() - _parallel_start

        # ========================================
        # Process C-GRPO results → compute outcome_rewards
        # ========================================
        _cgrpo_process_start = time.time()
        if cgrpo_results is not None and len(normal_indices) > 0:
            try:
                normal_end_to_end_scores = [r.get("end_to_end_score", 0.0) for r in cgrpo_results]
                normal_rubric_rewards = [r.get("rubric_reward", 0.0) for r in cgrpo_results]
                normal_rubric_details = [r.get("rubric_details", {}) for r in cgrpo_results]

                for idx, i in enumerate(normal_indices):
                    try:
                        outcome_reward = 1.0 if normal_end_to_end_scores[idx] >= 1.0 else 0.0
                        final_reward = outcome_reward

                        valid_len = int(valid_response_length[i].item())
                        if valid_len > 0:
                            reward_tensor[i, valid_len - 1] = final_reward

                        all_end_to_end_scores[i] = normal_end_to_end_scores[idx]
                        all_outcome_rewards[i] = outcome_reward
                        all_rubric_rewards_raw[i] = 0.0
                        all_rubric_rewards_normalized[i] = 0.0
                        all_final_rewards[i] = final_reward
                        all_rubric_details[i] = normal_rubric_details[idx]

                        all_r_identify[i] = 0.0
                        all_r_support[i] = 0.0
                        all_r_connect[i] = 0.0
                        all_n_connected[i] = 0
                        all_n_total[i] = 0

                    except Exception as e:
                        logger.error(f"Sample {i} reward computation failed: {e}")
                        all_final_rewards[i] = 0.0
            except Exception as e:
                logger.error(f"C-GRPO result processing failed: {e}")
                logger.error(traceback.format_exc())
        _reward_phase_timing["reward_cgrpo_s"] = time.time() - _cgrpo_process_start

        # ========================================
        # GiG Phase B: Compute step advantages (needs outcome_rewards from C-GRPO)
        # ========================================
        _process_reward_start = time.time()
        if enable_process_reward and phase_a_result is not None:
            try:
                # Isolate overlong samples from GiG grouping by assigning unique UIDs.
                # This prevents overlong samples (reward=0, incomplete trajectory) from
                # distorting normal samples' step-level advantage in state-based groups.
                _enable_overlong_isolation = self.config.get("enable_overlong_isolation", True)
                _phase_b_uid_list = list(uid_list)
                if _enable_overlong_isolation:
                    _overlong_isolated_count = 0
                    for _i, _af in enumerate(abnormal_flags_list):
                        if _af.get("exceed_max_tokens", False):
                            _phase_b_uid_list[_i] = f"__overlong_{_i}_{uuid.uuid4()}"
                            _overlong_isolated_count += 1
                    if _overlong_isolated_count > 0:
                        logger.info(
                            f"[GiG Overlong Isolation] Isolated {_overlong_isolated_count}/{batch_size} "
                            f"overlong samples from Phase B grouping"
                        )

                process_result = compute_process_reward_phase_b(
                    phase_a_result=phase_a_result,
                    trajectories=sequences_str,
                    questions=all_questions,
                    rubrics_list=all_rubrics,
                    outcome_rewards=all_outcome_rewards,
                    uid_list=_phase_b_uid_list,
                    gamma=process_reward_gamma,
                    global_step=global_step,
                    dump_grouping=self.dump_grouping,
                    dump_dir=self.dump_dir,
                    dump_max_queries=self.dump_max_queries,
                    dump_step_text_max_chars=self.dump_step_text_max_chars,
                    omega=self.config.get("process_reward_omega", 0.5),
                    rank=rank,
                )

                raw_step_data = process_result.get("step_advantages_per_sample", [])
                process_metrics = process_result.get("process_reward_metrics", {})

                # Map step advantages to token positions
                response_length = response_ids.shape[-1]
                if raw_step_data:
                    raw_step_data = map_step_advantages_to_tokens(
                        raw_step_data, sequences_str, response_length
                    )
                    for i, steps in enumerate(raw_step_data):
                        if i < batch_size:
                            step_advantages_per_sample[i] = steps

                logger.info(
                    f"[GiG Process Reward] Computed for {batch_size} samples, "
                    f"metrics: {process_metrics}"
                )

            except Exception as e:
                logger.error(f"Process reward Phase B computation failed: {e}")
                logger.error(traceback.format_exc())
        _reward_phase_timing["reward_process_s"] = time.time() - _process_reward_start

        # ========================================
        # Step 5: Store process reward data in non_tensor_batch
        # ========================================
        # Store step advantage data as numpy object array for ray_trainer to use
        data.non_tensor_batch["gig_step_advantages"] = np.array(
            step_advantages_per_sample, dtype=object
        )
        data.non_tensor_batch["gig_omega"] = np.array(
            [self.config.get("process_reward_omega", 0.5)] * batch_size,
            dtype=np.float64,
        )
        # Pass dynamic omega and filtering config to ray_trainer
        data.non_tensor_batch["gig_enable_dynamic_omega"] = np.array(
            [self.config.get("enable_dynamic_omega", False)] * batch_size,
            dtype=object,
        )
        data.non_tensor_batch["gig_omega_start"] = np.array(
            [self.config.get("omega_start", 0.5)] * batch_size,
            dtype=np.float64,
        )
        data.non_tensor_batch["gig_omega_end"] = np.array(
            [self.config.get("omega_end", 0.1)] * batch_size,
            dtype=np.float64,
        )
        data.non_tensor_batch["gig_omega_warmup_ratio"] = np.array(
            [self.config.get("omega_warmup_ratio", 0.0)] * batch_size,
            dtype=np.float64,
        )
        data.non_tensor_batch["gig_enable_group_filtering"] = np.array(
            [self.config.get("enable_group_filtering", True)] * batch_size,
            dtype=object,
        )
        data.non_tensor_batch["gig_enable_overlong_isolation"] = np.array(
            [self.config.get("enable_overlong_isolation", True)] * batch_size,
            dtype=object,
        )

        # ========================================
        # Step 6: Fill reward_extra_info (same as V2)
        # ========================================
        reward_extra_info["end_to_end_score"] = all_end_to_end_scores
        reward_extra_info["outcome_reward"] = all_outcome_rewards
        reward_extra_info["rubric_reward_raw"] = all_rubric_rewards_raw
        reward_extra_info["rubric_reward_normalized"] = all_rubric_rewards_normalized
        reward_extra_info["final_reward"] = all_final_rewards
        reward_extra_info["abnormal_types"] = all_abnormal_types
        reward_extra_info["r_identify"] = all_r_identify
        reward_extra_info["r_support"] = all_r_support
        reward_extra_info["r_connect"] = all_r_connect
        reward_extra_info["n_connected"] = all_n_connected
        reward_extra_info["n_total"] = all_n_total

        # Add process reward metrics to extra_info
        # process_metrics keys are like "gig/num_state_groups", "gig/avg_group_size", etc.
        # Expose them both under their original key and under legacy names for wandb
        for k, v in process_metrics.items():
            reward_extra_info[k] = [v] * batch_size
        # Legacy aliases for backward compatibility with existing wandb dashboards
        reward_extra_info["num_uid_groups"] = [process_metrics.get("gig/num_state_groups", 0)] * batch_size
        reward_extra_info["avg_group_size"] = [process_metrics.get("gig/avg_group_size", 0.0)] * batch_size

        # Rubric error types
        all_rubric_error_types = []
        for i in range(batch_size):
            error_code = all_rubric_details[i].get("error_code", RubricRewardError.SUCCESS)
            all_rubric_error_types.append(error_code)
        reward_extra_info["rubric_error_types"] = all_rubric_error_types

        # Save C-GRPO + GiG info
        cgrpo_reward_info = []
        for i in range(batch_size):
            cgrpo_reward_info.append(
                {
                    "rubric_reward_raw": all_rubric_rewards_raw[i],
                    "rubric_details": all_rubric_details[i],
                    "gig_num_steps": len(step_advantages_per_sample[i]),
                }
            )
        data.non_tensor_batch["cgrpo_reward_info"] = np.array(
            cgrpo_reward_info, dtype=object
        )

        _reward_phase_timing["reward_total_s"] = time.time() - _reward_total_start

        logger.info(
            f"[Reward Summary] avg_reward={reward_tensor.sum(dim=-1).mean().item():.3f}, "
            f"timing: parallel={_reward_phase_timing.get('reward_parallel_s', 0):.1f}s, "
            f"cgrpo={_reward_phase_timing.get('reward_cgrpo_s', 0):.1f}s, "
            f"process_phase_b={_reward_phase_timing.get('reward_process_s', 0):.1f}s, "
            f"total={_reward_phase_timing.get('reward_total_s', 0):.1f}s"
        )

        # Store timing info in reward_extra_info for ray_trainer to pick up
        for timing_key, timing_val in _reward_phase_timing.items():
            reward_extra_info[timing_key] = [timing_val] * batch_size

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor

    def _verify_normal_samples(
        self,
        prompts_strs: List[str],
        sequences_strs: List[str],
        ground_truths: List[Dict[str, Any]],
        data_sources: List[str],
        task_extra_infos: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute C-GRPO rewards for normal samples."""
        batch_size = len(sequences_strs)
        default_result = [
            {"end_to_end_score": 0.0, "rubric_reward": 0.0, "rubric_details": {}}
        ] * batch_size

        try:
            results = asyncio.run(
                parallel_compute_cgrpo_score_async(
                    self.config,
                    prompts_strs,
                    sequences_strs,
                    ground_truths,
                    data_sources,
                    task_extra_infos=task_extra_infos,
                    num_processes=self.config.get("num_workers", 64),
                )
            )
            return results
        except RuntimeError as e:
            if "asyncio.run()" in str(e):
                try:
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(
                        parallel_compute_cgrpo_score_async(
                            self.config,
                            prompts_strs,
                            sequences_strs,
                            ground_truths,
                            data_sources,
                            task_extra_infos=task_extra_infos,
                            num_processes=self.config.get("num_workers", 64),
                        )
                    )
                except Exception:
                    return default_result
            return default_result
        except Exception as e:
            logger.error(f"verify_normal_samples failed: {e}")
            return default_result
