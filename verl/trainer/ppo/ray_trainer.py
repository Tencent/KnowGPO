# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging  # Added by anonymous@example.com: 用于日志输出
import os
import traceback  # Added for GiG advantage fusion error handling
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]

logger = logging.getLogger(__name__)  # Added for GiG advantage fusion logging


# ========================================
# === Online RL Step Summary Logger ===
# === Added by anonymous@example.com ===
# ========================================
class StepSummaryLogger:
    """Structured per-step summary logger for online RL monitoring.

    Writes a human-readable summary to both console and a persistent
    debug log file after each training step. Also computes additional
    monitoring metrics (GRPO grouping, advantage distribution, rollout
    aggregation, reward sub-phase timing, GiG traceability) and returns
    them for wandb/tensorboard upload.

    The debug log file is stored at ``{debug_log_dir}/step_summary.log``.
    """

    def __init__(self, debug_log_dir: str | None = None):
        self._file_logger = None
        if debug_log_dir:
            os.makedirs(debug_log_dir, exist_ok=True)
            log_path = os.path.join(debug_log_dir, "step_summary.log")
            self._file_logger = logging.getLogger("step_summary")
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.propagate = False
            # Avoid adding duplicate handlers on repeated init
            if not self._file_logger.handlers:
                fh = logging.FileHandler(log_path, mode="a")
                fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                self._file_logger.addHandler(fh)
            print(f"[StepSummaryLogger] Writing debug logs to {log_path}")

    # ------------------------------------------------------------------
    # Public entry point - called once per training step
    # ------------------------------------------------------------------
    def log_step(
        self,
        global_step: int,
        epoch: int,
        timing_raw: dict,
        batch,
        metrics: dict,
        config,
    ) -> dict:
        """Compute extra monitoring metrics and emit a step summary.

        Returns a dict of *new* metrics to be merged into ``metrics``.
        """
        extra_metrics: dict = {}

        # --- 1. Reward sub-phase timing (from reward_extra_info) ---
        extra_metrics.update(self._compute_reward_timing_metrics(batch, timing_raw))

        # --- 2. Rollout aggregated metrics ---
        extra_metrics.update(self._compute_rollout_agg_metrics(batch))

        # --- 3. GRPO grouping & advantage distribution ---
        extra_metrics.update(self._compute_grpo_group_metrics(batch))

        # --- 4. Overlong trajectory impact on grouping & advantage ---
        extra_metrics.update(self._compute_overlong_impact_metrics(batch))

        # --- 5. GiG traceability ---
        extra_metrics.update(self._compute_gig_metrics(batch))

        # --- 6. Build the summary string ---
        all_m = {**metrics, **extra_metrics}
        summary = self._build_summary_string(global_step, epoch, timing_raw, all_m)

        # Emit to console
        print(summary)

        # Emit to file
        if self._file_logger:
            self._file_logger.info(summary)

        return extra_metrics

    # ------------------------------------------------------------------
    # Metric computation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_reward_timing_metrics(batch, timing_raw: dict) -> dict:
        """Extract reward sub-phase timing from batch.non_tensor_batch."""
        m = {}
        # Reward managers store timing as repeated scalars in non_tensor_batch
        for key in ("reward_total_s", "reward_cgrpo_s", "reward_process_s"):
            if key in batch.non_tensor_batch:
                arr = batch.non_tensor_batch[key]
                val = float(arr[0]) if hasattr(arr, '__len__') and len(arr) > 0 else 0.0
                m[f"timing_s/{key}"] = val
        return m

    @staticmethod
    def _compute_rollout_agg_metrics(batch) -> dict:
        """Aggregate rollout turn / tool-call statistics for wandb."""
        m = {}
        if "__num_turns__" in batch.non_tensor_batch:
            turns = batch.non_tensor_batch["__num_turns__"]
            if hasattr(turns, '__len__') and len(turns) > 0:
                m["rollout/avg_turns_per_query"] = float(np.mean(turns))
                m["rollout/median_turns_per_query"] = float(np.median(turns))
                m["rollout/max_turns_per_query"] = float(np.max(turns))
        if "tool_call_counts" in batch.non_tensor_batch:
            tcc = batch.non_tensor_batch["tool_call_counts"]
            if hasattr(tcc, '__len__') and len(tcc) > 0:
                m["rollout/total_tool_calls"] = int(np.sum(tcc))
                m["rollout/avg_tool_calls"] = float(np.mean(tcc))
        return m

    @staticmethod
    def _compute_grpo_group_metrics(batch) -> dict:
        """Compute GRPO uid-group reward distribution metrics."""
        m = {}
        uid_arr = batch.non_tensor_batch.get("uid", None)
        if uid_arr is None:
            return m

        try:
            # Sequence-level rewards
            token_level_rewards = batch.batch.get("token_level_rewards", None)
            advantages = batch.batch.get("advantages", None)
            response_mask = batch.batch.get("response_mask", None)

            if token_level_rewards is None:
                return m

            seq_rewards = token_level_rewards.sum(-1).cpu().numpy()

            # Group by uid
            uid_to_indices: dict[str, list] = defaultdict(list)
            for idx, uid in enumerate(uid_arr):
                uid_to_indices[str(uid)].append(idx)

            num_groups = len(uid_to_indices)
            group_sizes = [len(v) for v in uid_to_indices.values()]
            spreads = []
            zero_var_count = 0
            for indices in uid_to_indices.values():
                rewards_in_group = seq_rewards[indices]
                spread = float(rewards_in_group.max() - rewards_in_group.min())
                spreads.append(spread)
                if np.std(rewards_in_group) < 1e-8:
                    zero_var_count += 1

            m["grpo/num_uid_groups"] = num_groups
            m["grpo/avg_group_size"] = float(np.mean(group_sizes)) if group_sizes else 0.0
            m["grpo/reward_spread_mean"] = float(np.mean(spreads)) if spreads else 0.0
            m["grpo/reward_spread_std"] = float(np.std(spreads)) if spreads else 0.0
            m["grpo/zero_variance_groups_rate"] = zero_var_count / max(num_groups, 1)

            # Advantage distribution
            if advantages is not None and response_mask is not None:
                valid_adv = advantages[response_mask.bool()].cpu().numpy()
                if len(valid_adv) > 0:
                    pos_ratio = float((valid_adv > 0).mean())
                    neg_ratio = float((valid_adv < 0).mean())
                    m["grpo/positive_advantage_ratio"] = pos_ratio
                    m["grpo/negative_advantage_ratio"] = neg_ratio
                    m["grpo/advantage_abs_mean"] = float(np.abs(valid_adv).mean())
        except Exception as e:
            logging.getLogger(__name__).warning(f"[GRPO Group Metrics] Failed: {e}")

        return m

    @staticmethod
    def _compute_overlong_impact_metrics(batch) -> dict:
        """Compute the impact of overlong (exceed_max_tokens) trajectories on
        GRPO grouping and advantage calculation.

        This method monitors:
        - Per-group overlong count and ratio
        - How overlong samples affect group mean/std (with vs without overlong)
        - Advantage distortion caused by overlong samples in each group
        """
        m = {}
        rollout_extra_info_list = batch.non_tensor_batch.get("rollout_extra_info", None)
        uid_arr = batch.non_tensor_batch.get("uid", None)
        if rollout_extra_info_list is None or uid_arr is None:
            return m

        try:
            token_level_rewards = batch.batch.get("token_level_rewards", None)
            advantages = batch.batch.get("advantages", None)
            response_mask = batch.batch.get("response_mask", None)
            if token_level_rewards is None:
                return m

            seq_rewards = token_level_rewards.sum(-1).cpu().numpy()
            batch_size = len(uid_arr)

            # Identify overlong sample indices
            overlong_set = set()
            for i, info in enumerate(rollout_extra_info_list):
                abnormal_flags = info.get("abnormal_flags", {}) if isinstance(info, dict) else {}
                if abnormal_flags.get("exceed_max_tokens", False):
                    overlong_set.add(i)

            total_overlong = len(overlong_set)
            m["overlong/total_count"] = total_overlong
            m["overlong/total_ratio"] = total_overlong / max(batch_size, 1)

            if total_overlong == 0:
                # No overlong samples, all impact metrics are zero
                m["overlong/groups_affected"] = 0
                m["overlong/groups_affected_ratio"] = 0.0
                m["overlong/mean_shift_abs_avg"] = 0.0
                m["overlong/std_inflation_avg"] = 0.0
                m["overlong/adv_distortion_avg"] = 0.0
                return m

            # Group by uid
            uid_to_indices: dict[str, list] = defaultdict(list)
            for idx, uid in enumerate(uid_arr):
                uid_to_indices[str(uid)].append(idx)

            groups_affected = 0
            mean_shifts = []       # |mean_all - mean_clean|
            std_inflations = []    # std_all / std_clean
            adv_distortions = []   # avg |adv_with_overlong - adv_without_overlong| for clean samples
            per_group_overlong_ratios = []
            groups_majority_overlong = 0  # groups where >50% are overlong

            for uid, indices in uid_to_indices.items():
                overlong_in_group = [i for i in indices if i in overlong_set]
                clean_in_group = [i for i in indices if i not in overlong_set]
                n_overlong = len(overlong_in_group)
                n_total = len(indices)

                if n_overlong == 0:
                    continue

                groups_affected += 1
                overlong_ratio = n_overlong / n_total
                per_group_overlong_ratios.append(overlong_ratio)

                if overlong_ratio > 0.5:
                    groups_majority_overlong += 1

                # Rewards for all vs clean-only
                rewards_all = seq_rewards[indices]
                mean_all = float(np.mean(rewards_all))
                std_all = float(np.std(rewards_all))

                if len(clean_in_group) > 0:
                    rewards_clean = seq_rewards[clean_in_group]
                    mean_clean = float(np.mean(rewards_clean))
                    std_clean = float(np.std(rewards_clean))
                else:
                    # Entire group is overlong
                    mean_clean = mean_all
                    std_clean = std_all

                mean_shifts.append(abs(mean_all - mean_clean))
                if std_clean > 1e-8:
                    std_inflations.append(std_all / std_clean)
                else:
                    # If clean std is ~0, any non-zero all-std is infinite inflation
                    std_inflations.append(std_all / 1e-8 if std_all > 1e-8 else 1.0)

                # Advantage distortion: how much do advantages of clean samples change
                # due to presence of overlong samples?
                if len(clean_in_group) > 0 and std_all > 1e-8:
                    # Advantage with overlong: (r - mean_all) / std_all
                    adv_with = (rewards_clean - mean_all) / (std_all + 1e-8)
                    # Hypothetical advantage without overlong: (r - mean_clean) / std_clean
                    if std_clean > 1e-8:
                        adv_without = (rewards_clean - mean_clean) / (std_clean + 1e-8)
                    else:
                        adv_without = np.zeros_like(rewards_clean)
                    distortion = float(np.mean(np.abs(adv_with - adv_without)))
                    adv_distortions.append(distortion)

            num_groups = len(uid_to_indices)
            m["overlong/groups_affected"] = groups_affected
            m["overlong/groups_affected_ratio"] = groups_affected / max(num_groups, 1)
            m["overlong/groups_majority_overlong"] = groups_majority_overlong
            m["overlong/per_group_ratio_mean"] = float(np.mean(per_group_overlong_ratios)) if per_group_overlong_ratios else 0.0
            m["overlong/mean_shift_abs_avg"] = float(np.mean(mean_shifts)) if mean_shifts else 0.0
            m["overlong/std_inflation_avg"] = float(np.mean(std_inflations)) if std_inflations else 0.0
            m["overlong/adv_distortion_avg"] = float(np.mean(adv_distortions)) if adv_distortions else 0.0

            # Reward comparison: overlong vs clean
            overlong_indices = list(overlong_set)
            clean_indices = [i for i in range(batch_size) if i not in overlong_set]
            m["overlong/reward_mean_overlong"] = float(np.mean(seq_rewards[overlong_indices])) if overlong_indices else 0.0
            m["overlong/reward_mean_clean"] = float(np.mean(seq_rewards[clean_indices])) if clean_indices else 0.0

        except Exception as e:
            logging.getLogger(__name__).warning(f"[Overlong Impact Metrics] Failed: {e}")

        return m

    @staticmethod
    def _compute_gig_metrics(batch) -> dict:
        """Compute GiG process reward traceability metrics."""
        m = {}
        if "gig_step_advantages" not in batch.non_tensor_batch:
            return m

        try:
            step_advs = batch.non_tensor_batch["gig_step_advantages"]
            total_steps = 0
            non_empty_count = 0
            all_step_adv_values = []

            for idx, sa in enumerate(step_advs):
                # sa may be: list of dicts, numpy array of dicts, or empty
                try:
                    sa_list = list(sa) if hasattr(sa, '__iter__') else []
                except (TypeError, ValueError):
                    sa_list = []

                if len(sa_list) > 0:
                    non_empty_count += 1
                    total_steps += len(sa_list)
                    for step_info in sa_list:
                        if isinstance(step_info, dict):
                            val = step_info.get("step_advantage", 0.0)
                            all_step_adv_values.append(float(val))
                        elif isinstance(step_info, (int, float, np.floating, np.integer)):
                            # Fallback: if step_info is a scalar value directly
                            all_step_adv_values.append(float(step_info))

            batch_size = len(step_advs)
            m["gig/avg_steps_per_trajectory"] = total_steps / max(non_empty_count, 1)
            m["gig/trajectories_with_steps"] = non_empty_count
            m["gig/single_element_group_rate"] = 1.0 - (non_empty_count / max(batch_size, 1))

            if all_step_adv_values:
                arr = np.array(all_step_adv_values, dtype=np.float64)
                m["gig/step_advantage_abs_mean"] = float(np.abs(arr).mean())
                m["gig/step_advantage_std"] = float(np.std(arr))
            else:
                m["gig/step_advantage_abs_mean"] = 0.0
                m["gig/step_advantage_std"] = 0.0

            # Read process reward metrics stored in non_tensor_batch by reward manager.
            # Keys: "gig/num_state_groups", "gig/avg_group_size", "gig/max_group_size",
            #       "gig/num_non_final_steps", "gig/num_final_steps", "gig/num_total_steps"
            # Also legacy keys: "num_uid_groups", "avg_group_size"
            _gig_metric_keys = [
                "gig/num_state_groups",
                "gig/avg_group_size",
                "gig/max_group_size",
                "gig/num_non_final_steps",
                "gig/num_final_steps",
                "gig/num_total_steps",
                "num_uid_groups",
                "process_reward/total_steps",
                "process_reward/avg_steps_per_trajectory",
                "process_reward/non_zero_advantage_count",
                "process_reward/mean_step_advantage",
                "process_reward/std_step_advantage",
            ]
            for key in _gig_metric_keys:
                if key in batch.non_tensor_batch:
                    arr = batch.non_tensor_batch[key]
                    try:
                        val = float(arr[0]) if hasattr(arr, '__len__') and len(arr) > 0 else float(arr)
                        m[key] = val
                    except Exception:
                        pass

        except Exception as e:
            logging.getLogger(__name__).warning(f"[GiG Metrics] Failed: {e}")
            import traceback as _tb
            logging.getLogger(__name__).warning(_tb.format_exc())

        return m

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary_string(
        global_step: int, epoch: int, timing_raw: dict, m: dict
    ) -> str:
        """Build a human-readable step summary string."""

        def _g(key, default=0.0, fmt=".1f"):
            """Get metric value with format."""
            val = m.get(key, default)
            if isinstance(val, float):
                return f"{val:{fmt}}"
            return str(val)

        def _pct(key):
            """Get a percentage metric."""
            val = m.get(key, None)
            if val is None:
                return "N/A"
            return f"{val*100:.1f}%"

        lines = [
            f"\n[Step {global_step} Summary] ==========================================",
            # Timing
            (
                f"  Timing: step={_g('timing_s/step')}s, "
                f"gen={_g('timing_s/gen')}s, "
                f"reward={_g('timing_s/reward')}s"
                f"(cgrpo={_g('timing_s/reward_cgrpo_s')}s, "
                f"process={_g('timing_s/reward_process_s')}s), "
                f"adv={_g('timing_s/adv')}s, "
                f"update_actor={_g('timing_s/update_actor')}s"
            ),
            # Rollout
            (
                f"  Rollout: avg_turns={_g('rollout/avg_turns_per_query')}, "
                f"median_turns={_g('rollout/median_turns_per_query')}, "
                f"max_turns={_g('rollout/max_turns_per_query')}, "
                f"total_tool_calls={_g('rollout/total_tool_calls', 0)}"
            ),
            # Abnormal
            (
                f"  Abnormal: normal={_pct('abnormal/normal_rate')}, "
                f"max_turns={_pct('abnormal/exceed_max_turns_rate')}, "
                f"max_tokens={_pct('abnormal/exceed_max_tokens_rate')}, "
                f"parse_err={_pct('abnormal/tool_parse_error_rate')}"
            ),
            # Reward
            (
                f"  Reward: mean={_g('critic/score/mean', fmt='.3f')}, "
                f"max={_g('critic/score/max', fmt='.3f')}, "
                f"min={_g('critic/score/min', fmt='.3f')}, "
                f"zero_var_groups={_pct('grpo/zero_variance_groups_rate')}"
            ),
            # GRPO
            (
                f"  GRPO: groups={_g('grpo/num_uid_groups', 0)}, "
                f"avg_size={_g('grpo/avg_group_size')}, "
                f"reward_spread={_g('grpo/reward_spread_mean', fmt='.3f')}, "
                f"pos_adv={_pct('grpo/positive_advantage_ratio')}, "
                f"neg_adv={_pct('grpo/negative_advantage_ratio')}, "
                f"adv_abs={_g('grpo/advantage_abs_mean', fmt='.4f')}"
            ),
        ]

        # Overlong Impact (conditional - only when overlong samples exist)
        overlong_count = m.get("overlong/total_count", 0)
        if overlong_count > 0:
            lines.append(
                f"  Overlong Impact: count={_g('overlong/total_count', 0)}"
                f"({_pct('overlong/total_ratio')}), "
                f"groups_affected={_g('overlong/groups_affected', 0)}"
                f"({_pct('overlong/groups_affected_ratio')}), "
                f"majority_overlong_groups={_g('overlong/groups_majority_overlong', 0)}"
            )
            lines.append(
                f"    reward: overlong_mean={_g('overlong/reward_mean_overlong', fmt='.3f')}, "
                f"clean_mean={_g('overlong/reward_mean_clean', fmt='.3f')}, "
                f"mean_shift={_g('overlong/mean_shift_abs_avg', fmt='.4f')}, "
                f"std_inflation={_g('overlong/std_inflation_avg', fmt='.2f')}x, "
                f"adv_distortion={_g('overlong/adv_distortion_avg', fmt='.4f')}"
            )

        # GiG (conditional)
        if "gig/omega" in m or "gig/avg_steps_per_trajectory" in m:
            lines.append(
                f"  GiG: omega={_g('gig/omega', fmt='.2f')}, "
                f"steps/traj={_g('gig/avg_steps_per_trajectory')}, "
                f"step_adv_abs={_g('gig/step_advantage_abs_mean', fmt='.4f')}, "
                f"with_steps={_g('gig/trajectories_with_steps', 0)}"
            )

        # GiG Ablation (conditional)
        if "gig_ablation/ae_as_correlation" in m:
            lines.append(
                f"  GiG-Ablation: ae_as_corr={_g('gig_ablation/ae_as_correlation', fmt='.3f')}, "
                f"dir_change={_pct('gig_ablation/advantage_change_ratio')}, "
                f"mag_change={_g('gig_ablation/advantage_magnitude_change', fmt='.2f')}x, "
                f"nonzero={_pct('gig_ablation/step_adv_nonzero_ratio')}, "
                f"agree={_pct('gig_ablation/reward_direction_agreement')}, "
                f"conflict={_pct('gig_ablation/conflicting_signal_ratio')}, "
                f"ent_gain={_g('gig_ablation/entropy_gain', fmt='.3f')}"
            )

        # Perf
        lines.append(
            f"  Perf: throughput={_g('perf/throughput', fmt='.0f')} tok/s/gpu, "
            f"total_tokens={_g('perf/total_num_tokens', 0)}"
        )

        lines.append("=" * 60)
        return "\n".join(lines)


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def safe_convert_reward_extra_info_to_numpy(reward_extra_infos_dict: dict) -> dict:
    """安全地将 reward_extra_info 转换为 numpy 数组，适配不同的 reward manager。
    
    该函数会自动检测数据类型并应用正确的转换策略：
    - 对于标量列表（如 float、int），直接转换为 numpy 数组
    - 对于嵌套列表（如 abnormal_types），使用 dtype=object 转换
    - 对于其他复杂对象，保持 dtype=object
    
    这样可以兼容不同 reward manager 返回的自定义字段，无需修改 ray_trainer。
    
    Args:
        reward_extra_infos_dict: 从 reward manager 返回的 reward_extra_info 字典
        
    Returns:
        转换后的字典，所有值都是 numpy 数组
        
    Added by: anonymous@example.com
    Date: 2026-01-01
    """
    if not reward_extra_infos_dict:
        return {}
    
    converted_dict = {}
    for key, value in reward_extra_infos_dict.items():
        if not isinstance(value, list):
            # 如果不是列表，尝试直接转换（保持原有行为）
            converted_dict[key] = np.array(value)
            continue
        
        if len(value) == 0:
            # 空列表，使用 object 类型
            converted_dict[key] = np.array(value, dtype=object)
            continue
        
        # 检查第一个元素的类型
        first_elem = value[0]
        
        # 情况1：标量类型（int, float, bool, str）
        if isinstance(first_elem, (int, float, bool, str, np.integer, np.floating)):
            try:
                # 尝试直接转换（最快）
                converted_dict[key] = np.array(value)
            except (ValueError, TypeError):
                # 如果失败，使用 dtype=object
                converted_dict[key] = np.array(value, dtype=object)
        
        # 情况2：嵌套结构（list, dict, tuple 等）
        elif isinstance(first_elem, (list, dict, tuple)):
            # 对于嵌套结构，始终使用 dtype=object
            converted_dict[key] = np.array(value, dtype=object)
        
        # 情况3：其他类型（自定义对象等）
        else:
            # 使用 dtype=object 处理
            converted_dict[key] = np.array(value, dtype=object)
    
    return converted_dict


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            
            # Added by anonymous@example.com: 添加 try-except 捕获具体错误
            try:
                result = self.val_reward_fn(test_batch, return_dict=True)
                # update1017: 验证集reward用val_reward_tensor
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_extra_infos_dict["reward"].extend(scores)
                print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)
                        print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")
            except Exception as e:
                import traceback
                logging.error(f"[Validation] val_reward_fn 调用失败！")
                logging.error(f"[Validation] 错误类型: {type(e).__name__}")
                logging.error(f"[Validation] 错误信息: {str(e)}")
                logging.error(f"[Validation] 完整堆栈:\n{traceback.format_exc()}")
                # 打印 test_batch 的关键信息以便调试
                logging.error(f"[Validation] test_batch.batch.keys(): {list(test_batch.batch.keys())}")
                logging.error(f"[Validation] test_batch.non_tensor_batch.keys(): {list(test_batch.non_tensor_batch.keys())}")
                if "rollout_extra_info" in test_batch.non_tensor_batch:
                    logging.error(f"[Validation] rollout_extra_info 存在，长度: {len(test_batch.non_tensor_batch['rollout_extra_info'])}")
                else:
                    logging.error(f"[Validation] rollout_extra_info 不存在！这可能是验证失败的原因。")
                raise  # 重新抛出异常，保持原有行为

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # 过滤掉非数值字段（例如 abnormal_types 是字符串列表）
        # 只保留可以计算统计量的数值字段
        # 这样可以防止自定义 reward_manager 返回的奇怪参数（字符串、列表、字典等）导致报错
        numeric_reward_extra_infos_dict = {}
        for key, values in reward_extra_infos_dict.items():
            if len(values) == 0:
                continue
            
            # 检查第一个非 None 元素的类型
            first_valid = next((v for v in values if v is not None), None)
            if first_valid is None:
                continue
            
            # 只保留数值类型（int, float, np.number）
            # 排除字符串、列表、字典等
            if isinstance(first_valid, (int, float, np.number)):
                numeric_reward_extra_infos_dict[key] = values
            elif isinstance(first_valid, (list, dict, str)):
                # 跳过非数值类型（例如 abnormal_types 是列表）
            
                continue
            else:
                # 未知类型，尝试转换为 float
                try:
                    float(first_valid)
                    numeric_reward_extra_infos_dict[key] = values
                except (TypeError, ValueError):
                    # 转换失败，跳过
        
                    continue
        
        # 使用 try-except 包裹 process_validation_metrics 防止任何潜在的报错
        try:
            data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, numeric_reward_extra_infos_dict)
        except Exception as e:
        
            # 如果失败，返回空的 metrics
            data_src2var2metric2val = {}
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Initialize step summary logger for online RL monitoring
        debug_log_dir = self.config.trainer.get("debug_log_dir", None)
        step_summary_logger = StepSummaryLogger(debug_log_dir=debug_log_dir)

        # load checkpoint before doing anything
        self._load_checkpoint()

        # ======================================================================
        # IGPO V8: inject actor_rollout_wg + tokenizer into BOTH reward_fn and
        # val_reward_fn BEFORE val_before_train kicks off. The V8 reward
        # manager needs actor_rollout_wg to run an offline GT-logprob replay
        # (its compute_log_prob calls). Other reward managers do not implement
        # set_actor_rollout_wg and are unaffected.
        # ======================================================================
        for _rf_name, _rf in (("reward_fn", self.reward_fn), ("val_reward_fn", self.val_reward_fn)):
            if _rf is not None and hasattr(_rf, "set_actor_rollout_wg"):
                try:
                    _rf.set_actor_rollout_wg(
                        actor_rollout_wg=self.actor_rollout_wg,
                        tokenizer=self.tokenizer,
                    )
                except Exception as _set_e:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        f"[IGPO] set_actor_rollout_wg on {_rf_name} failed (non-fatal): {_set_e}"
                    )

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate() 
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                # === Inject Incremental Phase A config into gen_batch ===
                # This allows the rollout engine to start Phase A evaluation
                # incrementally as each request completes, instead of waiting
                # for all requests to finish.
                # NOTE: Must be done AFTER repeat() so that extra_info indices
                # align with the repeated batch (e.g., 128 samples, not 16).
                try:
                    from verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig import (
                        ApiPrimeDapoTrajectoryV3GiGRewardManager,
                    )
                    if isinstance(self.reward_fn, ApiPrimeDapoTrajectoryV3GiGRewardManager):
                        _gig_rm = self.reward_fn
                        _gig_config = _gig_rm.config
                        if _gig_config.get("enable_process_reward", True):
                            # Copy extra_info into gen_batch so that each DP rank's rollout
                            # engine can access it from prompts.non_tensor_batch["extra_info"].
                            # batch.non_tensor_batch["extra_info"] has NOT been repeated yet
                            # (batch is still 40 prompts), so we repeat it manually to match
                            # gen_batch size (640 samples after repeat).
                            _pre_repeat_extra_info = batch.non_tensor_batch.get("extra_info", None)
                            if _pre_repeat_extra_info is not None:
                                _n = self.config.actor_rollout_ref.rollout.n
                                _repeated_extra_info = np.repeat(_pre_repeat_extra_info, _n, axis=0)
                                gen_batch.non_tensor_batch["extra_info"] = _repeated_extra_info
                                print(
                                    f"[IncrementalPhaseA] Copied extra_info to gen_batch: "
                                    f"original={len(_pre_repeat_extra_info)}, "
                                    f"repeated={len(_repeated_extra_info)} (n={_n})"
                                )

                            # Inject Phase A config into gen_batch.meta_info.
                            # The rollout engine will read extra_info directly from
                            # prompts.non_tensor_batch["extra_info"]
                            # which IS correctly chunked per DP rank.
                            gen_batch.meta_info["incremental_phase_a_config"] = {
                                "model": _gig_rm.step_api_model or _gig_config.get("step_eval_model", "gpt-4o"),
                                "num_workers": _gig_config.get("num_workers", 64),
                                "timeout": 600.0,
                                "api_provider": _gig_rm.step_api_provider,
                                "api_key": _gig_rm.step_api_key,
                                "fallback_models": _gig_rm.step_api_fallback_models,
                                "global_timeout": _gig_rm.step_api_global_timeout,
                                # NOTE: Do NOT pass extra_info_list in meta_info!
                                # meta_info is shared across all DP ranks (not chunked),
                                # so each rank would use the same global list with wrong indices.
                                # Instead, sglang_rollout.py reads from prompts.non_tensor_batch["extra_info"]
                                # which IS correctly chunked per DP rank.
                            }
                            print(
                                f"[IncrementalPhaseA] Config injected into gen_batch.meta_info "
                                f"(model={gen_batch.meta_info['incremental_phase_a_config']['model']})"
                            )
                except (ImportError, AttributeError, Exception) as _e:
                    print(f"[IncrementalPhaseA] WARNING: Failed to inject config: {_e}")

                is_last_step = self.global_steps >= self.total_training_steps

                import datetime
                def _phase_log(phase_name, extra=""):
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elapsed = timing_raw.get(phase_name, 0)
                    msg = f"[Step {self.global_steps}] [{ts}] Phase '{phase_name}' done in {elapsed:.1f}s" if elapsed else f"[Step {self.global_steps}] [{ts}] Phase '{phase_name}' starting..."
                    if extra:
                        msg += f" | {extra}"
                    pprint(msg)

                _phase_log("step", "beginning step")
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    _phase_log("gen")

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # pass global_steps to reward manager (e.g. for dump filename)
                    batch.meta_info["global_steps"] = self.global_steps

                    # ========================================
                    # === Phase A Parallel Launch ===
                    # ========================================
                    # Launch GiG Phase A (step extraction + state evaluation) in a
                    # background thread BEFORE reward computation starts.
                    # Phase A does NOT depend on outcome_rewards, so it can run
                    # concurrently with C-GRPO scoring and old_log_prob computation.
                    #
                    # NEW: If incremental Phase A was already completed during rollout,
                    # use those results directly instead of launching a new Phase A.
                    _phase_a_future = None
                    _phase_a_executor = None

                    # Check if incremental Phase A results are available from rollout
                    # Phase A results are stored in non_tensor_batch (not meta_info)
                    # so that DataProto.concat() correctly merges all DP ranks.
                    # After batch.union(gen_batch_output), these keys are in batch.non_tensor_batch.
                    # After balance_batch, the per-sample ordering is preserved (all keys reordered together).
                    _has_incremental_phase_a = "_phase_a_available" in batch.non_tensor_batch
                    if _has_incremental_phase_a:
                        _pa_steps = list(batch.non_tensor_batch["_phase_a_steps"])
                        _pa_svs = list(batch.non_tensor_batch["_phase_a_state_vectors"])
                        _pa_batch_size = len(_pa_steps)
                        _incremental_result = {
                            "all_steps_per_sample": _pa_steps,
                            "corrected_state_vectors_per_sample": _pa_svs,
                            "batch_size": _pa_batch_size,
                        }
                        print(
                            f"[Phase A Parallel] Reassembled incremental Phase A results from "
                            f"non_tensor_batch (batch_size={_pa_batch_size}). "
                            f"All DP ranks merged correctly. Skipping separate Phase A launch."
                        )
                        # Wrap the pre-computed result in a resolved Future
                        from concurrent.futures import ThreadPoolExecutor as _TPE
                        _phase_a_executor = _TPE(max_workers=1)
                        _phase_a_future = _phase_a_executor.submit(lambda r: r, _incremental_result)
                        # Clean up temporary keys from non_tensor_batch
                        for _tmp_key in ("_phase_a_steps", "_phase_a_state_vectors", "_phase_a_available"):
                            batch.non_tensor_batch.pop(_tmp_key, None)
                    else:
                        # Fallback: launch Phase A in background thread (original behavior)
                        try:
                            from verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig import (
                                ApiPrimeDapoTrajectoryV3GiGRewardManager,
                            )
                            if isinstance(self.reward_fn, ApiPrimeDapoTrajectoryV3GiGRewardManager):
                                _gig_rm = self.reward_fn
                                _gig_config = _gig_rm.config
                                if _gig_config.get("enable_process_reward", True):
                                    from concurrent.futures import ThreadPoolExecutor as _TPE
                                    from verl.utils.reward_score.process_reward_gig import (
                                        compute_process_reward_phase_a,
                                        build_rubrics_from_constraints_gt,
                                    )
                                    
                                    _batch_size = len(batch.batch["responses"])
                                    _response_ids = batch.batch["responses"]
                                    _sequences_str = self.tokenizer.batch_decode(_response_ids, skip_special_tokens=True)
                                    _uid_list = list(batch.non_tensor_batch.get("uid", [""] * _batch_size))
                                    
                                    # Build rubrics and questions
                                    _original_extra_info_list = batch.non_tensor_batch.get("extra_info", [{}] * _batch_size)
                                    _all_rubrics = []
                                    _all_questions = []
                                    for _i in range(_batch_size):
                                        try:
                                            _ei = _original_extra_info_list[_i] if _i < len(_original_extra_info_list) else {}
                                            if not isinstance(_ei, dict):
                                                _ei = {}
                                            _cgt = _ei.get("constraints_gt", [])
                                            _egt = _ei.get("entities_gt", {})
                                            if _cgt and _egt:
                                                _rubrics = build_rubrics_from_constraints_gt(_cgt, _egt)
                                            else:
                                                _rubrics = []
                                            _q = _ei.get("question", "")
                                            _all_rubrics.append(_rubrics)
                                            _all_questions.append(_q if _q else "")
                                        except Exception:
                                            _all_rubrics.append([])
                                            _all_questions.append("")
                                    
                                    _step_eval_model = _gig_rm.step_api_model or _gig_config.get("step_eval_model", "gpt-4o")
                                    _num_workers = _gig_config.get("num_workers", 64)
                                    
                                    _phase_a_executor = _TPE(max_workers=1)
                                    _phase_a_future = _phase_a_executor.submit(
                                        compute_process_reward_phase_a,
                                        trajectories=_sequences_str,
                                        questions=_all_questions,
                                        rubrics_list=_all_rubrics,
                                        uid_list=_uid_list,
                                        model=_step_eval_model,
                                        num_workers=_num_workers,
                                        api_provider=_gig_rm.step_api_provider,
                                        api_key=_gig_rm.step_api_key,
                                        fallback_models=_gig_rm.step_api_fallback_models,
                                        global_timeout=_gig_rm.step_api_global_timeout,
                                    )
                                    print(
                                        f"[Phase A Parallel] Launched Phase A async with {_batch_size} samples, "
                                        f"model={_step_eval_model}, workers={_num_workers}"
                                    )
                        except ImportError:
                            pass
                        except Exception as _e:
                            print(f"[Phase A Parallel] WARNING: Failed to launch async Phase A: {_e}")
                        _phase_a_future = None

                    # Store Phase A future in batch meta_info for reward manager to pick up
                    if _phase_a_future is not None:
                        batch.meta_info["phase_a_future"] = _phase_a_future
                        batch.meta_info["phase_a_executor"] = _phase_a_executor

                    _phase_log("pre_reward", "entering reward phase")
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # ======================================================
                        # IGPO V8: inject actor_rollout_wg + tokenizer into the
                        # reward manager so that the V8 manager can run an
                        # offline "GT log-prob replay" forward pass to compute
                        # per-turn information-gain rewards.
                        # The setter is a no-op for any reward manager that
                        # does not implement it (all V1~V7 managers).
                        # ======================================================
                        if hasattr(self.reward_fn, "set_actor_rollout_wg"):
                            try:
                                self.reward_fn.set_actor_rollout_wg(
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    tokenizer=self.tokenizer,
                                )
                            except Exception as _set_e:  # noqa: BLE001
                                logging.getLogger(__name__).warning(
                                    f"[IGPO] set_actor_rollout_wg failed (non-fatal): {_set_e}"
                                )

                        if self.config.reward_model.launch_reward_fn_async:
                            # In async mode, batch is serialized via Ray.
                            # concurrent.futures.Future and ThreadPoolExecutor are NOT
                            # serializable, so we must remove them before ray.put().
                            # The reward manager will fall back to running Phase A
                            # internally when no pre-computed future is found.
                            if _phase_a_future is not None:
                                print(
                                    "[Phase A Parallel] Async reward mode detected. "
                                    "Cancelling pre-launched Phase A (not serializable for Ray). "
                                    "Reward manager will run Phase A internally."
                                )
                                _phase_a_future.cancel()
                                if _phase_a_executor is not None:
                                    _phase_a_executor.shutdown(wait=False)
                                batch.meta_info.pop("phase_a_future", None)
                                batch.meta_info.pop("phase_a_executor", None)
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    _phase_log("reward")

                    # recompute old_log_probs
                    _phase_log("old_log_prob", "entering compute_log_prob (OOM-prone)")
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    _phase_log("old_log_prob")

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    _phase_log("pre_adv", "entering advantage computation")
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            # 使用安全转换函数，自动适配不同 reward manager 的自定义字段
                            converted_extra_info = safe_convert_reward_extra_info_to_numpy(reward_extra_infos_dict)
                            batch.non_tensor_batch.update(converted_extra_info)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # ========================================
                        # === Pre-Advantage: Overlong Sample Isolation ===
                        # ========================================
                        # Before advantage computation, isolate overlong samples
                        # by assigning them unique UIDs so they form singleton groups.
                        # This prevents overlong samples (reward typically 0) from
                        # distorting normal samples' group mean/std in GRPO.
                        _overlong_isolation_enabled = False
                        _original_uids = None
                        _overlong_indices_set = set()
                        if "rollout_extra_info" in batch.non_tensor_batch:
                            _enable_overlong_isolation = True
                            if "gig_enable_overlong_isolation" in batch.non_tensor_batch:
                                _val = batch.non_tensor_batch["gig_enable_overlong_isolation"][0]
                                _enable_overlong_isolation = bool(_val)
                            
                            if _enable_overlong_isolation:
                                _rollout_extra_info_list = batch.non_tensor_batch["rollout_extra_info"]
                                for _i, _rei in enumerate(_rollout_extra_info_list):
                                    _af = _rei.get("abnormal_flags", {}) if isinstance(_rei, dict) else {}
                                    if _af.get("exceed_max_tokens", False):
                                        _overlong_indices_set.add(_i)
                                
                                if _overlong_indices_set:
                                    _overlong_isolation_enabled = True
                                    _original_uids = batch.non_tensor_batch["uid"].copy()
                                    for _idx in _overlong_indices_set:
                                        batch.non_tensor_batch["uid"][_idx] = f"__overlong_{_idx}_{uuid.uuid4()}"
                                    print(
                                        f"[Overlong Isolation] Isolated {len(_overlong_indices_set)}/"
                                        f"{len(_rollout_extra_info_list)} overlong samples into singleton groups"
                                    )
                                    metrics["filtering/overlong_isolated_count"] = len(_overlong_indices_set)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # Restore original UIDs after advantage computation
                        if _overlong_isolation_enabled and _original_uids is not None:
                            batch.non_tensor_batch["uid"] = _original_uids

                        # ========================================
                        # === Post-Advantage: Group Filtering ===
                        # ========================================
                        # Filter uid groups based on two criteria:
                        # 1. Valid (non-overlong) complete trajectories < 4
                        # 2. All valid trajectories' outcome rewards are 0 or all are 1
                        # These groups provide zero/insufficient gradient signal.
                        # Set their response_mask to 0 so they don't participate in loss.
                        _enable_group_filtering = True
                        if "gig_enable_group_filtering" in batch.non_tensor_batch:
                            _val = batch.non_tensor_batch["gig_enable_group_filtering"][0]
                            _enable_group_filtering = bool(_val)
                        
                        if _enable_group_filtering:
                            _uid_arr = batch.non_tensor_batch.get("uid", None)
                            _token_level_rewards = batch.batch.get("token_level_rewards", None)
                            if _uid_arr is not None and _token_level_rewards is not None:
                                _seq_rewards = _token_level_rewards.sum(dim=-1)  # [batch_size]

                                # --- Plan B-aware setup ---
                                # Traj "length" = number of valid response tokens
                                # (response_mask.sum(-1)). This is a proxy for the
                                # RGC non-final step count (both grow with search
                                # rounds), so a non-zero L_std at the uid level
                                # means Plan B can still extract "short > long"
                                # gradient even when outcomes are uniform.
                                _resp_mask_t = batch.batch.get("response_mask", None)
                                if _resp_mask_t is not None:
                                    _traj_len_t = _resp_mask_t.sum(dim=-1).float()
                                else:
                                    _traj_len_t = None

                                # Plan B is only ON when the RGC reward manager
                                # turned it on. If the batch does not carry a
                                # fusion flag, be conservative and keep the old
                                # filtering semantic (no Plan B rescue).
                                _plan_b_active = (
                                    "gig_step_advantages" in batch.non_tensor_batch
                                    and _traj_len_t is not None
                                )
                                _length_eps = 1e-6
                                _length_min_group_size = 2  # matches RGC default
                                
                                # Build uid -> all indices (including overlong)
                                _uid_to_all_indices = defaultdict(list)
                                for _i, _uid in enumerate(_uid_arr):
                                    _uid_to_all_indices[str(_uid)].append(_i)
                                
                                # Build uid -> valid (non-overlong) indices only
                                _uid_to_valid_indices = defaultdict(list)
                                for _i, _uid in enumerate(_uid_arr):
                                    if _i not in _overlong_indices_set:
                                        _uid_to_valid_indices[str(_uid)].append(_i)
                                
                                _filtered_groups = 0
                                _filtered_samples = 0
                                _filtered_insufficient = 0
                                _filtered_all_same = 0
                                _rescued_by_plan_b = 0     # NEW: groups kept alive
                                _total_groups = len(_uid_to_all_indices)
                                
                                for _uid, _all_indices in _uid_to_all_indices.items():
                                    _valid_indices = _uid_to_valid_indices.get(_uid, [])
                                    _should_filter = False
                                    
                                    # Criterion 1: Valid complete trajectories < 4
                                    if len(_valid_indices) < 4:
                                        _should_filter = True
                                        _filtered_insufficient += 1
                                    else:
                                        # Criterion 2: outcomes are uniform (all 0 / all 1)
                                        _group_rewards = _seq_rewards[_valid_indices]
                                        _all_zero = (_group_rewards.abs() < 1e-6).all().item()
                                        _all_one = ((_group_rewards - 1.0).abs() < 1e-6).all().item()
                                        _outcomes_uniform = _all_zero or _all_one

                                        if _outcomes_uniform:
                                            # --- Plan B rescue clause ---
                                            # If the group still has meaningful
                                            # length variance AND Plan B is
                                            # active, RGC can generate a
                                            # "short > long" gradient even
                                            # without outcome differences -> do
                                            # NOT filter.
                                            _keep_for_plan_b = False
                                            if (
                                                _plan_b_active
                                                and len(_valid_indices) >= _length_min_group_size
                                            ):
                                                _group_lengths = _traj_len_t[_valid_indices]
                                                _len_std = float(_group_lengths.std().item())
                                                if _len_std > _length_eps:
                                                    _keep_for_plan_b = True

                                            if _keep_for_plan_b:
                                                _rescued_by_plan_b += 1
                                            else:
                                                _should_filter = True
                                                _filtered_all_same += 1
                                    
                                    if _should_filter:
                                        # Mask ALL samples in this group (including overlong)
                                        for _idx in _all_indices:
                                            batch.batch["response_mask"][_idx] = 0.0
                                        _filtered_groups += 1
                                        _filtered_samples += len(_all_indices)
                                
                                # Compute per-group valid trajectory count distribution
                                _valid_counts = [len(_uid_to_valid_indices.get(_uid, [])) for _uid in _uid_to_all_indices]
                                _effective_groups = _total_groups - _filtered_groups
                                
                                if _filtered_groups > 0 or _rescued_by_plan_b > 0:
                                    print(
                                        f"[Group Filtering] Filtered {_filtered_groups}/{_total_groups} groups "
                                        f"({_filtered_samples} samples): "
                                        f"insufficient_valid={_filtered_insufficient}, "
                                        f"all_same_reward={_filtered_all_same}, "
                                        f"plan_b_rescued={_rescued_by_plan_b}, "
                                        f"effective_groups={_effective_groups}"
                                    )
                                metrics.update({
                                    "filtering/total_groups": _total_groups,
                                    "filtering/filtered_groups": _filtered_groups,
                                    "filtering/filtered_samples": _filtered_samples,
                                    "filtering/filtered_groups_ratio": _filtered_groups / max(_total_groups, 1),
                                    "filtering/filtered_insufficient_valid": _filtered_insufficient,
                                    "filtering/filtered_all_same_reward": _filtered_all_same,
                                    "filtering/plan_b_rescued_groups": _rescued_by_plan_b,
                                    "filtering/effective_groups": _effective_groups,
                                    "filtering/valid_trajectories_per_group_mean": sum(_valid_counts) / max(len(_valid_counts), 1),
                                    "filtering/valid_trajectories_per_group_min": min(_valid_counts) if _valid_counts else 0,
                                    "filtering/valid_trajectories_per_group_max": max(_valid_counts) if _valid_counts else 0,
                                    "filtering/total_valid_trajectories": sum(_valid_counts),
                                })
                        
                        # ========================================
                        # === Group-in-Group Advantage Fusion ===
                        # === Added by anonymous@example.com ===
                        # ========================================
                        # Module 4 (need_summary V3): Fuse episode-level A_E with step-level A_S
                        # A = A_E_norm + omega * A_S_norm
                        # gig_step_advantages is set by the V3 GiG reward manager
                        if "gig_step_advantages" in batch.non_tensor_batch:
                            try:
                                from verl.utils.reward_score.process_reward_gig import align_and_fuse_advantages
                                
                                gig_step_advantages = batch.non_tensor_batch["gig_step_advantages"]
                                gig_omega_arr = batch.non_tensor_batch.get("gig_omega", None)
                                omega = float(gig_omega_arr[0]) if gig_omega_arr is not None else 0.3
                                
                                # === Dynamic Omega Scheduling ===
                                # When enable_dynamic_omega=True, omega linearly decays
                                # from omega_start to omega_end over training progress.
                                _enable_dynamic_omega = False
                                if "gig_enable_dynamic_omega" in batch.non_tensor_batch:
                                    _val = batch.non_tensor_batch["gig_enable_dynamic_omega"][0]
                                    _enable_dynamic_omega = bool(_val)
                                
                                if _enable_dynamic_omega:
                                    _omega_start = float(batch.non_tensor_batch.get("gig_omega_start", np.array([0.5]))[0])
                                    _omega_end = float(batch.non_tensor_batch.get("gig_omega_end", np.array([0.1]))[0])
                                    _warmup_ratio = float(batch.non_tensor_batch.get("gig_omega_warmup_ratio", np.array([0.0]))[0])
                                    _progress = self.global_steps / max(self.total_training_steps, 1)
                                    
                                    if _warmup_ratio > 0 and _progress < _warmup_ratio:
                                        # Warmup phase: linearly increase from 0 to omega_start
                                        omega = _omega_start * (_progress / _warmup_ratio)
                                    else:
                                        # Decay phase: linearly decay from omega_start to omega_end
                                        _decay_progress = (_progress - _warmup_ratio) / max(1.0 - _warmup_ratio, 1e-8)
                                        _decay_progress = min(max(_decay_progress, 0.0), 1.0)
                                        omega = _omega_start + (_omega_end - _omega_start) * _decay_progress
                                    
                                    print(
                                        f"[Dynamic Omega] step={self.global_steps}/{self.total_training_steps}, "
                                        f"progress={_progress:.3f}, omega={omega:.4f} "
                                        f"(start={_omega_start}, end={_omega_end}, warmup={_warmup_ratio})"
                                    )
                                
                                # Convert step advantages to list of lists
                                step_adv_list = []
                                for sa in gig_step_advantages:
                                    if isinstance(sa, (list, np.ndarray)) and len(sa) > 0:
                                        step_adv_list.append(list(sa))
                                    else:
                                        step_adv_list.append([])
                                
                                # Get current episode-level advantages (A_E)
                                episode_advantages = batch.batch["advantages"].cpu().numpy()
                                response_mask = batch.batch["response_mask"].cpu().numpy()

                                # === C4: advantage fusion mode switch ===
                                # "convex"               : A_final = (1-ω)·A_E_norm + ω·A_S_norm   (V4 RGC default)
                                # "additive"             : A_final = A_E_norm + ω·A_S_norm         (legacy V3 GiG)
                                # "gigpo_additive"       : A_final = A_E + w_s·A_S                 (gigpo reference — no extra z-norm)
                                # "convex_final_exempt"  : non-final tokens: (1-ω)·A_E_norm + ω·A_S_norm
                                #                          final-step tokens: A_E (unattenuated)     (V7 RGC-IR)
                                _fusion_mode = "convex"
                                if "gig_advantage_fusion_mode" in batch.non_tensor_batch:
                                    try:
                                        _fusion_mode = str(
                                            batch.non_tensor_batch["gig_advantage_fusion_mode"][0]
                                        ).lower()
                                    except Exception:
                                        _fusion_mode = "convex"
                                if _fusion_mode not in (
                                    "convex", "additive", "gigpo_additive", "convex_final_exempt"
                                ):
                                    logging.getLogger(__name__).warning(
                                        f"[Advantage Fusion] unknown mode '{_fusion_mode}', "
                                        f"falling back to 'convex'"
                                    )
                                    _fusion_mode = "convex"

                                # Build token-level A_S matrix once, used by both
                                # fusion modes AND by downstream ablation metrics.
                                # While iterating, also build a boolean mask of
                                # final-step tokens so the V7 convex_final_exempt
                                # branch can route those tokens to raw A_E.
                                _batch_sz, _seq_len = episode_advantages.shape
                                step_adv_token_level = np.zeros((_batch_sz, _seq_len), dtype=np.float32)
                                final_token_mask = np.zeros((_batch_sz, _seq_len), dtype=bool)
                                for _i, _steps in enumerate(step_adv_list):
                                    if not _steps:
                                        continue
                                    for _si in _steps:
                                        _adv = _si.get("step_advantage", 0.0)
                                        _ts = max(_si.get("token_start", 0), 0)
                                        _te = min(_si.get("token_end", _seq_len), _seq_len)
                                        step_adv_token_level[_i, _ts:_te] = _adv
                                        if _si.get("is_final", False):
                                            final_token_mask[_i, _ts:_te] = True

                                # Independent z-score normalization against the
                                # valid tokens only (response_mask>0), so the two
                                # signals have comparable magnitudes before fusion.
                                _vm = response_mask > 0
                                _ae_valid = episode_advantages[_vm]
                                _as_valid = step_adv_token_level[_vm]
                                _ae_std = float(np.std(_ae_valid)) if _ae_valid.size > 0 else 0.0
                                _as_std = float(np.std(_as_valid)) if _as_valid.size > 0 else 0.0
                                episode_advantages_norm = (
                                    episode_advantages / _ae_std if _ae_std > 1e-8 else episode_advantages
                                )
                                step_adv_norm = (
                                    step_adv_token_level / _as_std if _as_std > 1e-8 else step_adv_token_level
                                )

                                if _fusion_mode == "convex":
                                    # A_final = (1-ω)·A_E_norm + ω·A_S_norm
                                    _w = max(0.0, min(1.0, float(omega)))
                                    fused_advantages = (
                                        (1.0 - _w) * episode_advantages_norm
                                        + _w * step_adv_norm
                                    )
                                elif _fusion_mode == "convex_final_exempt":
                                    # V7 RGC-IR: non-final tokens do convex
                                    # combination; final-step tokens receive the
                                    # RAW (un-std-normalized) A_E so the answer
                                    # gradient is not attenuated.
                                    #   non-final:  (1-ω)·A_E_norm + ω·A_S_norm
                                    #   final    :  A_E                  (raw)
                                    # Rationale:
                                    #   * A_S = 0 on the final step BY DESIGN,
                                    #     not because outcome is uninformative;
                                    #   * plain convex mode would scale answer
                                    #     tokens by (1-ω), silently shrinking
                                    #     the outcome gradient.
                                    _w = max(0.0, min(1.0, float(omega)))
                                    conv = (
                                        (1.0 - _w) * episode_advantages_norm
                                        + _w * step_adv_norm
                                    )
                                    fused_advantages = np.where(
                                        final_token_mask,
                                        episode_advantages,  # raw A_E on answer tokens
                                        conv,
                                    )
                                elif _fusion_mode == "gigpo_additive":
                                    # GiGPO reference (core_gigpo.compute_gigpo_outcome_advantage):
                                    #   A_final = A_E + step_advantage_w * A_S
                                    # NO independent z-score renormalization: both A_E and A_S
                                    # are already mean/std-normalized within their own groups
                                    # (A_E by GRPO over uid, A_S by compute_gigpo_step_advantage_for_batch
                                    # over anchor-state clusters). Applying an extra global z-norm
                                    # here would double-normalize and change the effective
                                    # step_advantage_w. ``omega`` holds gigpo's ``step_advantage_w``.
                                    _ws = float(omega)
                                    fused_advantages = (
                                        episode_advantages
                                        + _ws * step_adv_token_level
                                    )
                                else:
                                    # Legacy additive: delegate to library for
                                    # strict bit-wise parity with old experiments.
                                    fused_advantages = align_and_fuse_advantages(
                                        episode_advantages=episode_advantages,
                                        step_advantages_per_sample=step_adv_list,
                                        response_mask=response_mask,
                                        omega=omega,
                                    )

                                # Write back to batch
                                batch.batch["advantages"] = torch.tensor(
                                    fused_advantages,
                                    dtype=batch.batch["advantages"].dtype,
                                    device=batch.batch["advantages"].device,
                                )

                                # Log metrics
                                has_step_adv = sum(1 for sa in step_adv_list if len(sa) > 0)
                                logging.getLogger(__name__).info(
                                    f"[GiG Advantage Fusion] mode={_fusion_mode}, "
                                    f"omega={omega:.3f}, "
                                    f"samples_with_step_adv={has_step_adv}/{len(step_adv_list)}"
                                )
                                valid_mask = response_mask > 0
                                fused_vals = fused_advantages[valid_mask] if valid_mask.sum() > 0 else np.array([0.0])
                                ae_vals = episode_advantages[valid_mask] if valid_mask.sum() > 0 else np.array([0.0])
                                as_vals = step_adv_token_level[valid_mask] if valid_mask.sum() > 0 else np.array([0.0])
                                metrics.update({
                                    "gig/omega": omega,
                                    "gig/fusion_mode_is_convex": 1.0 if _fusion_mode == "convex" else 0.0,
                                    "gig/fusion_mode_is_gigpo": 1.0 if _fusion_mode == "gigpo_additive" else 0.0,
                                    "gig/fusion_mode_is_convex_final_exempt": (
                                        1.0 if _fusion_mode == "convex_final_exempt" else 0.0
                                    ),
                                    "gig/samples_with_step_advantage": has_step_adv,
                                    "gig/fused_advantage_mean": float(np.mean(fused_vals)),
                                    "gig/fused_advantage_std": float(np.std(fused_vals)),
                                })

                                # --- V7-specific: final vs non-final token split ---
                                # Only informative for convex_final_exempt; for
                                # other modes we still expose them as 0 so the
                                # dashboard keys are stable across runs.
                                try:
                                    _final_mask_valid = final_token_mask & valid_mask
                                    _nonfinal_mask_valid = (~final_token_mask) & valid_mask
                                    _n_final = int(_final_mask_valid.sum())
                                    _n_total = int(valid_mask.sum())
                                    _final_ratio = float(_n_final) / max(1, _n_total)
                                    _ans_adv = (
                                        float(fused_advantages[_final_mask_valid].mean())
                                        if _n_final > 0 else 0.0
                                    )
                                    _nf_adv = (
                                        float(fused_advantages[_nonfinal_mask_valid].mean())
                                        if int(_nonfinal_mask_valid.sum()) > 0 else 0.0
                                    )
                                    metrics.update({
                                        "gig/final_token_ratio": _final_ratio,
                                        "gig/answer_advantage_mean": _ans_adv,
                                        "gig/nonfinal_advantage_mean": _nf_adv,
                                    })
                                except Exception:
                                    pass

                                # ========================================
                                # === GiG Ablation Metrics (方案A) ===
                                # === Compare A_E vs A_fused to validate grouping effectiveness ===
                                # ========================================
                                try:
                                    ablation_metrics = {}

                                    # --- 1. A_E vs A_S correlation ---
                                    # Low correlation => A_S provides new info not in A_E
                                    if len(ae_vals) > 1 and np.std(ae_vals) > 1e-8 and np.std(as_vals) > 1e-8:
                                        ae_as_corr = float(np.corrcoef(ae_vals, as_vals)[0, 1])
                                        if np.isnan(ae_as_corr):
                                            ae_as_corr = 0.0
                                        ablation_metrics["gig_ablation/ae_as_correlation"] = ae_as_corr
                                    else:
                                        ablation_metrics["gig_ablation/ae_as_correlation"] = 0.0

                                    # --- 2. Advantage direction change ratio ---
                                    # How many tokens changed advantage sign after fusion
                                    sign_ae = np.sign(ae_vals)
                                    sign_fused = np.sign(fused_vals)
                                    direction_changed = np.mean(sign_ae != sign_fused)
                                    ablation_metrics["gig_ablation/advantage_change_ratio"] = float(direction_changed)

                                    # --- 3. Advantage magnitude change ---
                                    ae_abs_mean = np.mean(np.abs(ae_vals))
                                    fused_abs_mean = np.mean(np.abs(fused_vals))
                                    ablation_metrics["gig_ablation/advantage_magnitude_change"] = (
                                        float(fused_abs_mean / ae_abs_mean) if ae_abs_mean > 1e-8 else 1.0
                                    )

                                    # --- 4. Step advantage signal sparsity ---
                                    ablation_metrics["gig_ablation/step_adv_nonzero_ratio"] = float(
                                        np.mean(np.abs(as_vals) > 1e-8)
                                    )
                                    as_positive = as_vals[np.abs(as_vals) > 1e-8]
                                    ablation_metrics["gig_ablation/step_adv_positive_ratio"] = (
                                        float(np.mean(as_positive > 0)) if len(as_positive) > 0 else 0.5
                                    )

                                    # --- 5. Effective omega (A_S contribution ratio) ---
                                    fused_std = np.std(fused_vals)
                                    # Compute the normalized A_S contribution
                                    as_std_raw = np.std(as_vals)
                                    as_norm_std = np.std(as_vals / as_std_raw) if as_std_raw > 1e-8 else 0.0
                                    ablation_metrics["gig_ablation/effective_omega"] = (
                                        float(omega * as_norm_std / fused_std) if fused_std > 1e-8 else 0.0
                                    )

                                    # --- 6. Direction agreement between A_E and A_S ---
                                    sign_as = np.sign(as_vals)
                                    # Only compare where both are nonzero
                                    both_nonzero = (np.abs(ae_vals) > 1e-8) & (np.abs(as_vals) > 1e-8)
                                    if both_nonzero.sum() > 0:
                                        agree = np.mean(sign_ae[both_nonzero] == sign_as[both_nonzero])
                                        conflict = np.mean(sign_ae[both_nonzero] != sign_as[both_nonzero])
                                        ablation_metrics["gig_ablation/reward_direction_agreement"] = float(agree)
                                        ablation_metrics["gig_ablation/conflicting_signal_ratio"] = float(conflict)
                                    else:
                                        ablation_metrics["gig_ablation/reward_direction_agreement"] = 0.0
                                        ablation_metrics["gig_ablation/conflicting_signal_ratio"] = 0.0

                                    # --- 7. Information entropy gain ---
                                    def _advantage_entropy(vals):
                                        abs_vals = np.abs(vals)
                                        total = abs_vals.sum()
                                        if total < 1e-8:
                                            return 0.0
                                        probs = abs_vals / total
                                        probs = probs[probs > 0]
                                        return float(-np.sum(probs * np.log(probs + 1e-12)))

                                    fused_entropy = _advantage_entropy(fused_vals)
                                    ae_entropy = _advantage_entropy(ae_vals)
                                    ablation_metrics["gig_ablation/fused_advantage_entropy"] = fused_entropy
                                    ablation_metrics["gig_ablation/ae_only_advantage_entropy"] = ae_entropy
                                    ablation_metrics["gig_ablation/entropy_gain"] = fused_entropy - ae_entropy

                                    metrics.update(ablation_metrics)
                                except Exception as e:
                                    logging.getLogger(__name__).warning(
                                        f"[GiG Ablation Metrics] Failed: {e}"
                                    )
                            except Exception as e:
                                logging.getLogger(__name__).error(f"[GiG Advantage Fusion] Failed: {e}")
                                logging.getLogger(__name__).error(traceback.format_exc())
                                # Fall back to episode-level advantage only (no modification)
                        # === End of Group-in-Group Advantage Fusion ===
                        
                        # ========================================
                        # === 特殊处理：超出token限制的样本 ===
                        # === Added by anonymous@example.com ===
                        # ========================================
                        # 对于 exceed_max_tokens=True 的样本：
                        # - 已参与优势计算（减少统计偏差）
                        # - 但不应参与 loss 更新（避免截断带来的噪声）
                        # 
                        # 实现方式：将这些样本的 response_mask 设置为 0
                        # 这样在 compute_policy_loss 中，这些样本不会贡献 loss
                        # 
                        if "rollout_extra_info" in batch.non_tensor_batch:
                            rollout_extra_info_list = batch.non_tensor_batch["rollout_extra_info"]
                            
                            # 找出所有 exceed_max_tokens=True 的样本索引
                            exceed_token_indices = []
                            for i, rollout_extra_info in enumerate(rollout_extra_info_list):
                                abnormal_flags = rollout_extra_info.get("abnormal_flags", {})
                                if abnormal_flags.get("exceed_max_tokens", False):
                                    exceed_token_indices.append(i)
                            
                            # 将这些样本的 response_mask 设置为 0
                            if len(exceed_token_indices) > 0:
                                original_response_mask = batch.batch["response_mask"].clone()
                                for idx in exceed_token_indices:
                                    batch.batch["response_mask"][idx] = 0.0
                                
                                # 记录有多少样本被排除
                                num_excluded = len(exceed_token_indices)
                                total_samples = len(rollout_extra_info_list)
                                print(
                                    f"[Advantage Optimization] Excluded {num_excluded}/{total_samples} samples "
                                    f"from loss update due to exceed_max_tokens (still used for advantage calculation)"
                                )
                                
                                # 统计被排除样本的信息（用于监控）
                                metrics.update({
                                    "training/samples_excluded_from_loss": num_excluded,
                                    "training/samples_excluded_ratio": num_excluded / total_samples if total_samples > 0 else 0.0,
                                })
                        # === End of Added by anonymous@example.com ===

                    _phase_log("adv")

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        _phase_log("update_actor", "entering actor weight update")
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        _phase_log("update_actor")

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                # 构造用于 dump 的 reward_extra_infos_dict
                                # 自动提取所有来自 reward manager 的字段
                                if 'reward_extra_infos_dict' not in locals():
                                    reward_extra_infos_dict = {}
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            
                            # 自动从 batch.non_tensor_batch 提取 reward manager 的自定义字段
                            # 这些字段已经通过 safe_convert_reward_extra_info_to_numpy 转换并存储
                            if 'reward_extra_infos_dict' not in locals():
                                reward_extra_infos_dict = {}
                            
                            known_reward_fields = [
                                'end_to_end_score', 'final_reward', 'abnormal_types',
                                'coverage_rate_raw', 'coverage_rate_normalized'
                            ]
                            for field_name in known_reward_fields:
                                if field_name in batch.non_tensor_batch:
                                    reward_extra_infos_dict.setdefault(
                                        field_name,
                                        batch.non_tensor_batch[field_name].tolist()
                                    )

                            # Add filtering/overlong markers for offline analysis
                            _n_samples = len(inputs)
                            _response_mask_sum = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
                            reward_extra_infos_dict.setdefault(
                                "is_filtered",
                                [1 if _response_mask_sum[i] < 1e-6 else 0 for i in range(_n_samples)]
                            )
                            if "rollout_extra_info" in batch.non_tensor_batch:
                                _overlong_flags = []
                                for _rei in batch.non_tensor_batch["rollout_extra_info"]:
                                    _af = _rei.get("abnormal_flags", {}) if isinstance(_rei, dict) else {}
                                    _overlong_flags.append(1 if _af.get("exceed_max_tokens", False) else 0)
                                reward_extra_infos_dict.setdefault("is_overlong", _overlong_flags)

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                # ========================================
                # === anonymous新增：从 batch_statistics 和 extra_info 提取监控指标 ===
                # ========================================
                
                # 1. 提取批次统计指标（batch_statistics）
                if "batch_statistics" in batch.non_tensor_batch:
                    # batch_statistics 是一个包含 batch_size 个相同 dict 的 np.array
                    # 取第一个元素即可（所有元素都相同）
                    batch_stats = batch.non_tensor_batch["batch_statistics"][0]
                    
                    # 1.1 异常样本相关指标
                    metrics.update({
                        "abnormal/normal_rate": batch_stats.get("normal_rate", 0.0),
                        "abnormal/repeated_query_rate": batch_stats.get("repeated_query_rate", 0.0),
                        "abnormal/tool_parse_error_rate": batch_stats.get("tool_parse_error_rate", 0.0),
                        "abnormal/exceed_max_turns_rate": batch_stats.get("exceed_max_turns_rate", 0.0),
                        "abnormal/exceed_max_tokens_rate": batch_stats.get("exceed_max_tokens_rate", 0.0),
                        "abnormal/excessive_tool_calls_rate": batch_stats.get("excessive_tool_calls_rate", 0.0),
                        "abnormal/search_error_rate": batch_stats.get("search_error_rate", 0.0),
                        "abnormal/unknown_tool_rate": batch_stats.get("unknown_tool_rate", 0.0),
                    })
                    
                    # 1.2 全体轨迹的 search steps 和 subq calls
                    metrics.update({
                        "trajectory_all/search_steps_avg": batch_stats.get("search_steps_avg", 0.0),
                        "trajectory_all/search_steps_median": batch_stats.get("search_steps_median", 0.0),
                        "trajectory_all/search_steps_max": batch_stats.get("search_steps_max", 0),
                        "trajectory_all/subq_calls_avg": batch_stats.get("subq_calls_avg", 0.0),
                        "trajectory_all/subq_calls_median": batch_stats.get("subq_calls_median", 0.0),
                        "trajectory_all/subq_calls_max": batch_stats.get("subq_calls_max", 0),
                    })
                
                # 2. 提取每个样本的轨迹指标（rollout_extra_info），计算正常样本的统计
                if "rollout_extra_info" in batch.non_tensor_batch:
                    rollout_extra_info_list = batch.non_tensor_batch["rollout_extra_info"]
                    
                    # 2.1 筛选正常样本
                    normal_search_steps = []
                    normal_subq_calls = []
                    normal_tool_call_attempts = []
                    
                    for rollout_extra_info in rollout_extra_info_list:
                        abnormal_flags = rollout_extra_info.get("abnormal_flags", {})
                        trajectory_metrics = rollout_extra_info.get("trajectory_metrics", {})
                        
                        # 判断是否为正常样本
                        is_abnormal = any([
                            abnormal_flags.get("excessive_tool_calls_per_turn", False),
                            abnormal_flags.get("tool_parse_error", False),
                            abnormal_flags.get("repeated_query", False),
                            abnormal_flags.get("exceed_max_turns", False),
                            abnormal_flags.get("exceed_max_tokens", False),
                            abnormal_flags.get("search_error", False),
                            abnormal_flags.get("unknown_tool_in_trajectory", False),
                        ])
                        
                        if not is_abnormal:
                            normal_search_steps.append(trajectory_metrics.get("search_steps", 0))
                            normal_subq_calls.append(trajectory_metrics.get("subq_calls", 0))
                            normal_tool_call_attempts.append(trajectory_metrics.get("tool_call_attempts", 0))
                    
                    # 2.2 计算正常样本的统计指标
                    if len(normal_search_steps) > 0:
                        metrics.update({
                            "trajectory_normal/search_steps_avg": float(np.mean(normal_search_steps)),
                            "trajectory_normal/search_steps_median": float(np.median(normal_search_steps)),
                            "trajectory_normal/search_steps_max": int(np.max(normal_search_steps)),
                            "trajectory_normal/subq_calls_avg": float(np.mean(normal_subq_calls)),
                            "trajectory_normal/subq_calls_median": float(np.median(normal_subq_calls)),
                            "trajectory_normal/subq_calls_max": int(np.max(normal_subq_calls)),
                            "trajectory_normal/tool_call_attempts_avg": float(np.mean(normal_tool_call_attempts)),
                            "trajectory_normal/tool_call_attempts_median": float(np.median(normal_tool_call_attempts)),
                            "trajectory_normal/tool_call_attempts_max": int(np.max(normal_tool_call_attempts)),
                        })
                    
                
                # 3. 计算 reject sampling 相关指标（reward = 1 的样本比例）
                if "token_level_rewards" in batch.batch:
                    reward_tensor = batch.batch["token_level_rewards"]
                    response_mask = batch.batch.get("response_mask", None)
                    
                    # 计算序列级别的 reward（对整个序列求和，与 compute_data_metrics 保持一致）
                    if response_mask is not None:
                        # 使用 sum(-1) 计算序列 reward，与 metric_utils.py 中的 compute_data_metrics 一致
                        sequence_rewards = reward_tensor.sum(-1)
                        
                        # 统计 reward = 1（acc = 1）的样本比例
                        reward_eq_1_mask = (sequence_rewards >= 0.999)  # 使用 0.999 避免浮点精度问题
                        reward_eq_1_ratio = torch.mean(reward_eq_1_mask.float()).item()
                        
                        metrics.update({
                            "reward/acc_eq_1_ratio": reward_eq_1_ratio,
                            "reward/acc_eq_1_count": int(torch.sum(reward_eq_1_mask).item()),
                        })
                        
                        # 计算 reward 的分布
                        metrics.update({
                            "reward/sequence_reward_mean": torch.mean(sequence_rewards).item(),
                            "reward/sequence_reward_std": torch.std(sequence_rewards).item(),
                            "reward/sequence_reward_min": torch.min(sequence_rewards).item(),
                            "reward/sequence_reward_max": torch.max(sequence_rewards).item(),
                        })
                
                # 4. PPO训练相关指标：importance ratio 和 TIS ratio 的分位数
                if "old_log_probs" in batch.batch and "rollout_log_probs" in batch.batch:
                    old_log_probs = batch.batch["old_log_probs"]
                    rollout_log_probs = batch.batch["rollout_log_probs"]
                    response_mask = batch.batch.get("response_mask", None)
                    
                    if response_mask is not None:
                        # 计算 importance ratio: π_θ / π_old
                        log_ratio = old_log_probs - rollout_log_probs
                        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)  # 防止数值溢出
                        importance_ratio = torch.exp(log_ratio)
                        
                        # 只统计有效 token 的 ratio
                        valid_importance_ratio = importance_ratio[response_mask.bool()]
                        
                        if valid_importance_ratio.numel() > 0:
                            # 计算分位数
                            importance_ratio_sorted = torch.sort(valid_importance_ratio)[0]
                            n = importance_ratio_sorted.numel()
                            
                            metrics.update({
                                "ppo/importance_ratio_p5": importance_ratio_sorted[int(n * 0.05)].item(),
                                "ppo/importance_ratio_p50": importance_ratio_sorted[int(n * 0.50)].item(),
                                "ppo/importance_ratio_p95": importance_ratio_sorted[int(n * 0.95)].item(),
                                "ppo/importance_ratio_mean": torch.mean(valid_importance_ratio).item(),
                                "ppo/importance_ratio_max": torch.max(valid_importance_ratio).item(),
                                "ppo/importance_ratio_min": torch.min(valid_importance_ratio).item(),
                            })
                            
                            # 计算超出 clip range 的 token 占比（使用不对称的 clip higher 策略）
                            clip_ratio = self.config.actor_rollout_ref.actor.get("clip_ratio", 0.2)
                            # clip_higher 策略：upper bound 更宽松，lower bound 更严格
                            clip_lower = 1 - clip_ratio  # 例如 0.8
                            clip_upper = 1 + 2 * clip_ratio  # 例如 1.4（更宽松）
                            
                            clipped_lower_mask = valid_importance_ratio < clip_lower
                            clipped_upper_mask = valid_importance_ratio > clip_upper
                            clipped_mask = clipped_lower_mask | clipped_upper_mask
                            
                            metrics.update({
                                "ppo/importance_ratio_clipped_ratio": torch.mean(clipped_mask.float()).item(),
                                "ppo/importance_ratio_clipped_lower_ratio": torch.mean(clipped_lower_mask.float()).item(),
                                "ppo/importance_ratio_clipped_upper_ratio": torch.mean(clipped_upper_mask.float()).item(),
                                "ppo/clip_lower_bound": clip_lower,
                                "ppo/clip_upper_bound": clip_upper,
                            })
                
                # 5. TIS (Token Importance Sampling) ratio 分位数
                # TIS ratio = π_rollout / π_ref
                # 用于检测训推不一致导致的 token mismatch：
                # - rollout 后端使用 sglang/vllm
                # - 训练后端使用 MSDP/Megatron
                # - 即使参数完全相同，不同后端也可能导致 π_old / π_old ≠ 1
                # - 监控此指标可检测因训推不一致导致的训练不稳定
                if self.use_reference_policy and "ref_log_prob" in batch.batch and "rollout_log_probs" in batch.batch:
                    ref_log_prob = batch.batch["ref_log_prob"]
                    rollout_log_probs = batch.batch["rollout_log_probs"]
                    response_mask = batch.batch.get("response_mask", None)
                    
                    if response_mask is not None:
                        # 计算 TIS ratio: π_rollout / π_ref
                        log_tis_ratio = rollout_log_probs - ref_log_prob
                        log_tis_ratio = torch.clamp(log_tis_ratio, min=-50.0, max=50.0)  # 更宽的 clamp 范围
                        tis_ratio = torch.exp(log_tis_ratio)
                        
                        # 只统计有效 token 的 TIS ratio
                        valid_tis_ratio = tis_ratio[response_mask.bool()]
                        
                        if valid_tis_ratio.numel() > 0:
                            # 计算分位数
                            tis_ratio_sorted = torch.sort(valid_tis_ratio)[0]
                            n = tis_ratio_sorted.numel()
                            
                            metrics.update({
                                "ppo/tis_ratio_p5": tis_ratio_sorted[int(n * 0.05)].item(),
                                "ppo/tis_ratio_p50": tis_ratio_sorted[int(n * 0.50)].item(),
                                "ppo/tis_ratio_p95": tis_ratio_sorted[int(n * 0.95)].item(),
                                "ppo/tis_ratio_p100": torch.max(valid_tis_ratio).item(),
                                "ppo/tis_ratio_mean": torch.mean(valid_tis_ratio).item(),
                                "ppo/tis_ratio_min": torch.min(valid_tis_ratio).item(),
                            })
                            
                            # 统计超过阈值的 token 占比
                            # - 最小值可能到 1e-15（极小概率 token）
                            # - 最大值若显著高于 10 通常需要优先排查 token mismatch
                            high_tis_mask = valid_tis_ratio > 10.0
                            low_tis_mask = valid_tis_ratio < 1e-10  # 监控极小值
                            
                            metrics.update({
                                "ppo/tis_ratio_over_10_ratio": torch.mean(high_tis_mask.float()).item(),
                                "ppo/tis_ratio_under_1e10_ratio": torch.mean(low_tis_mask.float()).item(),
                                "ppo/tis_ratio_over_10_count": int(torch.sum(high_tis_mask).item()),
                            })
                
                # ========================================
                # === C-GRPO 专用指标统计 ===
                # === Added by anonymous@example.com ===
                # ========================================
                # 
                # 该部分统计 C-GRPO 训练过程的关键指标，包括：
                # 1. normal_trajectory_rate: 正常轨迹比例
                # 2. end_to_end_reward: 端到端奖励统计
                # 3. rubric_raw/rubric_norm: Rubric奖励统计（只在做对的轨迹中计算）
                # 4. r_identify/r_support/r_connect: 三步评估指标
                # 5. rubric_error_types: Rubric错误类型统计
                #
                # 数据类型说明：
                # - batch.non_tensor_batch中的所有字段都是numpy数组（通过safe_convert_reward_extra_info_to_numpy转换）
                # - end_to_end_score, rubric_reward_raw等：np.ndarray[float]
                # - abnormal_types：np.ndarray[object]，每个元素是List[str]
                # - rubric_error_types：np.ndarray[str]
                #
                if "end_to_end_score" in batch.non_tensor_batch:
                    try:
                        from verl.utils.reward_score.trajectory_reward_v2_cgrpo import compute_cgrpo_metrics
                        
                        # 提取所需数据（使用.get()方式，增加鲁棒性）
                        # 注意：这些字段都已经是numpy数组，需要转换为Python列表
                        end_to_end_scores = batch.non_tensor_batch.get("end_to_end_score", [])
                        rubric_rewards_raw = batch.non_tensor_batch.get("rubric_reward_raw", [])
                        rubric_rewards_norm = batch.non_tensor_batch.get("rubric_reward_normalized", [])
                        r_identify_list = batch.non_tensor_batch.get("r_identify", [])
                        r_support_list = batch.non_tensor_batch.get("r_support", [])
                        r_connect_list = batch.non_tensor_batch.get("r_connect", [])
                        rubric_error_types = batch.non_tensor_batch.get("rubric_error_types", [])
                        abnormal_types_list = batch.non_tensor_batch.get("abnormal_types", [])
                        
                        # 判断是否为正常轨迹（abnormal_types为空列表的样本）
                        # 注意：abnormal_types_list是np.ndarray(dtype=object)，每个元素是List[str]
                        is_normal_trajectory = []
                        for abnormal_types in abnormal_types_list:
                            # abnormal_types 可能是列表、numpy数组或tensor
                            if isinstance(abnormal_types, list):
                                is_normal_trajectory.append(len(abnormal_types) == 0)
                            elif isinstance(abnormal_types, (np.ndarray, torch.Tensor)):
                                # 如果是numpy数组或tensor，转换为列表
                                is_normal_trajectory.append(len(list(abnormal_types)) == 0)
                            else:
                                # 其他情况，视为异常
                                is_normal_trajectory.append(False)
                        
                        # 转换为Python列表（确保类型正确）
                        def safe_to_list(data):
                            """安全地转换为Python列表
                            
                            处理从DataProto.non_tensor_batch中提取的numpy数组，
                            将其转换为Python列表供compute_cgrpo_metrics使用。
                            """
                            if isinstance(data, (list, tuple)):
                                return list(data)
                            elif isinstance(data, np.ndarray):
                                return data.tolist()
                            elif isinstance(data, torch.Tensor):
                                return data.cpu().tolist()
                            else:
                                return list(data) if hasattr(data, '__iter__') else []
                        
                        end_to_end_scores = safe_to_list(end_to_end_scores)
                        rubric_rewards_raw = safe_to_list(rubric_rewards_raw)
                        rubric_rewards_norm = safe_to_list(rubric_rewards_norm)
                        r_identify_list = safe_to_list(r_identify_list)
                        r_support_list = safe_to_list(r_support_list)
                        r_connect_list = safe_to_list(r_connect_list)
                        rubric_error_types = safe_to_list(rubric_error_types)
                        
                        # 计算C-GRPO指标
                        cgrpo_metrics = compute_cgrpo_metrics(
                            end_to_end_rewards=end_to_end_scores,
                            rubric_rewards_raw=rubric_rewards_raw,
                            rubric_rewards_norm=rubric_rewards_norm,
                            r_identify_list=r_identify_list,
                            r_support_list=r_support_list,
                            r_connect_list=r_connect_list,
                            is_normal_trajectory=is_normal_trajectory,
                            rubric_error_types=rubric_error_types
                        )
                        
                        # 添加到metrics
                        # 注意：这里只需要 metrics.update()，不需要额外的 print 或 log 输出
                        # 所有指标会在后续统一通过 logger.log(metrics, step) 记录到 wandb/tensorboard/console
                        # 这是官方代码的标准模式（参考 verl/verl/trainer/ppo/ray_trainer.py:1204, 1244, 1278等）
                        metrics.update(cgrpo_metrics)
                        
                    except Exception as e:
                        # 异常情况下不记录任何指标，避免污染metrics字典
                        # 如需调试，可以临时取消下面的注释
                        # import traceback
                        # print(f"[C-GRPO Metrics] 计算失败: {e}")
                        # print(traceback.format_exc())
                        pass
                # === End of C-GRPO Metrics ===

                # ========================================
                # === Step Summary Logger: compute extra monitoring metrics and print summary ===
                # ========================================
                try:
                    extra_monitoring_metrics = step_summary_logger.log_step(
                        global_step=self.global_steps,
                        epoch=epoch,
                        timing_raw=timing_raw,
                        batch=batch,
                        metrics=metrics,
                        config=self.config,
                    )
                    metrics.update(extra_monitoring_metrics)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[StepSummaryLogger] Failed: {e}")

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
