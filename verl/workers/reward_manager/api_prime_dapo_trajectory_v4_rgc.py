#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Reward Manager - V4 RGC
(Rubric-Guided Grouping with Counterfactual Reward + Plan B length penalty)

Inherits from V3 GiG and reuses ALL of V3's machinery (abnormal detection,
C-GRPO scoring, parallel Phase A state evaluation, token mapping, overlong
isolation, non_tensor_batch hand-off to ray_trainer).

The only training-path change is the Phase B function: we transparently
substitute GiG's `compute_process_reward_phase_b` with RGC's
`compute_rgc_phase_b` for the duration of each __call__.

Downstream advantage fusion in `ray_trainer.py` (see `advantage_fusion_mode`):
    "convex"   : A_final = (1 - omega) * A_E_norm + omega * A_S_norm  (default)
    "additive" : A_final = A_E_norm + omega * A_S_norm                (legacy)
where A_S comes from within-group R_final comparison plus Plan B's
intra-group length z-score penalty (when enabled).

Author: anonymous
Version: 4.1.0
Date: 2026-04-21
"""

import logging
import os
from typing import Any

import yaml

from verl.workers.reward_manager import register
from verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig import (
    ApiPrimeDapoTrajectoryV3GiGRewardManager,
)

# Module we will monkey-patch
from verl.utils.reward_score import process_reward_gig as _gig_mod
from verl.utils.reward_score.process_reward_rgc import compute_rgc_phase_b

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Candidate locations for the RGC config file. First existing wins.
_RGC_CONFIG_CANDIDATES = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../..",
        "trainer/config/reward_dapo_trajectory_v4_rgc.yaml",
    ),
]


def _load_rgc_config_overrides() -> dict:
    for p in _RGC_CONFIG_CANDIDATES:
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            try:
                with open(p_abs, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                logger.info(f"[RGC] loaded config overrides from {p_abs}")
                return cfg
            except Exception as e:
                logger.warning(f"[RGC] failed to load {p_abs}: {e}")
    logger.warning("[RGC] no RGC config file found, using built-in defaults")
    return {}


# Plan B default values (used when no config file is present)
_RGC_BUILTIN_DEFAULTS = {
    "length_penalty_mode": "group_zscore",
    "length_penalty_alpha": 0.3,
    "length_penalty_target": 8,
    "length_min_group_size": 2,
    "length_excl_correct_only": False,
    "efficiency_penalty_alpha": 0.0,
    "process_reward_omega": 0.3,
    "advantage_fusion_mode": "convex",
    "external_eval_temperature": 0.0,
    "enable_process_reward": True,
}


@register("apiprimedapotrajectoryv4rgc")
class ApiPrimeDapoTrajectoryV4RGCRewardManager(ApiPrimeDapoTrajectoryV3GiGRewardManager):
    """RGC Reward Manager (V4).

    Reuses V3's entire __call__ pipeline; only Phase B is swapped for RGC.
    No core_algos.py changes required: downstream ray_trainer applies
    `A = A_E + omega * A_S` just as it does for V3 GiG.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Merge RGC-specific config into self.config
        rgc_cfg = _load_rgc_config_overrides()

        # Start with built-in defaults, overlay file, preserve user-provided
        merged = dict(_RGC_BUILTIN_DEFAULTS)
        merged.update(rgc_cfg)

        # GiG-specific knobs that RGC does not use - remove from config so
        # downstream code does not pick up stale values.
        for k in ("process_reward_gamma", "cgrpo_alpha"):
            if k in self.config:
                self.config.pop(k, None)

        # Overlay RGC config onto existing self.config (API endpoints etc.
        # from V3 stay as-is if not overridden).
        for k, v in merged.items():
            self.config[k] = v

        # === P1 fix: re-sync V3 dump-related member attributes ============
        # V3.__init__ already fixed self.dump_grouping / self.dump_dir /
        # self.dump_max_queries / self.dump_step_text_max_chars from the V3
        # YAML BEFORE our RGC overlay ran. Without this re-sync, V3's
        # __call__ would still dump to `gig_grouping_details` and ignore
        # our RGC-specific dump config.
        self.dump_grouping = bool(self.config.get("dump_grouping_details", False))
        self.dump_max_queries = int(self.config.get("dump_max_queries", 0))
        self.dump_step_text_max_chars = int(
            self.config.get("dump_step_text_max_chars", 0)
        )
        _base_dump_dir = self.config.get("dump_dir", "") or ""
        if _base_dump_dir and self.dump_grouping:
            from datetime import datetime as _dt
            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            self.dump_dir = os.path.join(_base_dump_dir, f"run_{_ts}")
            try:
                os.makedirs(self.dump_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"[RGC] failed to create dump dir: {e}")
        else:
            self.dump_dir = _base_dump_dir
        # ==================================================================

        logger.info("✓ ApiPrimeDapoTrajectoryV4RGCRewardManager (Plan B) initialized")
        logger.info(
            f"  RGC params: omega={self.config.get('process_reward_omega', 0.3)} | "
            f"fusion_mode={self.config.get('advantage_fusion_mode', 'convex')} | "
            f"length_mode={self.config.get('length_penalty_mode', 'group_zscore')} | "
            f"alpha={self.config.get('length_penalty_alpha', 0.3)} | "
            f"target={self.config.get('length_penalty_target', 8)} | "
            f"min_group={self.config.get('length_min_group_size', 2)} | "
            f"excl_wrong={self.config.get('length_excl_correct_only', False)}"
        )
        logger.info(
            f"  RGC dump: enabled={self.dump_grouping} dir={self.dump_dir!r} "
            f"max_queries={self.dump_max_queries}"
        )

        # === C3 startup assertion: external eval temperature contract ===
        # Both Phase A step-state judge and C-GRPO answer judge MUST use
        # temperature=0 to guarantee idempotent rubric evaluation, so that
        # (uid, rubric_state) group keys are stable across mini-batches.
        _ext_temp = float(self.config.get("external_eval_temperature", 0.0))
        if abs(_ext_temp) > 1e-6:
            logger.warning(
                f"[RGC][C3 CONTRACT VIOLATION] external_eval_temperature={_ext_temp} "
                f"is NOT 0. This breaks rubric idempotency and may cause group-key "
                f"drift across mini-batches. Revert to 0.0 unless running an ablation."
            )
        else:
            logger.info(
                "[RGC][C3] external_eval_temperature=0.0 ✓ (step judge & answer "
                "judge idempotent)"
            )

    # ------------------------------------------------------------------
    # Transparent Phase B swap
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """Swap GiG's Phase B for RGC's during this reward computation only.

        This keeps V3's entire machinery intact (abnormal detection, C-GRPO,
        Phase A state evaluation, overlong isolation, token mapping,
        non_tensor_batch hand-off to ray_trainer) while routing the
        per-batch step-advantage math through RGC + Plan B.
        """
        cfg_snapshot = dict(self.config)

        def _rgc_phase_b_proxy(
            phase_a_result,
            trajectories,
            questions,
            rubrics_list,
            outcome_rewards,
            uid_list,
            gamma=0.95,
            global_step=0,
            dump_grouping=False,
            dump_dir="",
            dump_max_queries=0,
            dump_step_text_max_chars=0,
            omega=0.5,
            rank=0,
        ):
            return compute_rgc_phase_b(
                phase_a_result=phase_a_result,
                trajectories=trajectories,
                questions=questions,
                rubrics_list=rubrics_list,
                outcome_rewards=outcome_rewards,
                uid_list=uid_list,
                config=cfg_snapshot,
                gamma=gamma,
                global_step=global_step,
                dump_grouping=dump_grouping,
                dump_dir=dump_dir,
                dump_max_queries=dump_max_queries,
                dump_step_text_max_chars=dump_step_text_max_chars,
                omega=omega,
                rank=rank,
            )

        # Bind the proxy by name at two places:
        #   (a) on the gig module, so any future accesses see it;
        #   (b) on the v3 reward manager module, because it imported the
        #       function by name (`from ... import compute_process_reward_phase_b`)
        #       and therefore has its own local binding.
        import verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig as _v3_mod

        orig_on_gig_mod = getattr(_gig_mod, "compute_process_reward_phase_b", None)
        orig_on_v3_mod = getattr(_v3_mod, "compute_process_reward_phase_b", None)

        _gig_mod.compute_process_reward_phase_b = _rgc_phase_b_proxy
        _v3_mod.compute_process_reward_phase_b = _rgc_phase_b_proxy

        try:
            result = super().__call__(*args, **kwargs)
        finally:
            # Restore to avoid leaking the monkey-patch to other reward managers
            if orig_on_gig_mod is not None:
                _gig_mod.compute_process_reward_phase_b = orig_on_gig_mod
            if orig_on_v3_mod is not None:
                _v3_mod.compute_process_reward_phase_b = orig_on_v3_mod

        # ==========================================================
        # C4: advertise advantage_fusion_mode to ray_trainer via
        # non_tensor_batch. V3's __call__ has already populated
        # `gig_omega`, `gig_enable_dynamic_omega`, etc. We append the
        # fusion-mode hint here so that ray_trainer can switch between
        # "convex" (default) and "additive" (legacy) fusion.
        # ==========================================================
        try:
            import numpy as _np
            # `data` is always the first positional argument in V3's __call__
            _data = args[0] if args else kwargs.get("data", None)
            if _data is not None and hasattr(_data, "non_tensor_batch"):
                _bsz = len(_data.non_tensor_batch.get("uid", [])) or 1
                _data.non_tensor_batch["gig_advantage_fusion_mode"] = _np.array(
                    [str(cfg_snapshot.get("advantage_fusion_mode", "convex"))] * _bsz,
                    dtype=object,
                )
        except Exception as _e:
            logger.warning(f"[RGC] failed to set gig_advantage_fusion_mode: {_e}")

        return result
