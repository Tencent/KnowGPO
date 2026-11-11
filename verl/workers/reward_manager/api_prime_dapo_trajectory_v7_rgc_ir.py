#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Reward Manager - V7 RGC-IR
(Rubric-Grouped Immediate Reward with Final-Step Exemption)

V7 combines:
  * RGC's rubric-state grouping (from V4) -> group_key = (uid, s_{t-1})
  * Immediate rubric increment as step value:
        r_t = ||s_t||_1 - ||s_{t-1}||_1      (non-final steps only)
  * Final answer step: NOT grouped, A_S hard-pinned to 0
  * Advantage fusion: "convex_final_exempt":
        non-final tokens: (1 - ω)·Â_E + ω·Â_S
        final    tokens: A_E             (raw, unattenuated)

V7 reuses V4's entire __call__ pipeline (abnormal detection, C-GRPO, Phase A
state evaluation, overlong isolation, token mapping, non_tensor_batch hand-off
to ray_trainer). The ONLY behavioural changes vs V4 are:
  1. Phase B function is replaced by compute_rgc_ir_phase_b
  2. advantage_fusion_mode is forced to "convex_final_exempt"
  3. length penalty is forced off (V7 has no length term by design)

V7's config is layered:
    reward_dapo_trajectory_v4_rgc.yaml       <- loaded by V4.__init__ (base)
    reward_dapo_trajectory_v7_rgc_ir.yaml    <- loaded by V7.__init__ (overlay)

Author: anonymous
Version: 7.0.0
Date: 2026-05-08
"""

import logging
import os
from typing import Any

import yaml

from verl.workers.reward_manager import register
from verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig import (
    ApiPrimeDapoTrajectoryV3GiGRewardManager,
)
from verl.workers.reward_manager.api_prime_dapo_trajectory_v4_rgc import (
    ApiPrimeDapoTrajectoryV4RGCRewardManager,
)

# Module we will monkey-patch (same as V4/V6)
from verl.utils.reward_score import process_reward_gig as _gig_mod
from verl.utils.reward_score.process_reward_rgc import compute_rgc_ir_phase_b

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Candidate locations for the V7 config overlay. First existing wins.
_V7_CONFIG_CANDIDATES = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../..",
        "trainer/config/reward_dapo_trajectory_v7_rgc_ir.yaml",
    ),
]


def _load_v7_config_overrides() -> dict:
    for p in _V7_CONFIG_CANDIDATES:
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            try:
                with open(p_abs, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                logger.info(f"[V7 RGC-IR] loaded config overrides from {p_abs}")
                return cfg
            except Exception as e:
                logger.warning(f"[V7 RGC-IR] failed to load {p_abs}: {e}")
    logger.warning(
        "[V7 RGC-IR] no V7 config file found, using built-in defaults"
    )
    return {}


# Built-in V7 defaults (used when no V7 yaml is present).
_V7_BUILTIN_DEFAULTS = {
    "value_function_mode": "rgc_immediate_reward",
    "advantage_fusion_mode": "convex_final_exempt",
    # Process-reward weight on non-final tokens (matches V4 convex ω).
    "process_reward_omega": 0.3,
    # V7 NEVER uses a length penalty: the telescoping sum of r_t already
    # encodes a short-path preference, and any extra length term would be
    # double-counting.
    "length_penalty_mode": "none",
    "length_penalty_alpha": 0.0,
    "length_penalty_target": 0,
    "length_min_group_size": 2,
    "length_excl_correct_only": False,
    "efficiency_penalty_alpha": 0.0,
    # Disable dynamic omega unless the user explicitly flips it on.
    "enable_dynamic_omega": False,
    "omega_start": 0.3,
    "omega_end": 0.3,
    "omega_warmup_ratio": 0.0,
    # Disabled by default in the public training chain; enable only with a safe path.
    "dump_dir": "",
}


@register("apiprimedapotrajectoryv7rgcir")
class ApiPrimeDapoTrajectoryV7RGCIRRewardManager(
    ApiPrimeDapoTrajectoryV4RGCRewardManager
):
    """V7 RGC-IR Reward Manager.

    Flow:
        1. V4.__init__ runs first, loading the RGC yaml and setting up API
           clients, dump paths, etc.
        2. We overlay the V7 yaml on top of self.config, flipping:
               value_function_mode    -> "rgc_immediate_reward"
               advantage_fusion_mode  -> "convex_final_exempt"
               process_reward_omega   -> 0.3
               length_penalty_mode    -> "none"
        3. In __call__, we route Phase B to `compute_rgc_ir_phase_b` and
           advertise `gig_advantage_fusion_mode="convex_final_exempt"` to
           ray_trainer so the token-level fusion uses the V7 branch.
    """

    def __init__(self, *args, **kwargs):
        # Run V4's full init first (loads RGC yaml, sets up dump dirs, etc.).
        super().__init__(*args, **kwargs)

        # Overlay V7-specific config.
        v7_file_cfg = _load_v7_config_overrides()
        merged_v7 = dict(_V7_BUILTIN_DEFAULTS)
        merged_v7.update(v7_file_cfg)

        for k, v in merged_v7.items():
            self.config[k] = v

        # --- V7 hard contracts (warn-and-force on mismatch) ---
        vf_mode = str(self.config.get("value_function_mode", "rgc_immediate_reward")).lower()
        if vf_mode != "rgc_immediate_reward":
            logger.warning(
                f"[V7 RGC-IR] value_function_mode={vf_mode!r} (expected "
                f"'rgc_immediate_reward'). V7 will still route Phase B to "
                f"compute_rgc_ir_phase_b; if you want the V4 / V6 value form "
                f"use the respective reward manager instead."
            )

        fusion_mode = str(self.config.get("advantage_fusion_mode", "")).lower()
        if fusion_mode != "convex_final_exempt":
            logger.warning(
                f"[V7 RGC-IR][C1 CONTRACT] advantage_fusion_mode={fusion_mode!r} "
                f"is NOT 'convex_final_exempt'. V7's final-step-exempt semantics "
                f"require this fusion mode. Forcing it."
            )
            self.config["advantage_fusion_mode"] = "convex_final_exempt"

        # V7 must NOT emit any length penalty at any time; silently force if user forgot.
        if str(self.config.get("length_penalty_mode", "none")).lower() != "none":
            logger.warning(
                "[V7 RGC-IR] length_penalty_mode is not 'none'; forcing to 'none'. "
                "V7 already encodes short-path preference via telescoping r_t."
            )
            self.config["length_penalty_mode"] = "none"
            self.config["length_penalty_alpha"] = 0.0

        # --- Rebuild dump dir from the V7 dump_dir if provided ---
        # (V4 already set self.dump_dir from the RGC yaml; override so V7
        # offline logs are isolated from V4/V6.)
        _v7_dump_dir = self.config.get("dump_dir", "") or ""
        if _v7_dump_dir and self.dump_grouping:
            from datetime import datetime as _dt
            _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            self.dump_dir = os.path.join(_v7_dump_dir, f"run_{_ts}")
            try:
                os.makedirs(self.dump_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"[V7 RGC-IR] failed to create dump dir: {e}")

        logger.info(
            "✓ ApiPrimeDapoTrajectoryV7RGCIRRewardManager initialized"
        )
        logger.info(
            f"  V7 params: value_function_mode="
            f"{self.config.get('value_function_mode', 'rgc_immediate_reward')!r} | "
            f"fusion_mode={self.config.get('advantage_fusion_mode', 'convex_final_exempt')!r} | "
            f"omega={self.config.get('process_reward_omega', 0.3)} | "
            f"length_penalty=OFF"
        )
        logger.info(
            f"  V7 dump: enabled={self.dump_grouping} dir={self.dump_dir!r} "
            f"max_queries={self.dump_max_queries}"
        )

    # ------------------------------------------------------------------
    # Phase B routing - swap to the V7 RGC-IR implementation
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """Route Phase B to compute_rgc_ir_phase_b for this batch.

        Mirrors V6's design: monkey-patch GiG's Phase B symbol on both the
        GiG module and the V3 reward-manager module (because V3 imported the
        function by name and therefore has its own local binding), call
        V3.__call__ directly to bypass V4's own patching, then restore.
        """
        cfg_snapshot = dict(self.config)

        def _phase_b_proxy(
            phase_a_result,
            trajectories,
            questions,
            rubrics_list,
            outcome_rewards,
            uid_list,
            gamma=0.0,
            global_step=0,
            dump_grouping=False,
            dump_dir="",
            dump_max_queries=0,
            dump_step_text_max_chars=0,
            omega=0.3,
            rank=0,
        ):
            return compute_rgc_ir_phase_b(
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

        import verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig as _v3_mod

        orig_on_gig_mod = getattr(_gig_mod, "compute_process_reward_phase_b", None)
        orig_on_v3_mod = getattr(_v3_mod, "compute_process_reward_phase_b", None)

        _gig_mod.compute_process_reward_phase_b = _phase_b_proxy
        _v3_mod.compute_process_reward_phase_b = _phase_b_proxy

        try:
            # Call V3.__call__ directly, bypassing V4.__call__ (which would
            # re-install compute_rgc_phase_b and clobber our proxy).
            result = ApiPrimeDapoTrajectoryV3GiGRewardManager.__call__(
                self, *args, **kwargs
            )
        finally:
            if orig_on_gig_mod is not None:
                _gig_mod.compute_process_reward_phase_b = orig_on_gig_mod
            if orig_on_v3_mod is not None:
                _v3_mod.compute_process_reward_phase_b = orig_on_v3_mod

        # Advertise fusion mode to ray_trainer (same contract as V4/V6).
        try:
            import numpy as _np
            _data = args[0] if args else kwargs.get("data", None)
            if _data is not None and hasattr(_data, "non_tensor_batch"):
                _bsz = len(_data.non_tensor_batch.get("uid", [])) or 1
                _data.non_tensor_batch["gig_advantage_fusion_mode"] = _np.array(
                    [str(cfg_snapshot.get("advantage_fusion_mode", "convex_final_exempt"))] * _bsz,
                    dtype=object,
                )
        except Exception as _e:
            logger.warning(
                f"[V7 RGC-IR] failed to set gig_advantage_fusion_mode: {_e}"
            )

        return result
