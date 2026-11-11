#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Only Reward Manager

基于端到端奖励（browsecomp_zh.evaluate）的奖励管理器，不使用轨迹覆盖率奖励。

核心特性：
- 只计算端到端奖励（答案正确性）
- 处理格式错误和超长 rollout（从 sglang rollout 读取异常标记）
- 异常样本直接给 0 奖励

与 Trajectory Reward V1 Naive 的差异：
- ❌ 不计算轨迹覆盖率奖励
- ❌ 不进行中间事实识别
- ❌ 不需要组内归一化
- ✅ 保留所有异常处理逻辑（与 trajectory_reward_v1_naive 相同）

作者: anonymous@example.com
版本: 1.0.0
日期: 2026-01-01
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional, List, Dict, Any
import traceback
from collections import defaultdict

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.browsecomp_zh import compute_score as browsecomp_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def single_compute_end_to_end_score(
    config: Dict[str, Any],
    prompts_str: str,
    sequences_str: str,
    ground_truth: Dict[str, Any],
    data_source: str,
    task_extra_info: Dict[str, Any],
    executor: ThreadPoolExecutor,
    timeout: float = 6000.0,
    api_config: Optional[Dict[str, Any]] = None,
) -> float:
    """单个样本计算端到端奖励。
    
    注意（2026-01-01）：
    - 只计算端到端奖励（browsecomp_zh.evaluate）
    - 不计算轨迹覆盖率奖励
    
    Args:
        config: 配置字典
        prompts_str: prompt 字符串
        sequences_str: 生成的序列字符串
        ground_truth: ground truth 数据
        data_source: 数据源
        task_extra_info: 额外任务信息
        executor: 线程池执行器
        timeout: 超时时间
        
    Returns:
        端到端奖励分数（0.0 ~ 1.0）
    """
    loop = asyncio.get_running_loop()
    
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(
                    browsecomp_compute_score,
                    sequences_str,
                    ground_truth,
                    task_extra_info,
                    api_config,
                ),
            ),
            timeout=timeout,
        )
        return float(result) if result is not None else 0.0
    except asyncio.TimeoutError:
        logger.warning(f"端到端奖励计算超时")
        return 0.0
    except Exception as e:
        logger.error(f"端到端奖励计算失败: {e}")
        logger.error(traceback.format_exc())
        return 0.0


async def parallel_compute_end_to_end_score_async(
    config: Dict[str, Any],
    prompts_strs: List[str],
    sequences_strs: List[str],
    ground_truths: List[Dict[str, Any]],
    data_sources: List[str],
    task_extra_infos: Optional[List[Dict[str, Any]]] = None,
    num_processes: int = 64,
    api_config: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """批量并发计算端到端奖励。
    
    注意（2026-01-01）：
    - 只计算端到端奖励，不计算覆盖率
    
    Args:
        config: 配置字典
        prompts_strs: prompt 字符串列表
        sequences_strs: 生成的序列字符串列表
        ground_truths: ground truth 列表
        data_sources: 数据源列表
        task_extra_infos: 额外任务信息列表
        num_processes: 并发进程数
        
    Returns:
        端到端奖励分数列表
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        if task_extra_infos is None:
            task_extra_infos = [{}] * len(data_sources)
        
        # 创建所有任务
        tasks_async = [
            single_compute_end_to_end_score(
                config, prompts_str, sequences_str, ground_truth, 
                data_source, task_extra_info, executor, timeout=6000.0,
                api_config=api_config,
            )
            for prompts_str, sequences_str, ground_truth, data_source, task_extra_info in zip(
                prompts_strs, sequences_strs, ground_truths, 
                data_sources, task_extra_infos
            )
        ]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks_async, return_exceptions=False)
    
    # 处理结果
    processed_results = []
    for result in results:
        if isinstance(result, (Exception, BaseException)) or result is None:
            # 处理失败或超时的任务
            processed_results.append(0.0)
        else:
            processed_results.append(float(result))
    
    return processed_results


@register("apiprimedapoendtoendonly")
class ApiPrimeDapoEndToEndOnlyRewardManager(AbstractRewardManager):
    """
    End-to-End Only Reward Manager
    
    特性：
    - 只计算端到端奖励（browsecomp_zh.evaluate）
    - 处理格式错误和超长 rollout
    - 不计算轨迹覆盖率奖励
    
    与 Trajectory Reward V1 Naive 的差异：
    - 最终奖励公式：reward = end_to_end_score（不涉及覆盖率）
    - 不保存 trajectory_reward_info
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        max_resp_len: Optional[int] = None,
        overlong_buffer_cfg: Optional[Any] = None,
        reward_api_provider: str = "vendor_a",
        reward_api_key: Optional[str] = None,
        reward_api_model: str = "gpt-4o",
        step_api_provider: str = "vendor_a",
        step_api_key: Optional[str] = None,
        step_api_model: str = "gpt-4o",
        val_api_provider: Optional[str] = None,
        val_api_key: Optional[str] = None,
        val_api_model: Optional[str] = None,
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
        
        self.config = {
            "num_workers": 64,  # Concurrent workers
        }
        
        # Build API configuration from reward_kwargs (passed via yaml)
        # Reward API config (for end-to-end scoring via browsecomp_zh)
        self.api_config = {
            "api_provider": reward_api_provider,
            "api_key": reward_api_key if reward_api_key else None,
            "grader_model_name": reward_api_model,
        }
        
        # Step API config (for step-level state evaluation)
        self.step_api_config = {
            "api_provider": step_api_provider,
            "api_key": step_api_key if step_api_key else None,
            "step_api_model": step_api_model,
        }
        
        # Validation API config (defaults to reward API config if not specified)
        self.val_api_config = {
            "api_provider": val_api_provider or reward_api_provider,
            "api_key": val_api_key or reward_api_key or None,
            "grader_model_name": val_api_model or reward_api_model,
        }
        
        logger.info(
            f"API Config - Reward: provider={self.api_config['api_provider']}, "
            f"model={self.api_config['grader_model_name']}; "
            f"Step: provider={self.step_api_config['api_provider']}, "
            f"model={self.step_api_config['step_api_model']}; "
            f"Val: provider={self.val_api_config['api_provider']}, "
            f"model={self.val_api_config['grader_model_name']}"
        )
        
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert (
                self.max_resp_len is not None
            ), f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            assert (
                self.max_resp_len >= self.overlong_buffer_cfg.len
            ), "max_resp_len must be larger than overlong_buffer.len"
        
        logger.info("✓ ApiPrimeDapoEndToEndOnlyRewardManager 初始化完成")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """计算最终奖励张量。
        
        新逻辑（2026-01-01）：
        1. **前置异常检测**：从 sglang rollout 中读取异常标记
           - 如果样本有任何异常（工具解析错误、重复query、超长等），直接给 0 奖励
           - 不再在 reward manager 中进行超长和工具异常检测
        
        2. **正常样本奖励计算**：
           - 只有没有异常的样本才进行 reward 计算
           - 只计算端到端奖励，不计算覆盖率
           - End-to-End Only 奖励公式：
             * reward = end_to_end_score（直接使用端到端分数）
        
        Args:
            data: 数据批次
            return_dict: 是否返回字典
            
        Returns:
            奖励张量或包含奖励信息的字典
        """
        # 如果已有 rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        # 批量计算奖励
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]
        
        # ========================================
        # === 第一步：从 rollout_extra_info 中提取异常标记 ===
        # ========================================
        # 
        # rollout_extra_info 结构（来自 sglang rollout）：
        # rollout_extra_info = [
        #     {
        #         "abnormal_flags": {
        #             "excessive_tool_calls_per_turn": bool,
        #             "tool_parse_error": bool,
        #             "repeated_query": bool,
        #             "search_error": bool,
        #             "exceed_max_turns": bool,
        #             "exceed_max_tokens": bool,
        #         },
        #         "trajectory_metrics": {
        #             "search_steps": int,
        #             "subq_calls": int,
        #             ...,
        #         }
        #     },
        #     ...,
        # ]
        # 
        rollout_extra_info_list = data.non_tensor_batch.get("rollout_extra_info", [{}] * len(data))
        
        # 提取异常标记
        abnormal_flags_list = []
        trajectory_metrics_list = []
        for rollout_extra_info in rollout_extra_info_list:
            if isinstance(rollout_extra_info, dict):
                abnormal_flags_list.append(rollout_extra_info.get("abnormal_flags", {}))
                trajectory_metrics_list.append(rollout_extra_info.get("trajectory_metrics", {}))
            else:
                # 兼容旧版本数据（没有 rollout_extra_info）
                abnormal_flags_list.append({})
                trajectory_metrics_list.append({})
        
        # ========================================
        # === 第二步：只为正常样本计算奖励 ===
        # ========================================
        # 
        # 正常样本：abnormal_flags 中所有标记都为 False
        # 异常样本：直接给 0 奖励，不调用 reward 计算
        # 
        normal_indices = []
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            is_abnormal = any([
                abnormal_flags.get("excessive_tool_calls_per_turn", False),
                abnormal_flags.get("tool_parse_error", False),
                abnormal_flags.get("repeated_query", False),
                abnormal_flags.get("search_error", False),
                abnormal_flags.get("exceed_max_turns", False),
                abnormal_flags.get("exceed_max_tokens", False),
            ])
            
            if not is_abnormal:
                normal_indices.append(i)
        
        logger.info(f"批次大小: {len(data)}, 正常样本数: {len(normal_indices)}, 异常样本数: {len(data) - len(normal_indices)}")
        
        # ========================================
        # === 预先初始化所有样本的 reward_extra_info ===
        # ========================================
        # 
        # 重要修复（2026-01-01）：
        # 之前的代码先添加正常样本的 extra_info，再添加异常样本的 extra_info
        # 这导致 reward_extra_info 的顺序和 reward_tensor 的顺序不一致！
        # 
        # 正确做法：按照原始数据顺序初始化所有样本的 extra_info
        # 
        batch_size = len(data)
        end_to_end_scores_all = [0.0] * batch_size  # 所有样本的端到端分数（异常样本为0）
        final_rewards_all = [0.0] * batch_size  # 所有样本的最终奖励（异常样本为0）
        abnormal_types_all = [[] for _ in range(batch_size)]  # 所有样本的异常类型
        
        # 先标记异常样本的异常类型
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            if i not in normal_indices:
                abnormal_types_all[i] = [k for k, v in abnormal_flags.items() if v]
        
        # 如果有正常样本，才进行 reward 计算
        if len(normal_indices) > 0:
            # 提取正常样本的数据
            normal_prompts_str = [self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True) for i in normal_indices]
            normal_sequences_str = [sequences_str[i] for i in normal_indices]
            
            # 提取 ground_truth（兼容旧版和新版数据结构）
            ground_truth_list = []
            for data_item in data:
                if "reward_model" in data_item.non_tensor_batch:
                    ground_truth_list.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
                else:
                    # 兼容旧版本
                    ground_truth_list.append({})
            
            normal_ground_truth = [ground_truth_list[i] for i in normal_indices]
            normal_data_sources = [data_sources[i] for i in normal_indices]
            
            # 提取 task_extra_infos 和 question 字段
            # 
            # 注意（2026-01-01）：
            # browsecomp_zh.compute_score 需要 extra_info 中包含 'question' 字段
            # 数据来源：
            #   1. data.non_tensor_batch["extra_info"]（原始数据的 extra_info，包含 question）
            #   2. data.non_tensor_batch["question"]（直接存储的 question 字段）
            # 
            task_extra_infos = []
            
            # 提取原始数据的 extra_info（包含 question）
            original_extra_info_list = data.non_tensor_batch.get("extra_info", [{}] * len(data))
            
            # 提取 question 字段（兼容两种数据格式）
            question_list = []
            missing_question_count = 0  # 统计缺少 question 的样本数
            for i, data_item in enumerate(data):
                # 优先从 non_tensor_batch["question"] 获取
                question = data_item.non_tensor_batch.get("question", None)
                
                # 如果没有，尝试从原始 extra_info 中获取
                if question is None and isinstance(original_extra_info_list[i], dict):
                    question = original_extra_info_list[i].get("question", None)
                
                # 如果还是没有，尝试从 reward_model 中获取
                if question is None and "reward_model" in data_item.non_tensor_batch:
                    question = data_item.non_tensor_batch["reward_model"].get("question", None)
                
                if question is None:
                    missing_question_count += 1
                
                question_list.append(question)
            
            # 打印调试信息
            if missing_question_count > 0:
                logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] {missing_question_count}/{len(data)} 样本缺少 question 字段！")
                logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] data.non_tensor_batch.keys(): {list(data.non_tensor_batch.keys())}")
                if len(data) > 0:
                    logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] data[0].non_tensor_batch.keys(): {list(data[0].non_tensor_batch.keys())}")
                    if "reward_model" in data[0].non_tensor_batch:
                        logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] data[0].non_tensor_batch['reward_model'].keys(): {list(data[0].non_tensor_batch['reward_model'].keys())}")
            
            for i in normal_indices:
                # 构造 task_extra_info，确保包含 question 字段
                task_extra_info = {}
                
                # 如果原始 extra_info 中有信息，先复制过来
                if isinstance(original_extra_info_list[i], dict):
                    task_extra_info.update(original_extra_info_list[i])
                
                # 确保包含 question 字段（browsecomp_zh.compute_score 必需）
                if question_list[i] is not None:
                    task_extra_info["question"] = question_list[i]
                
                task_extra_infos.append(task_extra_info)
            
            # 计算正常样本的端到端奖励
            end_to_end_scores = self.verify_normal_samples(
                normal_prompts_str,
                normal_sequences_str,
                normal_ground_truth,
                normal_data_sources,
                task_extra_infos
            )
            
            # ========================================
            # === 第三步：计算最终奖励（End-to-End Only） ===
            # ========================================
            # 
            # End-to-End Only 版本奖励计算：
            # - reward = end_to_end_score（直接使用端到端分数）
            # 
            for idx, i in enumerate(normal_indices):
                # End-to-End Only 版本奖励计算
                final_reward = end_to_end_scores[idx]
                
                # 设置奖励张量
                reward_tensor[i, valid_response_length[i].item() - 1] = final_reward
                
                # 更新对应位置的 extra_info（保持原始顺序）
                end_to_end_scores_all[i] = end_to_end_scores[idx]
                final_rewards_all[i] = final_reward
                # abnormal_types_all[i] 已经初始化为空列表
                
                # 打印调试信息
                data_source = data_sources[i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    logger.info(
                        f"[Normal Sample {i}] "
                        f"end_to_end={end_to_end_scores[idx]:.3f}, "
                        f"final_reward={final_reward:.3f}"
                    )
        
        # ========================================
        # === 第四步：打印异常样本信息（奖励已经是0） ===
        # ========================================
        for i in range(batch_size):
            if i not in normal_indices:
                # 打印调试信息
                data_source = data_sources[i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    logger.info(
                        f"[Abnormal Sample {i}] "
                        f"abnormal_types={abnormal_types_all[i]}, "
                        f"final_reward=0.0"
                    )
        
        # ========================================
        # === 第五步：按原始顺序填充 reward_extra_info ===
        # ========================================
        # 
        # 重要：这里按照原始数据顺序（0, 1, 2, ...）填充 extra_info
        # 确保和 reward_tensor 的顺序一致
        # 
        reward_extra_info['end_to_end_score'] = end_to_end_scores_all
        reward_extra_info['final_reward'] = final_rewards_all
        reward_extra_info['abnormal_types'] = abnormal_types_all
        
        logger.info(f"[Reward Summary] 平均奖励: {reward_tensor.sum(dim=-1).mean().item():.3f}")
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor
    
    def _extract_api_config(self, data: DataProto) -> Optional[Dict[str, Any]]:
        """Extract API configuration from data or reward_kwargs.
        
        Looks for api_config in the following locations:
        1. data.non_tensor_batch['reward_model']['api_config']
        2. Returns None if not found (will use default vendor_a provider)
        """
        try:
            if len(data) > 0:
                first_item = data[0]
                if 'reward_model' in first_item.non_tensor_batch:
                    rm_config = first_item.non_tensor_batch['reward_model']
                    if isinstance(rm_config, dict) and 'api_config' in rm_config:
                        return rm_config['api_config']
        except Exception as e:
            logger.debug(f"Could not extract api_config from data: {e}")
        return None

    def verify_normal_samples(
        self, 
        prompts_strs: List[str],
        sequences_strs: List[str],
        ground_truths: List[Dict[str, Any]],
        data_sources: List[str],
        task_extra_infos: List[Dict[str, Any]]
    ) -> List[float]:
        """为正常样本计算端到端奖励。
        
        这是一个辅助方法，只处理正常样本（没有异常标记的样本）。
        
        Args:
            prompts_strs: prompt 字符串列表
            sequences_strs: 生成的序列字符串列表
            ground_truths: ground truth 列表
            data_sources: 数据源列表
            task_extra_infos: 额外任务信息列表
            
        Returns:
            端到端奖励分数列表
        """
        try:
            results = asyncio.run(
                parallel_compute_end_to_end_score_async(
                    self.config,
                    prompts_strs,
                    sequences_strs,
                    ground_truths,
                    data_sources,
                    task_extra_infos=task_extra_infos,
                    num_processes=self.config.get("num_workers", 64),
                    api_config=self.api_config,
                )
            )
        except asyncio.TimeoutError:
            logger.error("正常样本奖励计算全局超时！设置所有奖励为 0")
            results = [0.0] * len(sequences_strs)
        except Exception as e:
            logger.error(f"正常样本奖励计算异常，设置所有奖励为 0: {e}")
            logger.error(traceback.format_exc())
            results = [0.0] * len(sequences_strs)
        
        return results
