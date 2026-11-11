#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中间过程奖励 - V1 Naive 版本

该模块实现基于轨迹中间事实识别的奖励计算（Naive版本），使用组内归一化来消除问题难度影响。

核心特性：
- 并发计算端到端奖励和轨迹覆盖率奖励
- 使用组内归一化消除问题难度对梯度的影响
- 处理格式错误和超长 rollout
- 早期信号放大促进探索

作者: anonymous@example.com
版本: 1.0.0
日期: 2026-01-06
"""

import json
import logging
import re
import traceback
from typing import List, Dict, Any, Optional

import json_repair
import numpy as np
from openai import OpenAI

from verl.utils.reward_score.api_client_factory import create_api_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryFactIdentifierV1:
    """轨迹中间事实识别器 V1 版本。
    
    从 search agent 的推理轨迹中识别已经发现的中间实体/事实。
    """

    # 中间事实识别 prompt 模板
    FACT_IDENTIFICATION_PROMPT = """你是一个专业的事实识别助手。

给定一个复杂的 search agent 查询、轨迹和中间事实列表，请判断轨迹中识别出了哪些中间事实。

**原始查询**：
{query}

**中间事实列表（Ground Truth）**：
{mid_facts_ground_truth}

**轨迹内容**：
{trajectory}

**任务**：
请仔细分析轨迹内容，判断轨迹中通过搜索和推理识别出了哪些中间事实。
对于每个中间事实，判断是否在轨迹中被发现（即通过搜索结果或推理过程中提及或验证）。

**输出格式**：
请严格按照以下 JSON 格式输出（不要添加任何其他文字）：
{{
    "identified_facts": [
        "事实1描述",
        "事实2描述"
    ],
    "analysis": "简要说明识别到这些事实的理由"
}}

注意：
1. 只列出在轨迹中明确识别或验证的事实
2. 如果某个事实在轨迹中没有被提及或验证，不要包含在列表中
3. 确保输出是有效的 JSON 格式
"""

    def __init__(self, config: Dict[str, Any]):
        """初始化轨迹事实识别器。
        
        Args:
            config: 配置字典，包含 API 调用所需的参数（包含model_name等）
        """
        self.config = config
        self.model_name = config.get('fact_identifier_model', 'gpt-4o')
        
        # Support configurable API provider
        api_provider = config.get('step_api_provider', config.get('api_provider', 'vendor_a'))
        api_key = config.get('step_api_key', config.get('api_key'))
        api_base_url = config.get('step_api_base_url', config.get('api_base_url'))
        
        self.client = create_api_client(
            provider=api_provider,
            api_key=api_key,
            base_url=api_base_url,
        )
        logger.info(
            f"TrajectoryFactIdentifierV1 initialized: provider={api_provider}, "
            f"model={self.model_name}"
        )

    def identify_facts(
        self, 
        trajectory: str, 
        query: str, 
        mid_facts_ground_truth_list: List[str]
    ) -> List[str]:
        """识别轨迹中发现的中间事实。
        
        Args:
            trajectory: 推理轨迹字符串
            query: 原始查询
            mid_facts_ground_truth_list: 中间事实 ground truth 列表
            
        Returns:
            识别出的中间事实列表
        """
        try:
            # 构建识别 prompt
            mid_facts_str = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(mid_facts_ground_truth_list)])
            
            prompt = self.FACT_IDENTIFICATION_PROMPT.format(
                query=query,
                mid_facts_ground_truth=mid_facts_str,
                trajectory=trajectory
            )
            
            # 使用和browsecomp_zh相同的Vendor_A方式调用LLM
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=1000
                    )
                    content = response.choices[0].message.content.strip()
                    
                    result = self._parse_identification_result(content)
                    identified_facts = result.get('identified_facts', [])
                    logger.info(f"识别到 {len(identified_facts)} 个中间事实: {identified_facts}")
                    
                    return identified_facts
                    
                except Exception as e:
                    logger.error(f"中间事实识别API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)
                    else:
                        logger.error("所有重试均失败")
                        return []
                
        except Exception as e:
            logger.error(f"中间事实识别失败: {e}")
            logger.error(traceback.format_exc())
            return []

    def _parse_identification_result(self, result_text: str) -> Dict[str, Any]:
        """解析中间事实识别结果。
        
        Args:
            result_text: API 返回的文本
            
        Returns:
            解析后的结果字典
        """
        try:
            # 尝试直接解析 JSON
            result = json_repair.loads(result_text)
            return result
        except Exception as e:
            logger.warning(f"JSON 解析失败，尝试正则提取: {e}")
            
            # 备用方案：使用正则表达式提取
            try:
                # 查找 JSON 代码块
                json_pattern = r'```json\s*(\{.*?\})\s*```'
                match = re.search(json_pattern, result_text, re.DOTALL)
                if match:
                    result = json_repair.loads(match.group(1))
                    return result
                
                # 查找普通 JSON
                json_pattern = r'\{[^}]*"identified_facts"[^}]*\}'
                match = re.search(json_pattern, result_text, re.DOTALL)
                if match:
                    result = json_repair.loads(match.group(0))
                    return result
                
                logger.error(f"无法从结果中提取 JSON: {result_text[:200]}")
                return {'identified_facts': []}
                
            except Exception as e2:
                logger.error(f"正则提取也失败: {e2}")
                return {'identified_facts': []}


def calculate_coverage_rate(
    mid_facts_identified: List[str],
    mid_facts_ground_truth: List[str]
) -> float:
    """计算中间事实覆盖率。
    
    Args:
        mid_facts_identified: 识别出的中间事实列表
        mid_facts_ground_truth: ground truth 中间事实列表
        
    Returns:
        覆盖率（0-1 之间的浮点数）
    """
    if len(mid_facts_ground_truth) == 0:
        logger.warning("ground truth 中间事实列表为空")
        return 0.0
    
    # 计算覆盖率：识别出的事实数量 / ground truth 事实数量
    coverage_rate = len(mid_facts_identified) / len(mid_facts_ground_truth)
    
    # 确保覆盖率在 [0, 1] 范围内
    coverage_rate = max(0.0, min(1.0, coverage_rate))
    
    logger.info(
        f"覆盖率计算: {len(mid_facts_identified)}/{len(mid_facts_ground_truth)} = {coverage_rate:.3f}"
    )
    
    return coverage_rate


def normalize_coverage_within_group(
    coverage_rates: List[float], 
    uid_list: List[str]
) -> List[float]:
    """对覆盖率进行组内归一化（完全类似 GRPO 优势计算实现）。
    
    组内归一化的目的：
    1. 消除问题难度对梯度大小的影响
    2. 防止奖励塌缩，早期信号放大以促进探索
    
    归一化方法（完全类似 GRPO）：
    - 按 uid 分组（同一个 uid = 同一个 query 的不同 rollout）
    - 在每个 uid 组内，归一化覆盖率 = 原始覆盖率 / 组内最大覆盖率
    - 如果组内最大覆盖率为 0，则该组所有归一化覆盖率都为 0
    - 如果组内只有 1 个样本，归一化覆盖率 = 1.0（如果原始覆盖率 > 0）或 0.0
    
    这样可以确保：
    - 每个 uid 组内的最优 rollout 的归一化覆盖率为 1.0
    - 其他 rollout 的归一化覆盖率相对于组内最优 rollout 缩放到 [0, 1] 范围
    - 不同 uid（不同 query）之间互不影响
    
    Args:
        coverage_rates: 覆盖率列表（可能来自不同的 query）
        uid_list: uid 列表，标识每个样本所属的 query（同一个 uid = 同一个 query 的不同 rollout）
        
    Returns:
        归一化后的覆盖率列表（0-1 之间）
    """
    if len(coverage_rates) == 0:
        return []
    
    if len(coverage_rates) != len(uid_list):
        raise ValueError(f"coverage_rates 和 uid_list 长度不匹配: {len(coverage_rates)} vs {len(uid_list)}")
    
    # 按 uid 分组（类似 GRPO 的 id2score）
    from collections import defaultdict
    uid2coverage = defaultdict(list)
    uid2indices = defaultdict(list)
    
    for i, (coverage, uid) in enumerate(zip(coverage_rates, uid_list)):
        uid2coverage[uid].append(coverage)
        uid2indices[uid].append(i)
    
    # 计算每个 uid 组的最大覆盖率（类似 GRPO 的 id2mean/id2std）
    uid2max_coverage = {}
    for uid, coverage_list in uid2coverage.items():
        if len(coverage_list) == 1:
            # 只有一条轨迹时，如果覆盖率 > 0 则归一化为 1.0，否则为 0.0
            uid2max_coverage[uid] = coverage_list[0] if coverage_list[0] > 0 else 1.0  # 避免除以 0
        else:
            # 多条轨迹时，使用组内最大覆盖率进行归一化
            uid2max_coverage[uid] = max(coverage_list)
    
    # 对每个样本，使用其所属 uid 组的最大覆盖率进行归一化
    normalized_coverage_rates = [0.0] * len(coverage_rates)
    
    for i, (coverage, uid) in enumerate(zip(coverage_rates, uid_list)):
        max_coverage = uid2max_coverage[uid]
        
        if max_coverage == 0:
            # 组内最大覆盖率为 0，归一化为 0
            normalized_coverage_rates[i] = 0.0
        else:
            # 归一化覆盖率 = 原始覆盖率 / 组内最大覆盖率
            normalized_coverage_rates[i] = coverage / max_coverage
    
    # 打印每个 uid 组的归一化信息
    for uid in uid2coverage:
        coverage_list = uid2coverage[uid]
        max_coverage = uid2max_coverage[uid]
        logger.info(
            f"组内归一化 (GRPO-style) [uid={uid[:8]}...]: "
            f"n_rollouts={len(coverage_list)}, max_coverage={max_coverage:.3f}"
        )
    
    logger.info(
        f"总体归一化统计: n_samples={len(coverage_rates)}, "
        f"原始范围=[{min(coverage_rates):.3f}, {max(coverage_rates):.3f}], "
        f"归一化后范围=[{min(normalized_coverage_rates):.3f}, {max(normalized_coverage_rates):.3f}]"
    )
    
    return normalized_coverage_rates


def extract_mid_facts_from_ground_truth(ground_truth: Optional[Dict[str, Any]]) -> List[str]:
    """从 ground truth 中提取中间事实列表。
    
    Args:
        ground_truth: ground truth 数据字典
        
    Returns:
        中间事实列表
    """
    try:
        if not ground_truth:
            logger.warning("ground_truth 为空")
            return []
        
        # 尝试从 ground_truth 中提取 mid_facts 字段
        mid_facts = ground_truth.get('mid_facts', [])
        
        if isinstance(mid_facts, list):
            return mid_facts
        elif isinstance(mid_facts, str):
            # 如果是字符串，尝试按行分割
            return [fact.strip() for fact in mid_facts.split('\n') if fact.strip()]
        else:
            logger.warning(f"mid_facts 字段格式不支持: {type(mid_facts)}")
            return []
            
    except Exception as e:
        logger.error(f"提取中间事实失败: {e}")
        logger.error(traceback.format_exc())
        return []

