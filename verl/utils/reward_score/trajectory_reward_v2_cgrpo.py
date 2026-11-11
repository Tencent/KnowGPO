#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中间过程奖励 - V2 C-GRPO 版本

该模块实现基于 CaRR (Citation-aware Rubric Reward) 框架的奖励计算。

核心特性（三步评估）：
1. Step 1: Hidden Entity Identification - 识别Agent回答中的隐藏实体
2. Step 2: Citation-based Rubric Judgment - 验证约束是否有引用支持
3. Step 3: Evidence Connectivity - 检查证据链连通性（BFS从答案实体出发）

最终奖励公式（C-GRPO）：
R_i = (1 - α) · R_o^(H_i) + α · R_o^(H_i) · R̂_r^(H_i)

其中：
- R_o: Outcome Reward（是否找到正确答案，0或1）
- R_r: Rubric Reward（推理质量，|R_connect| / |R_q|）
- R̂_r: 组内归一化的 Rubric Reward

作者: anonymous@example.com
版本: 2.0.0
日期: 2026-01-01
"""

import logging
import re
import traceback
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple, Set

import json_repair
from openai import OpenAI

from verl.utils.reward_score.api_client_factory import create_api_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Prompt 模板
# ============================================================

ENTITY_IDENTIFICATION_PROMPT = """Task Description
You will receive the following inputs:
1. A complex multi-hop question.
2. A list of single-hop constraints, decomposed from the original question.
• The final answer entity is labeled as <E0>.
• Intermediate entities are labeled as <E1>, <E2>, <E3>, and so on.
3. An AI assistant's response to the multi-hop question.
Your Task
Extract the explicitly stated real identities of <E0>, <E1>, <E2>, . . . from the assistant's response. Provide an
analysis first, then return a JSON object with one key per entity label.
Output Format
## Analysis
{{Explain, for each entity label, whether the assistant's response clearly
and explicitly provides its real identity. State the actual name or value
if it is explicitly mentioned; otherwise, indicate that the identity is
not clearly stated.}}
## Final JSON-format Summary
```json
{{
"E0": {{actual identity from assistant's response or null}},
"E1": {{actual identity from assistant's response or null}},
"E2": {{actual identity from assistant's response or null}},
...
}}
```
Important Rules
• Only use information that is explicitly and unambiguously stated in the assistant's response.
• Do not infer, guess, or deduce entity identities beyond what is explicitly provided.
• If the assistant's response does not clearly identify an entity, set its value to null.
• Follow the output format exactly.
[Begin of Question]
{question}
[End of Question]
[Begin of Constraints]
{constraints}
[End of Constraints]
[Begin of Assistant's Response]
{response}
[End of Assistant's Response]"""


CITATION_RUBRIC_JUDGMENT_PROMPT = """You will receive:
1. The contents of several webpages.
2. Several single-hop factual statements: S1, S2, ..., Sn.
Your task is to determine whether each statement is fully supported by the provided webpage contents.
For each statement:
• Find the exact evidence from the webpage contents that supports or contradicts it.
• Explain clearly why the statement is or is not fully supported, citing relevant parts of the provided text.
• List the URLs of webpages where the supporting evidence was found.
• Conclude with a judgment: Fully Supported: yes or Fully Supported: no.
At the end, summarize your results in a JSON object mapping each statement label (S1, S2, ...) to a boolean value (true
for fully supported, false for not fully supported).
Output Format:
## Supportness Analysis of S1
Explanation: {{Clearly explain why S1 is or is not fully supported
according to the given webpage contents}}
Evidence URLs: {{List of URLs containing evidence used in your explanation}}
Fully Supported: {{yes/no}}
## Supportness Analysis of S2
Explanation: {{Clearly explain why S2 is or is not fully supported
according to the given webpage contents}}
Evidence URLs: {{List of URLs containing evidence used in your explanation}}
Fully Supported: {{yes/no}}
...
## Final JSON-format Summary
**CRITICAL: You MUST output a complete valid JSON object at the end. Include ALL statements (S1, S2, S3, ...) in the JSON.**
```json
{{
"S1": {{true/false}},
"S2": {{true/false}},
"S3": {{true/false}},
...
}}
```
—
[Begin of Webpage Contents]
{context}
[End of Webpage Contents]
[Begin of Statements]
{statements}
[End of Statements]"""


# ============================================================
# 常量配置
# ============================================================

# 最大引用数量限制（防止Agent通过大量引用hack奖励）
MAX_CITATION_COUNT = 20


# ============================================================
# 错误码定义（用于容错机制）
# ============================================================

class RubricRewardError:
    """Rubric Reward 计算错误码"""
    SUCCESS = "success"
    NO_ANSWER_TAG = "no_answer_tag"  # 答案中没有 <answer> 标签
    NO_CITATION = "no_citation"  # 答案中没有 [citation:x] 格式的引用
    NO_TOOL_RESPONSE = "no_tool_response"  # 轨迹中没有 tool_response
    CITATION_NOT_FOUND = "citation_not_found"  # 引用的 citation 在 tool_response 中找不到
    NO_CONSTRAINTS = "no_constraints"  # ground truth 中没有约束
    LLM_CALL_FAILED = "llm_call_failed"  # LLM 调用失败
    PARSE_ERROR = "parse_error"  # 解析错误


class AbnormalTrajectoryType:
    """异常轨迹类型定义"""
    TOOL_PARSE_ERROR = "tool_parse_error"  # 工具解析错误（给0 reward）
    EXCEED_MAX_TURNS = "exceed_max_turns"  # 超过最大轮次（给0 reward）
    EXCEED_MAX_TOKENS = "exceed_max_tokens"  # 超过最大token数（给0 reward）
    EXCESSIVE_TOOL_CALLS = "excessive_tool_calls_per_turn"  # 单轮工具调用过多（正常计算reward）
    REPEATED_QUERY = "repeated_query"  # 重复查询（正常计算reward）


# ============================================================
# 辅助函数
# ============================================================

def extract_answer_from_trajectory(trajectory: str) -> Tuple[str, bool]:
    """从轨迹中提取 <answer> 标签内的最终回答。
    
    Args:
        trajectory: Agent的完整轨迹
        
    Returns:
        (提取到的答案文本, 是否成功提取)
    """
    # 匹配 <answer>...</answer> 标签
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, trajectory, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip(), True
    
    # 如果没有 <answer> 标签，返回失败
    return "", False


def extract_citation_indexes(answer_text: str, unique: bool = False, sorted_result: bool = False) -> Tuple[List[int], bool]:
    """从最终回答中解析引用文档索引。
    
    按照论文要求，只从答案中提取 [citation:x] 格式的引用，
    并且最多提取前 MAX_CITATION_COUNT 个（默认20个）。
    
    Args:
        answer_text: 包含引用标记的回答文本
        unique: 是否去重，默认False（保持出现顺序，用于截取前20个）
        sorted_result: 是否对结果排序，默认False
        
    Returns:
        (引用索引列表, 是否成功提取到引用)
        例如 ([1, 2, 3, 6, 8, 13, 18], True)
    """
    # 匹配 [citation:数字] 的模式
    citation_pattern = r'\[citation:(\d+)\]'
    
    # 找到所有匹配项（按出现顺序）
    matches = re.findall(citation_pattern, answer_text)
    
    if not matches:
        return [], False
    
    # 转换为整数
    citation_indexes = [int(idx) for idx in matches]
    
    # 截取前 MAX_CITATION_COUNT 个（在去重之前，保持出现顺序）
    citation_indexes = citation_indexes[:MAX_CITATION_COUNT]
    
    # 根据参数处理结果
    if unique:
        # 去重但保持顺序
        seen = set()
        unique_indexes = []
        for idx in citation_indexes:
            if idx not in seen:
                seen.add(idx)
                unique_indexes.append(idx)
        citation_indexes = unique_indexes
    
    if sorted_result:
        citation_indexes = sorted(citation_indexes)
    
    return citation_indexes, True


def extract_tool_responses_from_trajectory(trajectory: str) -> Tuple[str, bool]:
    """从轨迹中提取所有 <tool_response> 标签的内容并拼接。
    
    Args:
        trajectory: Agent的完整轨迹
        
    Returns:
        (拼接后的 tool_response 内容, 是否成功提取)
    """
    # 匹配多种可能的 tool_response 格式
    patterns = [
        r'<tool_response>(.*?)</tool_response>',
        r'<\|tool_response\|>(.*?)<\|/tool_response\|>',
        r'<\|observation\|>(.*?)<\|/observation\|>',
    ]
    
    all_responses = []
    for pattern in patterns:
        matches = re.findall(pattern, trajectory, re.DOTALL)
        all_responses.extend(matches)
    
    if not all_responses:
        return "", False
    
    return '\n'.join(all_responses), True


def parse_webpage_from_web_tpl(tool_response_text: str, citation_index: int) -> Optional[Dict[str, str]]:
    """从 tool_response 文本中解析指定索引的网页内容。
    
    WEB_TPL 格式:
    [webpage {idx} begin]
    [webpage title]
    {title}
    [webpage url]
    {domain}
    [webpage date published]
    {date}
    [webpage authoritativeness]
    {auth}
    [webpage content begin]
    {passage}
    [webpage content end]
    [webpage {idx} end]
    
    注意：WEB_TPL 中的 idx 和 [citation:x] 中的 x 是直接对应的。
    
    Args:
        tool_response_text: 拼接后的 tool_response 文本
        citation_index: 引用索引（从 [citation:x] 中提取的 x，直接对应 WEB_TPL 的 idx）
        
    Returns:
        包含 idx, title, url, date, auth, content 的字典，如果未找到则返回 None
    """
    # 构建正则表达式匹配特定索引的网页
    # citation_index 和 WEB_TPL 中的 idx 是直接对应的
    webpage_pattern = rf'\[webpage {citation_index} begin\](.*?)\[webpage {citation_index} end\]'
    match = re.search(webpage_pattern, tool_response_text, re.DOTALL)
    
    if not match:
        return None
    
    webpage_content = match.group(1)
    
    # 解析各个字段
    result: Dict[str, str] = {
        'idx': str(citation_index),
        'title': '',
        'url': '',
        'date': '',
        'auth': '',
        'content': ''
    }
    
    # 提取 title
    title_pattern = r'\[webpage title\]\s*(.*?)\s*\[webpage url\]'
    title_match = re.search(title_pattern, webpage_content, re.DOTALL)
    if title_match:
        result['title'] = title_match.group(1).strip()
    
    # 提取 url
    url_pattern = r'\[webpage url\]\s*(.*?)\s*\[webpage date published\]'
    url_match = re.search(url_pattern, webpage_content, re.DOTALL)
    if url_match:
        result['url'] = url_match.group(1).strip()
    
    # 提取 date
    date_pattern = r'\[webpage date published\]\s*(.*?)\s*\[webpage authoritativeness\]'
    date_match = re.search(date_pattern, webpage_content, re.DOTALL)
    if date_match:
        result['date'] = date_match.group(1).strip()
    
    # 提取 authoritativeness
    auth_pattern = r'\[webpage authoritativeness\]\s*(.*?)\s*\[webpage content begin\]'
    auth_match = re.search(auth_pattern, webpage_content, re.DOTALL)
    if auth_match:
        result['auth'] = auth_match.group(1).strip()
    
    # 提取 content
    content_pattern = r'\[webpage content begin\]\s*(.*?)\s*\[webpage content end\]'
    content_match = re.search(content_pattern, webpage_content, re.DOTALL)
    if content_match:
        result['content'] = content_match.group(1).strip()
    
    return result


def collect_cited_webpage_contents(
    trajectory: str,
    citation_indexes: List[int]
) -> Tuple[str, int, int]:
    """根据引用索引从轨迹中收集对应的网页内容。
    
    实现论文中的 ExtractCitation + CollectContent 逻辑：
    1. 从 <answer> 中提取 citation indexes（已由 extract_citation_indexes 完成）
    2. 从 <tool_response> 中匹配 WEB_TPL 格式的网页内容
    
    Args:
        trajectory: Agent的完整轨迹
        citation_indexes: 引用索引列表（已截取前20个）
        
    Returns:
        (收集到的网页内容字符串, 成功找到的引用数量, 总引用数量)
    """
    if not citation_indexes:
        return "", 0, 0
    
    # 提取所有 tool_response 内容
    tool_response_text, has_tool_response = extract_tool_responses_from_trajectory(trajectory)
    
    if not has_tool_response:
        logger.warning("未找到 tool_response 内容")
        return "", 0, len(citation_indexes)
    
    # 收集每个引用索引对应的网页内容
    collected_webpages = []
    found_indexes = set()
    unique_citation_indexes = list(dict.fromkeys(citation_indexes))  # 去重但保持顺序
    
    for idx in unique_citation_indexes:
        webpage = parse_webpage_from_web_tpl(tool_response_text, idx)
        if webpage:
            found_indexes.add(idx)
            # 格式化为可读的网页内容
            formatted = f"""[Cited Webpage {idx}]
Title: {webpage['title']}
URL: {webpage['url']}
Date: {webpage['date']}
Authoritativeness: {webpage['auth']}
Content:
{webpage['content']}
"""
            collected_webpages.append(formatted)
    
    logger.info(
        f"从 {len(unique_citation_indexes)} 个唯一引用中成功提取 {len(found_indexes)} 个网页内容"
    )
    
    if not collected_webpages:
        logger.warning(f"未能从轨迹中找到任何引用的网页内容，citation_indexes={unique_citation_indexes}")
        return "", 0, len(unique_citation_indexes)
    
    return '\n---\n'.join(collected_webpages), len(found_indexes), len(unique_citation_indexes)


def instantiate_constraint_with_entities(
    constraint: str,
    identified_entities: Dict[str, Optional[str]]
) -> Tuple[str, bool]:
    """用识别出的实体替换约束中的实体标签。
    
    Args:
        constraint: 原始约束（如 "C1. <E1>毕业于音乐院校<E2>"）
        identified_entities: 识别出的实体映射（如 {"E1": "单依纯", "E2": "浙江音乐学院"}）
        
    Returns:
        (实例化后的约束, 是否完全识别)
    """
    # 提取约束中的所有实体标签
    entity_pattern = r'<(E\d+)>'
    entity_labels = re.findall(entity_pattern, constraint)
    
    # 检查是否所有实体都被识别
    all_identified = True
    instantiated = constraint
    
    for label in entity_labels:
        entity_value = identified_entities.get(label)
        if entity_value is None:
            all_identified = False
            # 保留原始标签
        else:
            # 替换实体标签为实际值
            instantiated = instantiated.replace(f'<{label}>', entity_value)
    
    return instantiated, all_identified


def build_bipartite_graph(
    constraints: List[str],
    supported_constraints: Set[int],
    constraint_entities_map: Dict[int, List[str]]
) -> Dict[str, Any]:
    """构建实体-约束二部图。
    
    Args:
        constraints: 约束列表
        supported_constraints: 被支持的约束索引集合
        constraint_entities_map: 约束到实体的映射 {约束索引: [实体标签列表]}
        
    Returns:
        二部图结构 {
            'entities': {实体标签: [约束索引列表]},
            'constraints': {约束索引: [实体标签列表]}
        }
    """
    graph = {
        'entities': defaultdict(list),
        'constraints': {}
    }
    
    for constraint_idx in supported_constraints:
        if constraint_idx in constraint_entities_map:
            entities = constraint_entities_map[constraint_idx]
            graph['constraints'][constraint_idx] = entities
            
            for entity in entities:
                graph['entities'][entity].append(constraint_idx)
    
    return graph


def bfs_connected_constraints(
    graph: Dict[str, Any],
    answer_entity: str = 'E0'
) -> Set[int]:
    """从答案实体开始BFS，找到所有连接的约束。
    
    Args:
        graph: 二部图结构
        answer_entity: 答案实体标签（默认为 'E0'）
        
    Returns:
        连接到答案实体的约束索引集合
    """
    visited_entities = set()
    visited_constraints = set()
    queue = deque([answer_entity])
    visited_entities.add(answer_entity)
    
    while queue:
        entity = queue.popleft()
        
        # 访问该实体连接的所有约束
        if entity in graph['entities']:
            for constraint_idx in graph['entities'][entity]:
                if constraint_idx not in visited_constraints:
                    visited_constraints.add(constraint_idx)
                    
                    # 访问该约束涉及的所有实体
                    if constraint_idx in graph['constraints']:
                        for new_entity in graph['constraints'][constraint_idx]:
                            if new_entity not in visited_entities:
                                visited_entities.add(new_entity)
                                queue.append(new_entity)
    
    return visited_constraints


def extract_entities_from_constraint(constraint: str) -> List[str]:
    """从约束中提取所有实体标签。
    
    Args:
        constraint: 约束字符串（如 "C1. <E1>毕业于音乐院校<E2>"）
        
    Returns:
        实体标签列表（如 ["E1", "E2"]）
    """
    entity_pattern = r'<(E\d+)>'
    return re.findall(entity_pattern, constraint)


# ============================================================
# 核心类
# ============================================================

class CGRPORewardCalculator:
    """C-GRPO 奖励计算器。
    
    实现基于 CaRR 框架的三步评估：
    1. Step 1: Hidden Entity Identification
    2. Step 2: Citation-based Rubric Judgment
    3. Step 3: Evidence Connectivity
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 C-GRPO 奖励计算器。
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_name = config.get('cgrpo_judge_model', 'gpt-4o')
        
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
            f"CGRPORewardCalculator initialized: provider={api_provider}, "
            f"model={self.model_name}"
        )

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """调用 LLM API。
        
        Args:
            prompt: 输入 prompt
            max_tokens: 最大输出 token 数
            
        Returns:
            LLM 输出文本
        """
        max_retries = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                logger.error(f"LLM API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
        
        # 所有重试都失败后抛出最后一个异常
        raise RuntimeError(f"LLM API 调用在 {max_retries} 次重试后仍然失败") from last_error

    def step1_entity_identification(
        self,
        question: str,
        constraints: List[str],
        response: str,
        ground_truth_entities: Dict[str, str]
    ) -> Tuple[Dict[str, Optional[str]], List[int]]:
        """Step 1: Hidden Entity Identification。
        
        识别Agent回答中的隐藏实体。
        
        Args:
            question: 原始问题
            constraints: 约束列表
            response: Agent的回答
            ground_truth_entities: Ground truth 实体映射
            
        Returns:
            (识别出的实体映射, 完全识别的约束索引列表)
        """
        try:
            # 构建 prompt
            constraints_str = '\n'.join(constraints)
            prompt = ENTITY_IDENTIFICATION_PROMPT.format(
                question=question,
                constraints=constraints_str,
                response=response
            )
            
            # 调用 LLM
            result_text = self._call_llm(prompt)
            
            # 解析结果
            identified_entities = self._parse_entity_identification_result(result_text, ground_truth_entities)
            
            # 计算完全识别的约束
            fully_identified_constraints = []
            for i, constraint in enumerate(constraints):
                entity_labels = extract_entities_from_constraint(constraint)
                if all(identified_entities.get(label) is not None for label in entity_labels):
                    fully_identified_constraints.append(i)
            
            logger.info(
                f"Step 1 完成: 识别了 {sum(1 for v in identified_entities.values() if v is not None)}/{len(ground_truth_entities)} 个实体, "
                f"完全识别 {len(fully_identified_constraints)}/{len(constraints)} 个约束"
            )
            
            return identified_entities, fully_identified_constraints
            
        except Exception as e:
            logger.error(f"Step 1 失败: {e}")
            logger.error(traceback.format_exc())
            return {}, []

    def _parse_entity_identification_result(
        self,
        result_text: str,
        ground_truth_entities: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """解析实体识别结果。
        
        Args:
            result_text: LLM 输出文本
            ground_truth_entities: Ground truth 实体映射
            
        Returns:
            识别出的实体映射
        """
        identified = {}
        
        # 初始化所有实体为 None
        for entity_label in ground_truth_entities.keys():
            identified[entity_label] = None
        
        try:
            # 尝试提取 JSON
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                parsed = json_repair.loads(match.group(1))
            else:
                # 尝试直接解析
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json_repair.loads(result_text[json_start:json_end])
                else:
                    logger.warning(f"无法从结果中提取 JSON: {result_text}")
                    return identified
            
            # 更新识别结果
            for key, value in parsed.items():
                if key in identified and value is not None and value != 'null':
                    identified[key] = str(value)
                    
        except Exception as e:
            logger.error(f"解析实体识别结果失败: {e}")
        
        return identified

    def step2_citation_rubric_judgment(
        self,
        constraints: List[str],
        fully_identified_indices: List[int],
        identified_entities: Dict[str, Optional[str]],
        trajectory: str,
        response: str
    ) -> Tuple[Set[int], Dict[int, str], str]:
        """Step 2: Citation-based Rubric Judgment。
        
        验证被识别的约束是否有引用的网页内容支持。
        
        实现论文中的逻辑：
        1. 从 <answer> 标签中提取 [citation:x] 格式的引用索引
        2. 截取前 MAX_CITATION_COUNT (20) 个引用，防止 Agent 通过大量引用 hack 奖励
        3. 从 <tool_response> 中匹配 WEB_TPL 格式的网页内容
        4. 调用 Judge LLM 判断每个约束是否被引用内容支持
        
        容错机制：
        - 如果答案中没有 [citation:x] 格式的引用，返回错误码 NO_CITATION
        - 如果引用的 citation 在 tool_response 中找不到，返回错误码 CITATION_NOT_FOUND
        
        Args:
            constraints: 约束列表
            fully_identified_indices: 完全识别的约束索引列表
            identified_entities: 识别出的实体映射
            trajectory: Agent的完整轨迹
            response: Agent的最终回答
            
        Returns:
            (被支持的约束索引集合, 实例化后的约束映射, 错误码)
        """
        if not fully_identified_indices:
            logger.info("Step 2: 没有完全识别的约束，跳过")
            return set(), {}, RubricRewardError.SUCCESS
        
        try:
            # Step 2.1: 从答案中提取引用索引
            # response 应该是从 response_ids 解码出来的 sequence_str
            answer_text, has_answer = extract_answer_from_trajectory(response)
            
            if not has_answer:
                # 调试日志：输出原始 response 的前 500 字符，方便检查是否真的没有 <answer> 标签
                response_preview = response if len(response) > 500 else response
                logger.warning(
                    f"[DEBUG] NO_ANSWER_TAG 错误 - 原始 response 预览:\n"
                    f"{response_preview}\n"
                    f"response 长度: {len(response)}"
                )
                return set(), {}, RubricRewardError.NO_ANSWER_TAG
            
            # 提取引用索引（截取前 MAX_CITATION_COUNT 个，保持出现顺序）
            citation_indexes, has_citation = extract_citation_indexes(answer_text, unique=False, sorted_result=False)
            
            if not has_citation:
                # 调试日志：输出解析出的 answer_text，方便检查是否真的没有引用
                logger.warning(
                    f"[DEBUG] NO_CITATION 错误 - 解析出的 answer_text:\n"
                    f"{answer_text}\n"
                    f"answer 长度: {len(answer_text)}"
                )
                return set(), {}, RubricRewardError.NO_CITATION
            
            logger.info(f"Step 2: 从答案中提取了 {len(citation_indexes)} 个引用索引（限制前{MAX_CITATION_COUNT}个）")
            
            # Step 2.2: 从轨迹中收集引用的网页内容
            webpage_content, found_count, total_count = collect_cited_webpage_contents(trajectory, citation_indexes)
            
            if found_count == 0:
                # 调试日志：输出引用索引和轨迹预览，方便检查为什么找不到引用
                trajectory_preview = trajectory[:1000] if len(trajectory) > 1000 else trajectory
                logger.warning(
                    f"[DEBUG] CITATION_NOT_FOUND 错误 - 所有引用都找不到:\n"
                    f"提取的引用索引: {citation_indexes[:10]}（显示前10个）\n"
                    f"trajectory 长度: {len(trajectory)}\n"
                    f"trajectory 预览:\n{trajectory_preview}\n"
                    f"{'...' if len(trajectory) > 1000 else ''}"
                )
                return set(), {}, RubricRewardError.CITATION_NOT_FOUND
            
            if found_count < total_count:
                # 部分引用找不到，仍然继续处理（只记录info级别）
                logger.info(f"Step 2: {total_count - found_count}/{total_count} 个引用在 tool_response 中找不到")
            
            # Step 2.3: 实例化约束（用识别出的实体替换实体标签）
            instantiated_constraints = {}
            statements = []
            statement_to_constraint_idx = {}
            
            for i, constraint_idx in enumerate(fully_identified_indices):
                constraint = constraints[constraint_idx]
                instantiated, _ = instantiate_constraint_with_entities(constraint, identified_entities)
                instantiated_constraints[constraint_idx] = instantiated
                
                # 移除约束编号前缀（如 "C1. "）
                clean_statement = re.sub(r'^C\d+\.\s*', '', instantiated)
                statement_label = f"S{i+1}"
                statements.append(f"{statement_label}. {clean_statement}")
                statement_to_constraint_idx[statement_label] = constraint_idx
            
            # Step 2.4: 构建 prompt 并调用 Judge LLM
            statements_str = '\n'.join(statements)
            prompt = CITATION_RUBRIC_JUDGMENT_PROMPT.format(
                context=webpage_content[:15000],  # 限制上下文长度
                statements=statements_str
            )
            
            # Step 2.5: 调用 LLM 并重试直到成功解析 JSON
            max_parse_retries = 5
            supported_constraints = set()
            
            for parse_attempt in range(max_parse_retries):
                try:
                    result_text = self._call_llm(prompt, max_tokens=4000)
                    
                    # 尝试解析结果
                    supported_constraints = self._parse_citation_judgment_result(
                        result_text, statement_to_constraint_idx
                    )
                    
                    # 检查是否所有 statement 都被解析（验证 JSON 完整性）
                    expected_statements = set(statement_to_constraint_idx.keys())
                    parsed_statements = self._get_parsed_statements_from_result(result_text)
                    missing_statements = expected_statements - parsed_statements
                    
                    if not missing_statements:
                        # 解析成功且完整
                        logger.info(f"Step 2: JSON 解析成功 (尝试 {parse_attempt+1}/{max_parse_retries})")
                        break
                    else:
                        logger.warning(
                            f"Step 2: JSON 不完整 (尝试 {parse_attempt+1}/{max_parse_retries}), "
                            f"缺失 statements: {missing_statements}"
                        )
                        if parse_attempt == max_parse_retries - 1:
                            # 最后一次尝试也失败，返回错误
                            logger.error(f"Step 2: 经过 {max_parse_retries} 次重试仍无法解析完整 JSON: {result_text}")
                            return set(), instantiated_constraints, RubricRewardError.PARSE_ERROR
                        
                except Exception as e:
                    logger.error(f"Step 2: LLM 调用或解析失败 (尝试 {parse_attempt+1}/{max_parse_retries}): {e}")
                    if parse_attempt == max_parse_retries - 1:
                        # 最后一次尝试也失败
                        return set(), instantiated_constraints, RubricRewardError.LLM_CALL_FAILED
            
            logger.info(
                f"Step 2 完成: {len(supported_constraints)}/{len(fully_identified_indices)} 个约束被引用支持, "
                f"使用了 {found_count}/{total_count} 个有效引用"
            )
            
            return supported_constraints, instantiated_constraints, RubricRewardError.SUCCESS
            
        except Exception as e:
            logger.error(f"Step 2 失败: {e}")
            logger.error(traceback.format_exc())
            return set(), {}, RubricRewardError.PARSE_ERROR
            
    def _parse_citation_judgment_result(
        self,
        result_text: str,
        statement_to_constraint_idx: Dict[str, int]
    ) -> Set[int]:
        """解析引用判断结果。
        
        Args:
            result_text: LLM 输出文本
            statement_to_constraint_idx: 语句标签到约束索引的映射
            
        Returns:
            被支持的约束索引集合
        """
        supported = set()
        
        try:
            # 尝试提取 JSON
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                parsed = json_repair.loads(match.group(1))
            else:
                # 尝试直接解析
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json_repair.loads(result_text[json_start:json_end])
                else:
                    logger.warning(f"无法从结果中提取 JSON: {result_text[:200]}")
                    return supported
            
            # 解析支持状态
            for statement_label, is_supported in parsed.items():
                if statement_label in statement_to_constraint_idx:
                    if is_supported is True or str(is_supported).lower() == 'true':
                        constraint_idx = statement_to_constraint_idx[statement_label]
                        supported.add(constraint_idx)
                        
        except Exception as e:
            logger.error(f"解析引用判断结果失败: {e}")
        
        return supported

    def _get_parsed_statements_from_result(self, result_text: str) -> Set[str]:
        """从解析结果中提取已解析的 statement 标签集合。
        
        用于检查 JSON 是否包含所有预期的 statement。
        
        Args:
            result_text: LLM 输出文本
            
        Returns:
            已解析的 statement 标签集合（如 {"S1", "S2", "S3"}）
        """
        parsed_statements = set()
        
        try:
            # 尝试提取 JSON
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                parsed = json_repair.loads(match.group(1))
            else:
                # 尝试直接解析
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json_repair.loads(result_text[json_start:json_end])
                else:
                    return parsed_statements
            
            # 提取所有 statement 标签
            parsed_statements = set(parsed.keys())
                        
        except Exception as e:
            logger.error(f"提取 parsed statements 失败: {e}")
        
        return parsed_statements

    def step3_evidence_connectivity(
        self,
        constraints: List[str],
        supported_constraints: Set[int]
    ) -> Set[int]:
        """Step 3: Evidence Connectivity Check。
        
        检查被支持的约束是否与答案实体连接。
        
        Args:
            constraints: 约束列表
            supported_constraints: 被支持的约束索引集合
            
        Returns:
            连接到答案实体的约束索引集合
        """
        if not supported_constraints:
            logger.info("Step 3: 没有被支持的约束，跳过")
            return set()
        
        try:
            # 构建约束到实体的映射
            constraint_entities_map = {}
            for constraint_idx in supported_constraints:
                constraint = constraints[constraint_idx]
                entities = extract_entities_from_constraint(constraint)
                constraint_entities_map[constraint_idx] = entities
            
            # 构建二部图
            graph = build_bipartite_graph(
                constraints, supported_constraints, constraint_entities_map
            )
            
            # BFS 找到连接的约束
            connected_constraints = bfs_connected_constraints(graph, 'E0')
            
            logger.info(
                f"Step 3 完成: {len(connected_constraints)}/{len(supported_constraints)} 个约束连接到答案实体"
            )
            
            return connected_constraints
            
        except Exception as e:
            logger.error(f"Step 3 失败: {e}")
            logger.error(traceback.format_exc())
            return set()

    def calculate_rubric_reward(
        self,
        question: str,
        constraints: List[str],
        ground_truth_entities: Dict[str, str],
        trajectory: str,
        response: str
    ) -> Dict[str, Any]:
        """计算完整的 Rubric Reward。
        
        执行三步评估并计算最终的 Rubric Reward。
        
        容错机制：
        - 如果约束列表为空，返回 0 奖励，错误码 NO_CONSTRAINTS
        - 如果 Step 2 返回错误码（如 NO_CITATION, CITATION_NOT_FOUND），Rubric Reward 直接为 0
        - 其他解析错误也会导致 Rubric Reward 为 0
        
        Args:
            question: 原始问题
            constraints: 约束列表
            ground_truth_entities: Ground truth 实体映射
            trajectory: Agent的完整轨迹（用于提取 tool_response）
            response: Agent的最终回答（从 response_ids 解码的 sequence_str）
            
        Returns:
            包含各步骤结果、最终奖励和错误码的字典
        """
        total_constraints = len(constraints)
        
        # 重要优化（2026-01-28）：
        # 当 constraints 为空时，退化为只计算 Outcome Reward
        # 此时 Rubric Reward 应该为 1.0，相当于不惩罚（等价于只看端到端奖励）
        # 
        # 理由：
        # 1. 空约束意味着无法评估推理质量，不应该惩罚 Agent
        # 2. Rubric Reward = 1.0 使得 C-GRPO 公式退化为：
        #    R = (1-α)·R_o + α·R_o·1 = (1-α+α)·R_o = R_o
        # 3. 避免除以 0 的问题（原本 rubric_reward = n_connected / n_total）
        if total_constraints == 0:
            logger.info("约束列表为空，退化为只计算 Outcome Reward，设置 Rubric Reward = 1.0")
            return {
                'rubric_reward': 1.0,  # ← 关键修改：从 0.0 改为 1.0
                'r_identify': 1.0,  # 退化模式：所有比例都是 1.0
                'r_support': 1.0,
                'r_connect': 1.0,
                'n_total': 0,
                'n_identified': 0,
                'n_supported': 0,
                'n_connected': 0,
                'identified_entities': {},
                'supported_constraints': [],
                'connected_constraints': [],
                'error_code': RubricRewardError.NO_CONSTRAINTS,
            }
        
        # 检查 entities 是否为空（2026-01-28 新增）
        # 如果 entities 为空，也应该退化为只计算 Outcome Reward
        if not ground_truth_entities or len(ground_truth_entities) == 0:
            logger.info("实体列表为空，退化为只计算 Outcome Reward，设置 Rubric Reward = 1.0")
            return {
                'rubric_reward': 1.0,  # 退化模式
                'r_identify': 1.0,
                'r_support': 1.0,
                'r_connect': 1.0,
                'n_total': total_constraints,
                'n_identified': 0,
                'n_supported': 0,
                'n_connected': 0,
                'identified_entities': {},
                'supported_constraints': [],
                'connected_constraints': [],
                'error_code': RubricRewardError.NO_CONSTRAINTS,
            }
        
        # 定义默认结果模板（用于错误情况）
        default_result = {
            'rubric_reward': 0.0,
            'r_identify': 0.0,
            'r_support': 0.0,
            'r_connect': 0.0,
            'n_total': total_constraints,
            'n_identified': 0,
            'n_supported': 0,
            'n_connected': 0,
            'identified_entities': {},
            'supported_constraints': [],
            'connected_constraints': [],
            'error_code': RubricRewardError.SUCCESS,
        }
        
        # Step 1: Entity Identification
        identified_entities, fully_identified_indices = self.step1_entity_identification(
            question, constraints, response, ground_truth_entities
        )
        
        # Step 2: Citation-based Rubric Judgment
        supported_constraints, instantiated_constraints, step2_error = self.step2_citation_rubric_judgment(
            constraints, fully_identified_indices, identified_entities, trajectory, response
        )
        
        # 检查 Step 2 错误码
        if step2_error != RubricRewardError.SUCCESS:
            # 不再单独打印WARNING，改为在最终统计中汇总
            default_result['n_identified'] = len(fully_identified_indices)
            default_result['r_identify'] = len(fully_identified_indices) / total_constraints if total_constraints > 0 else 0.0
            default_result['identified_entities'] = identified_entities
            default_result['error_code'] = step2_error
            return default_result
        
        # Step 3: Evidence Connectivity
        connected_constraints = self.step3_evidence_connectivity(
            constraints, supported_constraints
        )
        
        # 计算各步骤比例
        n_identified = len(fully_identified_indices)
        n_supported = len(supported_constraints)
        n_connected = len(connected_constraints)
        
        r_identify = n_identified / total_constraints if total_constraints > 0 else 0.0
        r_support = n_supported / n_identified if n_identified > 0 else 0.0
        r_connect = n_connected / n_supported if n_supported > 0 else 0.0
        
        # 最终 Rubric Reward = |R_connect| / |R_q|
        rubric_reward = n_connected / total_constraints if total_constraints > 0 else 0.0
        
        logger.info(
            f"Rubric Reward 计算完成: "
            f"r_identify={r_identify:.3f} ({n_identified}/{total_constraints}), "
            f"r_support={r_support:.3f} ({n_supported}/{n_identified}), "
            f"r_connect={r_connect:.3f} ({n_connected}/{n_supported}), "
            f"final={rubric_reward:.3f} ({n_connected}/{total_constraints})"
        )
        
        return {
            'rubric_reward': rubric_reward,
            'r_identify': r_identify,
            'r_support': r_support,
            'r_connect': r_connect,
            'n_total': total_constraints,
            'n_identified': n_identified,
            'n_supported': n_supported,
            'n_connected': n_connected,
            'identified_entities': identified_entities,
            'supported_constraints': list(supported_constraints),
            'connected_constraints': list(connected_constraints),
            'error_code': RubricRewardError.SUCCESS,
        }


# ============================================================
# 辅助函数（用于 Reward Manager）
# ============================================================

def extract_constraints_and_entities_from_ground_truth(
    extra_info: Dict[str, Any]
) -> Tuple[List[str], Dict[str, str]]:
    """从 extra_info 中提取约束和实体。
    
    注意：constraints 和 entities 在数据的 extra_info 字段中，而不是 ground_truth 中。
    ground_truth 只包含 target（标准答案）。
    
    Args:
        extra_info: 数据中的 extra_info 字段，包含：
            - constraints: 约束列表（如 ["C1. <E1>毕业于音乐院校<E2>", ...]）
            - entities: 实体映射（如 {"E0": "李荣浩", "E1": "单依纯", ...}）
        
    Returns:
        (约束列表, 实体映射)
    """
    constraints = extra_info.get('constraints', [])
    entities = extra_info.get('entities', {})
    
    # 确保 constraints 是列表
    if isinstance(constraints, str):
        constraints = [c.strip() for c in constraints.split('\n') if c.strip()]
    
    # 确保 entities 是字典
    if not isinstance(entities, dict):
        entities = {}
    
    return constraints, entities


def normalize_rubric_reward_within_group(
    rubric_rewards: List[float],
    uid_list: List[str]
) -> List[float]:
    """对 Rubric Reward 进行组内归一化（GRPO-style）。
    
    归一化方法：
    - 按 uid 分组（同一个 uid = 同一个 query 的不同 rollout）
    - 在每个 uid 组内，归一化 = 原始值 / 组内最大值
    
    Args:
        rubric_rewards: Rubric Reward 列表
        uid_list: uid 列表
        
    Returns:
        归一化后的 Rubric Reward 列表
    """
    if len(rubric_rewards) == 0:
        return []
    
    if len(rubric_rewards) != len(uid_list):
        raise ValueError(f"rubric_rewards 和 uid_list 长度不匹配: {len(rubric_rewards)} vs {len(uid_list)}")
    
    # 按 uid 分组
    uid2rewards = defaultdict(list)
    uid2indices = defaultdict(list)
    
    for i, (reward, uid) in enumerate(zip(rubric_rewards, uid_list)):
        uid2rewards[uid].append(reward)
        uid2indices[uid].append(i)
    
    # 计算每个 uid 组的最大值
    uid2max = {}
    for uid, rewards in uid2rewards.items():
        max_reward = max(rewards) if rewards else 0.0
        uid2max[uid] = max_reward if max_reward > 0 else 1.0  # 避免除以 0
    
    # 归一化
    normalized = [0.0] * len(rubric_rewards)
    for i, (reward, uid) in enumerate(zip(rubric_rewards, uid_list)):
        max_reward = uid2max[uid]
        normalized[i] = reward / max_reward if max_reward > 0 else 0.0
    
    logger.info(
        f"Rubric Reward 组内归一化: n_samples={len(rubric_rewards)}, "
        f"原始范围=[{min(rubric_rewards):.3f}, {max(rubric_rewards):.3f}], "
        f"归一化后范围=[{min(normalized):.3f}, {max(normalized):.3f}]"
    )
    
    return normalized


def calculate_cgrpo_mixed_reward(
    outcome_reward: float,
    rubric_reward_normalized: float,
    alpha: float = 0.3
) -> float:
    """计算 C-GRPO 混合奖励。
    
    公式: R_i = (1 - α) · R_o + α · R_o · R̂_r
    
    Args:
        outcome_reward: Outcome Reward（0 或 1）
        rubric_reward_normalized: 归一化后的 Rubric Reward
        alpha: 平衡参数
        
    Returns:
        混合奖励
    """
    return (1 - alpha) * outcome_reward + alpha * outcome_reward * rubric_reward_normalized


def classify_abnormal_trajectory(abnormal_flags: Dict[str, bool]) -> Tuple[bool, List[str]]:
    """分类异常轨迹，判断是否应该给0 reward。
    
    根据新的策略，只有以下类型的异常才给0 reward：
    - tool_parse_error: 工具解析错误
    - exceed_max_turns: 超过最大轮次
    - exceed_max_tokens: 超过最大token数
    
    其他异常（excessive_tool_calls_per_turn、repeated_query）允许正常计算reward，
    因为它们可以通过 exceed_max_tokens 的0 reward来学习，而不应人为惩罚。
    
    重要：字段名与rollout阶段完全一致！
    - 字段名定义在 AbnormalTrajectoryType 常量类中
    - rollout阶段在 sglang_rollout.py 中设置 abnormal_flags 字典
    - 两处使用完全相同的字符串常量，确保匹配
    
    Args:
        abnormal_flags: 异常标志字典，键名必须与AbnormalTrajectoryType常量完全一致
        
    Returns:
        (是否应该给0 reward, 异常类型列表)
    """
    # 这些异常类型对应的字符串值：
    # TOOL_PARSE_ERROR = "tool_parse_error"
    # EXCEED_MAX_TURNS = "exceed_max_turns"
    # EXCEED_MAX_TOKENS = "exceed_max_tokens"
    zero_reward_types = [
        AbnormalTrajectoryType.TOOL_PARSE_ERROR,
        AbnormalTrajectoryType.EXCEED_MAX_TURNS,
        AbnormalTrajectoryType.EXCEED_MAX_TOKENS,
    ]
    
    abnormal_types = []
    should_give_zero_reward = False
    
    for abnormal_type in zero_reward_types:
        if abnormal_flags.get(abnormal_type, False):
            abnormal_types.append(abnormal_type)
            should_give_zero_reward = True
    
    # 记录其他异常类型（但不给0 reward）
    # 这些异常类型对应的字符串值：
    # EXCESSIVE_TOOL_CALLS = "excessive_tool_calls_per_turn"
    # REPEATED_QUERY = "repeated_query"
    other_abnormal_types = [
        AbnormalTrajectoryType.EXCESSIVE_TOOL_CALLS,
        AbnormalTrajectoryType.REPEATED_QUERY,
    ]
    
    for abnormal_type in other_abnormal_types:
        if abnormal_flags.get(abnormal_type, False):
            abnormal_types.append(abnormal_type)
    
    return should_give_zero_reward, abnormal_types


def compute_cgrpo_metrics(
    end_to_end_rewards: List[float],
    rubric_rewards_raw: List[float],
    rubric_rewards_norm: List[float],
    r_identify_list: List[float],
    r_support_list: List[float],
    r_connect_list: List[float],
    is_normal_trajectory: List[bool],
    rubric_error_types: List[str]
) -> Dict[str, float]:
    """计算C-GRPO训练过程的监控指标。
    
    该函数计算以下指标：
    1. 正常轨迹率（normal_rate）
    2. 端到端奖励统计（end_to_end_reward）：平均值、标准差
    3. Rubric奖励统计（只在做对的轨迹中计算）：
       - rubric_raw 平均值
       - rubric_norm 平均值
       - r_i/r_q (r_identify), r_s/r_q (r_support), r_c/r_q (r_connect)
    4. Rubric错误类型统计
    
    Args:
        end_to_end_rewards: 端到端奖励列表（0或1）
        rubric_rewards_raw: 原始rubric奖励列表
        rubric_rewards_norm: 归一化rubric奖励列表
        r_identify_list: r_identify列表
        r_support_list: r_support列表
        r_connect_list: r_connect列表
        is_normal_trajectory: 是否为正常轨迹的列表
        rubric_error_types: Rubric错误类型列表
        
    Returns:
        监控指标字典
    """
    import numpy as np
    
    metrics = {}
    
    # 1. 正常轨迹率
    if len(is_normal_trajectory) > 0:
        normal_count = sum(is_normal_trajectory)
        total_count = len(is_normal_trajectory)
        metrics['cgrpo/normal_trajectory_rate'] = normal_count / total_count
        metrics['cgrpo/normal_trajectory_count'] = float(normal_count)
        metrics['cgrpo/abnormal_trajectory_count'] = float(total_count - normal_count)
    
    # 2. 端到端奖励统计（只在正常轨迹中计算）
    normal_end_to_end_rewards = [
        r for r, is_normal in zip(end_to_end_rewards, is_normal_trajectory) if is_normal
    ]
    
    if len(normal_end_to_end_rewards) > 0:
        metrics['cgrpo/end_to_end_reward_mean'] = float(np.mean(normal_end_to_end_rewards))
        metrics['cgrpo/end_to_end_reward_std'] = float(np.std(normal_end_to_end_rewards))
        metrics['cgrpo/end_to_end_success_rate'] = float(np.mean([r >= 0.999 for r in normal_end_to_end_rewards]))
    
    # 3. Rubric奖励统计（只在做对的正常轨迹中计算）
    # 做对的定义：end_to_end_reward == 1
    correct_normal_indices = [
        i for i, (r, is_normal) in enumerate(zip(end_to_end_rewards, is_normal_trajectory))
        if r >= 0.999 and is_normal
    ]
    
    if len(correct_normal_indices) > 0:
        correct_rubric_raw = [rubric_rewards_raw[i] for i in correct_normal_indices]
        correct_rubric_norm = [rubric_rewards_norm[i] for i in correct_normal_indices]
        correct_r_identify = [r_identify_list[i] for i in correct_normal_indices]
        correct_r_support = [r_support_list[i] for i in correct_normal_indices]
        correct_r_connect = [r_connect_list[i] for i in correct_normal_indices]
        
        metrics['cgrpo/rubric_raw_mean'] = float(np.mean(correct_rubric_raw))
        metrics['cgrpo/rubric_norm_mean'] = float(np.mean(correct_rubric_norm))
        metrics['cgrpo/r_identify_mean'] = float(np.mean(correct_r_identify))
        metrics['cgrpo/r_support_mean'] = float(np.mean(correct_r_support))
        metrics['cgrpo/r_connect_mean'] = float(np.mean(correct_r_connect))
        
        # 计算标准差
        metrics['cgrpo/rubric_raw_std'] = float(np.std(correct_rubric_raw))
        metrics['cgrpo/rubric_norm_std'] = float(np.std(correct_rubric_norm))
        
        # 记录做对的样本数量
        metrics['cgrpo/correct_sample_count'] = float(len(correct_normal_indices))
    
    # 4. Rubric错误类型统计
    error_type_counts = defaultdict(int)
    for error_type in rubric_error_types:
        if error_type != RubricRewardError.SUCCESS:
            error_type_counts[error_type] += 1
    
    total_errors = sum(error_type_counts.values())
    if total_errors > 0:
        for error_type, count in error_type_counts.items():
            metrics[f'cgrpo/rubric_error/{error_type}'] = float(count)
        metrics['cgrpo/rubric_error/total'] = float(total_errors)
    
    return metrics
