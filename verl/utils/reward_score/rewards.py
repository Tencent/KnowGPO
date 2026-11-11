#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习奖励评分系统

该模块提供了基于多维度评估的奖励评分功能，支持答案质量评估和轨迹质量评估，
能够对模型生成的序列进行全面的质量评估和奖励计算。

主要功能：
- ✅ 多维度答案质量评估（内容组织、意图理解、拒答、阅读体验等）
- ✅ 轨迹质量评估（逻辑性、检索词重复冗余性）
- ✅ 参考答案对比评估
- ✅ 搜索轮次奖励计算
- ✅ 并发评估和分数归一化
- ✅ 多种奖励函数策略（平均分、最小分、参考答案对比等）

核心特性：
- **多维度评估**: 支持8个答案质量维度和2个轨迹质量维度
- **灵活配置**: 可配置维度权重、评估模型、提示词等
- **并发处理**: 使用线程池并发评估多个维度，提升效率
- **多种策略**: 提供mean、min、ref_v1、ref_v2、trajectory等多种奖励计算策略
- **两阶段评估**: 支持使用DeepSeek V3.1生成答案后再评估
- **错误恢复**: 完善的错误处理和降级机制

评估维度：
    答案质量维度：
    - 内容组织: 答案结构和组织的合理性
    - 意图理解: 对用户意图的理解准确度
    - 拒答: 拒答的合理性和礼貌性
    - 阅读体验: 答案的可读性和用户体验
    - 内容质量-忠实度: 答案对参考文档的忠实程度
    - 内容质量-时效性: 答案信息的时效性
    - 内容质量-可信度: 答案内容的可信度
    - 内容质量-其他: 其他内容质量因素

    轨迹质量维度：
    - 逻辑性: 推理过程的逻辑连贯性
    - 检索词重复冗余性: 检索词的重复和冗余程度

使用示例：
    # 1. 使用平均分策略
    from rewards import my_reward_function_mean

    config = {
        'url': 'http://api.example.com',
        'model': 'your-model',
        'max_retries': 3,
        'max_assistant_turns': 5
    }

    result = my_reward_function_mean(
        config=config,
        prompt_str=prompt,
        sequences_str=sequence,
        ground_truth=None,
        data_source='test',
        messages=messages,
        task_extra_info={}
    )
    print(f"训练分数: {result['train_final_score']}")
    print(f"验证分数: {result['val_final_score']}")

    # 2. 使用参考答案对比策略
    from rewards import my_reward_function_ref_v1

    result = my_reward_function_ref_v1(
        config=config,
        prompt_str=prompt,
        sequences_str=sequence,
        ground_truth={'rm_scores': ['{"内容组织": 3, ...}']},
        data_source='test',
        messages=messages,
        task_extra_info={}
    )

    # 3. 使用轨迹评估策略
    from rewards import my_reward_function_trajectory

    result = my_reward_function_trajectory(
        config=config,
        prompt_str=prompt,
        sequences_str=sequence,
        ground_truth=None,
        data_source='test',
        messages=messages,
        task_extra_info={}
    )

架构说明：
    - RewardConfig: 配置管理类，集中管理所有配置项
    - TextExtractor: 文本提取工具类，从prompt和response中提取信息
    - EvalItemBuilder: 评估项构建器，构建评估所需的数据结构
    - ModelAPIClient: 模型API调用客户端，支持重试和错误处理
    - AnswerEvaluator: 答案评估器，评估答案质量的各个维度
    - TrajectoryEvaluator: 轨迹评估器，评估推理轨迹的质量
    - ScoreCalculator: 分数计算器，归一化和聚合各维度分数
    - RewardOrchestrator: 奖励计算协调器，协调整个评估流程

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-12-01
"""

import json
import logging
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import json_repair
import requests
from requests.exceptions import Timeout, ConnectionError, RequestException

from verl.utils.reward_score.api_generative import prime_query_openai_async
from verl.utils.reward_score.api_client_factory import get_provider_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardConfig:
    """奖励评估配置管理类。

    集中管理所有评估相关的配置项，包括评估维度、维度映射、权重和API设置等。
    """

    # 答案质量评估维度（8个维度）
    DIMENSIONS: List[str] = [
        "内容组织", "意图理解", "拒答", "阅读体验",
        "内容质量-忠实度", "内容质量-时效性", "内容质量-可信度", "内容质量-其他"
    ]

    # 轨迹质量评估维度（2个维度）
    TRAJECTORY_DIMENSIONS: List[str] = ["逻辑性", "检索词重复冗余性"]

    # 维度名称到文件名的映射（用于加载对应的prompt文件）
    DIM_MAP: Dict[str, str] = {
        "内容组织": "neirongzuzhi",
        "意图理解": "yitulijie",
        "拒答": "juda",
        "阅读体验": "yuedutiyan",
        "多轮体验": "duoluntiyan",
        "内容质量": "neirongzhiliang",
        "COT": "cot",
        "COT一致性": "cotyizhixing",
        "逻辑性": "trajectory_reasoning",
        "检索词重复冗余性": "trajectory_subq",
        "内容质量-忠实度": "neirongzhiliang_zhongshidu",
        "内容质量-时效性": "neirongzhiliang_shixiaoxing",
        "内容质量-可信度": "neirongzhiliang_kexindu",
        "内容质量-其他": "neirongzhiliang_other",
    }

    # 各维度的权重配置（用于加权平均计算最终分数）
    DIM_WEIGHT: Dict[str, float] = {
        "内容组织": 0.1,
        "意图理解": 0.1,
        "拒答": 0.1,
        "阅读体验": 0.1,
        "内容质量-忠实度": 0.2,
        "内容质量-时效性": 0.2,
        "内容质量-可信度": 0.2,
        "内容质量-其他": 0.2
    }

    # API认证Token池（用于负载均衡和容错）
    TOKENS: List[str] = []

    # 默认配置路径
    DEFAULT_PROMPT_DIR: str = "utils/reward_score/prompts"
    DEFAULT_MODEL_NAME: str = "rm_0918sp_fix_qw_8b_v1_step130"

    # 评分范围常量
    MIN_SCORE: float = 0.0
    MAX_SCORE: float = 4.0
    VALID_SCORES: List[float] = [0.0, 1.0, 2.0, 3.0, 4.0]
    NORMALIZATION_FACTOR: float = 3.0  # 归一化因子（将0-4分转换为0-1）

    # 文档长度限制（避免超过模型上下文限制）
    MAX_DOCS_LENGTH: int = 95000


class TextExtractor:
    """文本提取工具类。

    提供静态方法用于从prompt和response字符串中提取结构化信息，
    包括查询、历史对话和元数据等。
    """

    @staticmethod
    def extract_prompt_info(prompt_str: str) -> Tuple[Optional[str], List[Dict[str, str]], str]:
        """从prompt字符串中提取查询、历史对话和时间戳。

        Args:
            prompt_str: 待解析的原始prompt字符串。

        Returns:
            包含以下元素的元组：
                - 当前用户查询 (Optional[str])
                - 历史对话列表 (List[Dict[str, str]])
                - 当前时间戳 (str)
        """
        # 提取历史对话记录
        history: List[Dict[str, str]] = []
        try:
            # 匹配多轮对话模式：user\n...\nassistant\n...\nuser\n
            qa_pattern = r"user\n(.*?)\nassistant\n(.*?)\nuser\n"
            matches = re.findall(qa_pattern, prompt_str, re.DOTALL)
            if matches:
                for qa in matches:
                    query = qa[0].strip()
                    answer = qa[1].strip()
                    history.append({"query": query, "answer": answer})
        except Exception as e:
            logger.error(f"提取历史对话时出错: {e}")

        # 提取当前用户问题（支持多种格式）
        q: Optional[str] = None
        try:
            # 尝试第一种格式：Question: ...
            q_pattern = r"Question: (.*?)\nassistant"
            matches = re.findall(q_pattern, prompt_str, re.DOTALL)
            if matches:
                q = matches[0].strip()
            else:
                # 尝试第二种格式：The user's message is: ...
                q_pattern = r"The user's message is:(.*?)assistant"
                matches = re.findall(q_pattern, prompt_str, re.DOTALL)
                if matches:
                    q = matches[0].strip()
        except Exception as e:
            logger.error(f"提取query时出错: {e}")

        # 提取时间信息（如果prompt中没有，则使用当前时间）
        current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            time_pattern = r"user\nThe current date is (.*?). Your primary task is to solve the user\\'s questions"
            matches = re.findall(time_pattern, prompt_str, re.DOTALL)
            if matches:
                current_time = matches[-1].strip()
        except Exception as e:
            logger.error(f"提取当前时间时出错: {e}")

        return q, history, current_time

    @staticmethod
    def extract_response_info(response_str: str) -> Tuple[str, str, str, int, str]:
        """从模型响应中提取文档、推理过程和答案。

        Args:
            response_str: 模型生成的响应字符串。

        Returns:
            包含以下元素的元组：
                - 搜索文档 (str)
                - 思考过程 (str)
                - 最终答案 (str)
                - 搜索轮次 (int)
                - 可读轨迹 (str)
        """
        # 提取所有轮次的搜索结果（从<tool_response>标签中）
        doc_pattern = r"<tool_response>(.*?)</tool_response>"
        matches = re.findall(doc_pattern, response_str, re.DOTALL)

        if matches:
            docs = [m.strip() for m in matches]
            doc_str = "\n\n".join(docs).strip()
            num_turns = len(matches)  # 搜索轮次等于tool_response的数量
        else:
            doc_str = ""
            num_turns = 0

        # 提取最后一轮的思考过程（从<think>标签中）
        think_pattern = r"<think>(.*?)</think>"
        matches = re.findall(think_pattern, response_str, re.DOTALL)
        think = matches[-1].strip() if matches else ""

        # 提取最终答案（优先从<answer>标签提取，否则从</think>后提取）
        answer = ""
        try:
            answer_pattern = r"<answer>(.*?)</answer>"
            matches = re.findall(answer_pattern, response_str, re.DOTALL)
            if matches:
                answer = matches[-1].strip()
            else:
                # 如果没有answer标签，尝试从</think>后提取
                answer = response_str.split("</think>")[-1].strip()
                # 如果包含tool_call标签，说明还在搜索阶段，没有最终答案
                if "<tool_call>" in answer:
                    answer = ""
        except Exception as e:
            logger.error(f"提取答案时出错: {e}")
            answer = ""

        # 构建可读的推理轨迹（包含每轮的思考和搜索查询）
        readable_trajectory = ""
        trajectory_pattern = r"assistant\n<think>(.*?)</think>\d*<tool_call>(.*?)</tool_call>"
        matches = re.findall(trajectory_pattern, response_str, re.DOTALL)
        if matches:
            for m in matches:
                readable_trajectory += f"<think>{m[0]}</think>\n"
                readable_trajectory += f"<search_query>{m[1]}</search_query>\n"
        # 添加最后一轮的思考
        readable_trajectory += f"<think>{think}</think>\n"

        return doc_str, think, answer, num_turns, readable_trajectory

    @staticmethod
    def history_to_str(history: List[Dict[str, str]]) -> str:
        """将对话历史转换为格式化字符串。

        Args:
            history: 包含查询和答案的对话轮次列表。

        Returns:
            格式化的对话历史字符串。
        """
        qa_str = ""
        for i, qa in enumerate(history):
            qa_str += f"第{i+1}轮用户prompt：\n{qa['query']}\n第{i+1}轮回复：\n{qa['answer']}\n"
        return qa_str


class EvalItemBuilder:
    """评估项构建器。

    从原始的prompt和sequence字符串构建结构化的评估数据项，
    用于下游评估任务。
    """

    @staticmethod
    def build_from_sequences(sequence_str: str, prompt_str: str) -> Dict[str, Any]:
        """从sequence和prompt构建评估数据项。

        Args:
            sequence_str: 模型生成的序列字符串。
            prompt_str: 输入的prompt字符串。

        Returns:
            包含所有评估所需字段的字典，包括user_question、answer、
            reference_doc、think、num_turns等。
        """
        # 从prompt中提取信息
        query, history, current_time = TextExtractor.extract_prompt_info(prompt_str)

        # 从response中提取信息
        reference_doc, think, answer, num_turns, readable_trajectory = TextExtractor.extract_response_info(
            sequence_str
        )

        # 构建组合文档（包含历史对话和当前搜索结果）
        combined_docs = ""
        if not history:
            combined_docs += "当前为第一轮，无历史对话信息\n\n"
        else:
            history_str = TextExtractor.history_to_str(history)
            combined_docs += f"# 多轮对话历史：\n{history_str}\n\n"

        if reference_doc:
            combined_docs += f"# 当前轮次搜索结果:\n{reference_doc}"
        else:
            combined_docs += "# 没有找到相关搜索结果"

        return {
            "user_question": query,
            "answer": answer,
            "reference_doc": reference_doc,
            "combined_docs": combined_docs,
            "think": think,
            "current_time": current_time,
            "num_turns": num_turns,
            "readable_trajectory": readable_trajectory,
            "history": TextExtractor.history_to_str(history) if history else ""
        }


class BaseAPIClient:
    """API客户端基类，提供通用的HTTP请求和重试逻辑。

    封装了HTTP请求、重试机制、错误处理等通用功能，
    为特定API客户端提供基础支持。
    """

    def __init__(self, base_url: str, default_config: Dict[str, Any] = None):
        """初始化API客户端。

        Args:
            base_url: API基础URL。
            default_config: 默认配置字典。
        """
        self.base_url = base_url
        self.default_config = default_config or {}

    def _handle_response_status(self, response: requests.Response, attempt: int) -> Optional[bool]:
        """统一处理HTTP响应状态码。

        Args:
            response: HTTP响应对象。
            attempt: 当前重试次数。

        Returns:
            成功返回True，不可恢复错误返回False，需要重试返回None。
        """
        status_code = response.status_code

        if status_code == 200:
            # 请求成功
            return True
        elif status_code == 400:
            # 请求参数错误，不可恢复
            logger.error(f"Bad Request (400): {response.text[:500]}")
            return False
        elif status_code == 401:
            # 认证失败，可能是token问题，尝试重试
            logger.error("Unauthorized (401): Invalid token")
            return None
        elif status_code == 429:
            # 频率限制，根据Retry-After头部等待
            retry_after = int(response.headers.get('Retry-After', 30))
            logger.warning(f"Rate limited (429), waiting {retry_after}s")
            time.sleep(retry_after)
            return None
        elif status_code in [500, 503]:
            # 服务器错误，指数退避后重试
            logger.warning(f"Server error ({status_code})")
            time.sleep(5 * (attempt + 1))
            return None
        else:
            # 其他错误，等待后重试
            logger.error(f"HTTP error {status_code}: {response.text[:200]}")
            time.sleep(5)
            return None

    def _call_api_with_retry(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        auth_token: Optional[str] = None,
        max_retries: int = 20,
        base_timeout: int = 60,
        use_token_pool: bool = False,
        adaptive_timeout: bool = False
    ) -> Optional[Dict[str, Any]]:
        """带重试逻辑和错误处理的核心API调用函数。

        Args:
            endpoint: API端点路径。
            payload: API请求的payload字典。
            auth_token: 认证token（当use_token_pool=False时使用）。
            max_retries: 最大重试次数。
            base_timeout: 基础超时时间（秒）。
            use_token_pool: 是否从配置的token池中随机选择token。
            adaptive_timeout: 是否使用自适应超时（指数增长）。

        Returns:
            API响应的JSON数据，所有重试失败返回None。
        """
        for attempt in range(max_retries):
            # 根据配置选择认证token（支持token池负载均衡）
            if use_token_pool:
                token = random.choice(RewardConfig.TOKENS)
            else:
                token = auth_token

            # 构建请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }

            # 计算超时时间（支持自适应超时策略）
            if adaptive_timeout:
                # 每5次重试后超时时间翻倍，最多600秒
                timeout = min(base_timeout * (2 ** (attempt // 5)), 600)
            else:
                timeout = 600

            try:
                # 发送HTTP POST请求
                full_url = f"{self.base_url}/{endpoint.lstrip('/')}"
                response = requests.post(
                    full_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout
                )

                # 处理响应状态码
                status_result = self._handle_response_status(response, attempt)

                if status_result is True:
                    # 请求成功，解析JSON响应
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"Response parsing error: {e}")
                        time.sleep(3)
                        continue
                elif status_result is False:
                    # 不可恢复的错误（如400），直接返回
                    return None
                else:
                    # status_result is None，表示需要重试
                    continue

            except (Timeout, ConnectionError, RequestException) as e:
                # 网络相关异常，指数退避后重试
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                sleep_time = min(2 ** attempt, 60)  # 指数退避，最多60秒
                time.sleep(sleep_time)
                continue
            except Exception as e:
                # 其他未预期异常
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                logger.error(traceback.format_exc())
                time.sleep(5)
                continue

        # 所有重试均失败
        logger.error(f"All {max_retries} attempts failed")
        return None


class ModelAPIClient(BaseAPIClient):
    """模型API调用客户端。

    封装了与外部模型API的交互逻辑，支持重试机制、错误处理和负载均衡。
    支持多种模型类型（OpenAI格式、DeepSeek等）的统一调用接口。

    核心特性：
    - ✅ 支持多种模型格式（OpenAI、DeepSeek等）
    - ✅ 智能重试机制和错误恢复
    - ✅ 负载均衡和token池管理
    - ✅ 自适应超时策略
    - ✅ 配置验证和安全性检查
    - ✅ 性能监控和统计
    - ✅ 向后兼容的静态方法
    - ✅ 缓存机制支持
    - ✅ 详细日志记录

    使用示例：
        ```python
        # 基本用法
        client = ModelAPIClient()

        # 调用OpenAI格式模型
        messages = [
            {"role": "user", "content": "你好，请介绍一下强化学习"}
        ]
        result = client.call_openai_format(messages, model='gpt-5')

        # 调用DeepSeek V3.1（支持思考链）
        content, thinking = client.call_deepseek_v31(messages)

        # 自定义配置
        config = {
            "max_retries": 10,
            "base_timeout": 30,
            "enable_cache": True,
            "cache_ttl": 600
        }
        client = ModelAPIClient(default_config=config)

        # 获取性能统计
        metrics = client.get_metrics()
        print(f"总调用次数: {metrics['total_calls']}")
        print(f"成功率: {metrics['successful_calls'] / metrics['total_calls'] * 100:.1f}%")

        # 向后兼容的静态方法（保持现有代码可用）
        result = ModelAPIClient.call_vendor_a_api_openai(messages)
        ```

    配置选项说明：
        - max_retries: 最大重试次数（默认20）
        - base_timeout: 基础超时时间（秒，默认60）
        - use_token_pool: 是否使用token池负载均衡（默认True）
        - adaptive_timeout: 是否使用自适应超时（默认True）
        - enable_cache: 是否启用缓存（默认False）
        - cache_ttl: 缓存过期时间（秒，默认300）
        - enable_metrics: 是否启用性能监控（默认True）
    """

    # 支持的模型类型和默认配置
    SUPPORTED_MODELS = {
        'gpt-5': {'type': 'openai', 'max_tokens': 4096},
        'deepseek-v3.1-terminus': {'type': 'deepseek', 'max_tokens': 8192},
        'qwen-7b': {'type': 'openai', 'max_tokens': 2048}
    }

    def __init__(self, base_url: str = None, default_config: Dict[str, Any] = None, api_provider: str = None):
        """初始化模型API客户端。

        Args:
            base_url: API基础URL，为None时使用默认值。
            default_config: 默认配置字典。
            api_provider: API provider ('vendor_a' or 'vendor_b'). If set, overrides base_url.

        Raises:
            ValueError: 当配置参数无效时。
        """
        # Support provider-based URL resolution
        if api_provider:
            provider_config = get_provider_config(api_provider)
            base_url = base_url or (provider_config["base_url"].rstrip("/") + "/chat/completions")
        else:
            base_url = base_url or "https://api.example.com/v1/chat/completions"
        default_config = default_config or {
            "max_retries": 20,
            "base_timeout": 60,
            "use_token_pool": True,
            "adaptive_timeout": True,
            "enable_cache": False,
            "cache_ttl": 300,
            "enable_metrics": True
        }

        # 验证配置参数
        self._validate_config(default_config)

        super().__init__(base_url, default_config)

        # 初始化性能统计
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0,
            'last_call_timestamp': None
        }

        # 初始化缓存（如果启用）
        self._cache = {}
        logger.info("✓ ModelAPIClient初始化完成")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置参数的合法性。

        Args:
            config: 配置字典。

        Raises:
            ValueError: 当配置参数无效时。
        """
        if config.get("max_retries", 0) < 0:
            raise ValueError("max_retries必须大于等于0")
        if config.get("base_timeout", 0) <= 0:
            raise ValueError("base_timeout必须大于0")
        if config.get("cache_ttl", 0) < 0:
            raise ValueError("cache_ttl必须大于等于0")

    def _update_metrics(self, success: bool, response_time: float) -> None:
        """更新性能统计指标。

        Args:
            success: 调用是否成功。
            response_time: 响应时间（秒）。
        """
        if not self.default_config.get("enable_metrics", True):
            return

        self.metrics['total_calls'] += 1
        if success:
            self.metrics['successful_calls'] += 1
        else:
            self.metrics['failed_calls'] += 1

        # 计算平均响应时间（移动平均）
        if self.metrics['total_calls'] == 1:
            self.metrics['average_response_time'] = response_time
        else:
            alpha = 0.1  # 平滑因子
            self.metrics['average_response_time'] = (
                alpha * response_time +
                (1 - alpha) * self.metrics['average_response_time']
            )

        self.metrics['last_call_timestamp'] = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前性能统计指标。

        Returns:
            包含各种性能指标的字典。
        """
        return self.metrics.copy()

    def _get_cache_key(self, messages: List[Dict[str, str]], model: str) -> str:
        """生成缓存键。

        Args:
            messages: 对话消息列表。
            model: 模型名称。

        Returns:
            缓存键字符串。
        """
        import hashlib
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """检查缓存中是否存在指定键的数据。

        Args:
            cache_key: 缓存键。

        Returns:
            缓存数据，不存在返回None。
        """
        if not self.default_config.get("enable_cache", False):
            return None

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            cache_ttl = self.default_config.get("cache_ttl", 300)

            if time.time() - timestamp < cache_ttl:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_data
            else:
                # 缓存过期，删除
                del self._cache[cache_key]

        return None

    def _set_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """将数据存入缓存。

        Args:
            cache_key: 缓存键。
            data: 要缓存的数据。
        """
        if self.default_config.get("enable_cache", False):
            self._cache[cache_key] = (data, time.time())
            logger.debug(f"缓存设置: {cache_key}")

    def _call_model_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0,
        thinking_enabled: bool = False,
        max_retries: int = None,
        base_timeout: int = None,
        use_cache: bool = None
    ) -> Optional[Dict[str, Any]]:
        """通用模型API调用方法。

        Args:
            messages: 对话消息列表。
            model: 模型名称。
            temperature: 温度参数。
            thinking_enabled: 是否启用思考链功能。
            max_retries: 最大重试次数（为None时使用默认值）。
            base_timeout: 基础超时时间（为None时使用默认值）。
            use_cache: 是否使用缓存（为None时使用默认值）。

        Returns:
            API响应数据，失败返回None。
        """
        # 验证模型支持性
        if model not in self.SUPPORTED_MODELS:
            logger.warning(f"模型 {model} 不在支持列表中，可能无法正常工作")

        # 使用默认配置或传入参数
        max_retries = max_retries or self.default_config.get("max_retries", 20)
        base_timeout = base_timeout or self.default_config.get("base_timeout", 60)
        use_cache = use_cache if use_cache is not None else self.default_config.get("enable_cache", False)

        # 检查缓存（如果启用）
        cache_key = self._get_cache_key(messages, model)
        if use_cache:
            cached_response = self._check_cache(cache_key)
            if cached_response:
                self._update_metrics(True, 0.0)  # 缓存命中，响应时间为0
                return cached_response

        # 构建payload
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "stream": False,
        }

        # 如果启用思考链功能，添加相应参数
        if thinking_enabled:
            payload["thinking_enabled"] = True

        # 调用API并记录性能
        start_time = time.time()
        try:
            response_data = self._call_api_with_retry(
                endpoint="chat/completions",
                payload=payload,
                auth_token=None,
                max_retries=max_retries,
                base_timeout=base_timeout,
                use_token_pool=self.default_config.get("use_token_pool", True),
                adaptive_timeout=self.default_config.get("adaptive_timeout", True)
            )

            response_time = time.time() - start_time

            if response_data:
                # 更新性能统计
                self._update_metrics(True, response_time)

                # 缓存响应数据（如果启用）
                if use_cache:
                    self._set_cache(cache_key, response_data)

                logger.info(f"✓ API调用成功: model={model}, response_time={response_time:.2f}s")
                return response_data
            else:
                # 调用失败
                self._update_metrics(False, response_time)
                logger.error(f"✗ API调用失败: model={model}")
                return None

        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            logger.error(f"✗ API调用异常: model={model}, error={str(e)}")
            logger.error(traceback.format_exc())
            return None

    def call_openai_format(
        self,
        messages: List[Dict[str, str]],
        model: str = 'gpt-5',
        max_retries: int = None,
        base_timeout: int = None
    ) -> Optional[str]:
        """调用OpenAI格式的模型API。

        Args:
            messages: 包含role和content的对话消息列表。
            model: 模型名称。
            max_retries: 最大重试次数。
            base_timeout: 基础超时时间（秒）。

        Returns:
            模型生成的文本内容，失败返回None。
        """
        response_data = self._call_model_api(
            messages=messages,
            model=model,
            max_retries=max_retries,
            base_timeout=base_timeout
        )

        if response_data:
            try:
                return response_data['choices'][0]['message']['content']
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract content from response: {e}")
                return None
        return None

    def call_deepseek_v31(
        self,
        messages: List[Dict[str, str]],
        model: str = 'deepseek-v3.1-terminus',
        max_retries: int = None,
        base_timeout: int = None
    ) -> Tuple[str, str]:
        """调用DeepSeek V3.1模型（支持思考链功能）。

        Args:
            messages: 对话消息列表。
            model: 模型名称。
            max_retries: 最大重试次数。
            base_timeout: 基础超时时间（秒）。

        Returns:
            包含(生成内容, 思考过程)的元组，失败返回("", "")。
        """
        response_data = self._call_model_api(
            messages=messages,
            model=model,
            thinking_enabled=True,
            max_retries=max_retries,
            base_timeout=base_timeout
        )

        if response_data:
            try:
                message = response_data.get("choices", [{}])[0].get("message", {})
                thinking_content = message.get("reasoning_content", "")
                content = message.get("content", "")
                return content, thinking_content
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract content from response: {e}")
                return "", ""
        return "", ""


class AnswerEvaluator:
    """答案质量评估器。

    对答案质量进行多维度评估，包括内容组织、意图理解、阅读体验等。
    """

    def __init__(self, model_name: str, prompt_dir: str):
        """初始化答案评估器。

        Args:
            model_name: 评估模型名称。
            prompt_dir: 包含prompt模板文件的目录。
        """
        self.model_name = model_name
        self.prompt_dir = prompt_dir
        self.dimensions = RewardConfig.DIMENSIONS
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """加载所有评估维度的prompt模板。"""
        prompts = {}
        for dimension in self.dimensions:
            dim = RewardConfig.DIM_MAP.get(dimension)
            if not dim:
                logger.warning(f"✗ 维度 {dimension} 在DIM_MAP中未找到映射")
                prompts[dimension] = ""
                continue

            prompt_file = os.path.join(self.prompt_dir, f"prompt_{dim}.txt")
            if os.path.exists(prompt_file):
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompts[dimension] = f.read()
                        logger.info(f"✓ 已加载维度 {dimension} 的提示词文件")
                except IOError as e:
                    logger.error(f"✗ 读取文件失败 {prompt_file}: {e}")
                    prompts[dimension] = ""
            else:
                logger.warning(f"✗ 未找到维度 {dimension} 的提示词文件: {prompt_file}")
                prompts[dimension] = ""
        return prompts

    def construct_eval_prompt(self, item: Dict[str, Any], dimension: str) -> Optional[str]:
        """构建特定维度的评估prompt。

        Args:
            item: 包含评估所需信息的数据项。
            dimension: 评估维度名称。

        Returns:
            构建好的prompt字符串，失败返回None。
        """
        try:
            query = item.get('user_question', '')
            answer = item.get('answer', '')
            docs = item.get('combined_docs', '')
            current_time = item.get('current_time', '')

            if not query or not answer:
                logger.error(f"维度 {dimension}: query或answer为空")
                return None

            prompt_template = self.prompts.get(dimension, "")
            if not prompt_template:
                logger.error(f"维度 {dimension} 没有提示词模板")
                return None

            # 字符串截断（避免超过模型上下文限制）
            if len(docs) > RewardConfig.MAX_DOCS_LENGTH:
                docs = docs[:RewardConfig.MAX_DOCS_LENGTH]
                logger.warning(f"文档内容过长，已截断至 {RewardConfig.MAX_DOCS_LENGTH} 字符")

            prompt_content = prompt_template.format(
                docs=docs,
                query=query,
                date_str=current_time,
                answer=answer
            )

            return prompt_content

        except KeyError as e:
            logger.error(f"构建prompt时缺少必需字段 (dimension={dimension}): {e}")
            return None
        except Exception as e:
            logger.error(f"构建prompt时出错 (dimension={dimension}): {e}")
            logger.error(traceback.format_exc())
            return None

    def extract_score(self, model_output_text: str, dimension: str) -> float:
        """从模型输出文本中提取分数。

        Args:
            model_output_text: 模型输出的文本。
            dimension: 评估维度名称。

        Returns:
            提取的分数（0-4），提取失败返回0.0。
        """
        try:
            pattern_score = re.compile(r"维度整体得分：(\d)\n", re.DOTALL)
            score_match = pattern_score.search(model_output_text)
            if score_match:
                score = float(score_match.group(1))
                # 验证分数范围
                if score in RewardConfig.VALID_SCORES:
                    return score
                else:
                    logger.warning(f"维度 {dimension} 提取的分数 {score} 不在有效范围内")
                    return 0.0

            logger.warning(f"维度 {dimension} 未能从输出中提取分数")
            return 0.0
        except (ValueError, AttributeError) as e:
            logger.error(f"提取分数失败 ({dimension}): {e}")
            return 0.0
        except Exception as e:
            logger.error(f"提取分数时发生未知错误 ({dimension}): {e}")
            return 0.0

    def evaluate_dimension(self, config: Dict[str, Any], item: Dict[str, Any], dimension: str) -> float:
        """评估单个维度并返回分数。

        Args:
            config: 配置字典。
            item: 评估数据项。
            dimension: 评估维度名称。

        Returns:
            评估分数（0-4），失败返回0.0。
        """
        eval_prompt = self.construct_eval_prompt(item, dimension)
        if eval_prompt is None:
            logger.warning(f"维度 {dimension}: prompt构建失败")
            return config.get('default_score', 0.0)

        resp = prime_query_openai_async(config, eval_prompt)
        if resp is None:
            logger.warning(f"维度 {dimension}: API调用失败")
            return config.get('default_score', 0.0)

        try:
            content = resp.json()['choices'][0]['message']['content']
            score = self.extract_score(content, dimension)

            # 验证分数合法性（extract_score已经做了验证，这里是双重保险）
            if score not in RewardConfig.VALID_SCORES:
                logger.warning(f"维度 {dimension} 返回了无效分数: {score}")
                return 0.0
            return score
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"评估维度 {dimension} 时解析响应失败: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"评估维度 {dimension} 失败: {e}")
            logger.error(traceback.format_exc())
            return 0.0


class TrajectoryEvaluator:
    """轨迹质量评估器。

    评估推理轨迹的质量，包括逻辑连贯性和检索词冗余性等。
    """

    def __init__(self, model: str, prompt_dir: str):
        self.model = model
        self.prompt_dir = prompt_dir
        self.dimensions = RewardConfig.TRAJECTORY_DIMENSIONS
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """加载所有评估维度的prompt模板。"""
        prompts = {}
        for dimension in self.dimensions:
            dim = RewardConfig.DIM_MAP[dimension]
            prompt_file = os.path.join(self.prompt_dir, f"prompt_{dim}.txt")
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[dimension] = f.read()
                    logger.info(f"✓ 已加载维度 {dimension} 的提示词文件")
            else:
                logger.warning(f"✗ 未找到维度 {dimension} 的提示词文件: {prompt_file}")
                prompts[dimension] = ""
        return prompts

    def build_prompt(self, data_item: Dict[str, Any], dimension: str) -> str:
        """构建特定维度的轨迹评估prompt。

        Args:
            data_item: 包含评估所需信息的数据项。
            dimension: 评估维度名称。

        Returns:
            构建好的prompt字符串，失败返回空字符串。
        """
        try:
            query = data_item.get('user_question', '')
            history = data_item.get('history', '')
            trajectory = data_item.get('readable_trajectory', '')
            time_str = data_item.get('current_time', '')

            prompt_template = self.prompts.get(dimension, "")
            format_params = {
                'history': history,
                'query': query,
                'trajectory': trajectory,
                'time_str': time_str,
            }

            formatted_prompt = prompt_template.format(**format_params)
            return formatted_prompt.strip()
        except Exception as e:
            logger.error(f"构建prompt时出错: {e}")
            return ""

    def evaluate_dimension(self, config: Dict[str, Any], item: Dict[str, Any], dimension: str) -> float:
        """评估单个轨迹维度并返回分数。

        Args:
            config: 配置字典。
            item: 评估数据项。
            dimension: 评估维度名称。

        Returns:
            评估分数（0-4），失败返回0.0。
        """
        try:
            # 构建prompt
            formatted_prompt = self.build_prompt(item, dimension)
            if not formatted_prompt:
                logger.error(f"维度 {dimension}: prompt构建失败")
                return config.get('default_score', 0.0)

            # 构建messages
            messages = [
                {
                    'role': 'system',
                    'content': '你是全领域的专家，需要根据问题和提供的文档来判断答案是否优质。'
                },
                {
                    'role': 'user',
                    'content': formatted_prompt
                }
            ]

            # 调用API
            resp = ModelAPIClient.call_openai_format(messages)
            if resp is None:
                logger.warning(f"维度 {dimension}: API调用失败")
                return config.get('default_score', 0.0)

            try:
                content = resp.json()['choices'][0]['message']['content']
            except (KeyError, IndexError) as e:
                logger.error(f"维度 {dimension}: 解析响应失败 - {e}")
                return 0.0

            # 解析结果
            parsed_result = self._parse_result(content)
            if not parsed_result or 'result' not in parsed_result:
                logger.error(f"维度 {dimension}: 解析结果为空")
                return 0.0

            # 从result字典中获取对应维度的分数
            score = parsed_result['result'].get(dimension, 0.0)

            # 检查分数是否合法
            if score not in RewardConfig.VALID_SCORES:
                logger.warning(f"维度 {dimension}: 无效分数 {score}")
                return 0.0
            return float(score)

        except Exception as e:
            logger.error(f"评估维度 {dimension} 失败: {e}")
            logger.error(traceback.format_exc())
            return 0.0

    def _parse_result(self, result_text: str) -> Dict[str, Any]:
        """解析LLM输出的轨迹质量评分。

        Args:
            result_text: LLM输出的文本。

        Returns:
            包含result、raw_result和可能的error字段的字典。
        """
        if result_text == "safety reason error":
            parsed_result = {
                'result': {},
                'raw_result': result_text,
                'error': "safety reason error",
            }
            return parsed_result
        try:
            import re
            parsed_result = {
                'result': {},
                'raw_result': result_text
            }

            # 1. 查找最终打分字典（在【最终打分】部分）
            final_score_pattern = r'【最终打分】[：:]?\s*({[^}]*})'
            final_match = re.search(final_score_pattern, result_text)

            if final_match:
                dict_str = final_match.group(1)

                # 尝试解析字典
                try:
                    scores = json_repair.loads(dict_str)
                    parsed_result['result'] = scores

                except Exception as e:
                    logger.warning(f"字典解析失败: {e}，尝试备用方法")
                    # 备用解析方法：正则提取各维度分数
                    scores = self._extract_trajectory_scores_by_regex(dict_str)
                    parsed_result['result'] = scores
            else:
                # 如果没有找到【最终打分】部分，尝试查找任何字典格式
                dict_patterns = []
                # 为每个维度生成匹配模式
                for dimension in self.dimensions:
                    pattern = r'\{{[^}}]*["\']' + re.escape(dimension) + r'["\'][^}}]*\}}'
                    dict_patterns.append(pattern)
                # 添加通用模式
                dict_patterns.append(r'\{[^}]*:\s*\d+[^}]*\}')

                for pattern in dict_patterns:
                    dict_match = re.search(pattern, result_text)
                    if dict_match:
                        dict_str = dict_match.group(0)
                        try:
                            scores = json_repair.loads(dict_str)
                            parsed_result['result'] = scores
                            break
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.debug(f"解析字典失败: {e}")
                            continue

                # 如果仍然没有找到，使用正则提取
                if not parsed_result['result']:
                    parsed_result['result'] = self._extract_trajectory_scores_by_regex(result_text)

            return parsed_result

        except Exception as e:
            logger.error(f"解析结果时出错: {e}")
            logger.error(f"原始文本: {result_text[:500]}...")  # 只记录前500字符
            logger.error(traceback.format_exc())
            return {
                'result': {},
                'raw_result': result_text,
                'error': str(e)
            }

    def _extract_trajectory_scores_by_regex(self, text: str) -> Dict[str, float]:
        """使用正则表达式提取轨迹评分（备用方法）。

        Args:
            text: 待解析的文本。

        Returns:
            维度名到分数的映射字典。
        """
        scores = {}

        # 使用配置的评估维度
        dimensions = self.dimensions

        # 通用的分数提取模式
        score_patterns = [
            r'["\']([^"\':]+)["\']\s*:\s*(\d+)',  # "维度名": 分数
            r'([^：:]+)[：:]\s*(\d+)分',  # 维度名：X分
            r'([^：:]+)[：:]\s*评分[：:]\s*(\d+)',  # 维度名：评分：X
            r'(\d+)\s*[.、]\s*\[([^\]]+)\]\s*分数[：:]\s*(\d+)',  # 1. [逻辑连贯性] 分数：3
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    dimension = match[0].strip()
                    try:
                        score = int(match[1])
                        # 只保留轨迹质量评估的维度
                        if dimension in dimensions:
                            scores[dimension] = score
                    except ValueError:
                        continue
                elif len(match) == 3:  # 处理带序号的格式
                    dimension = match[1].strip()
                    try:
                        score = int(match[2])
                        if dimension in dimensions:
                            scores[dimension] = score
                    except ValueError:
                        continue

        # 特别处理【问题详述】部分的分数
        problem_detail_pattern = r'(\d+)\. \[([^\]]+)\]\s*分数[：:]([^\s]+)'
        problem_matches = re.findall(problem_detail_pattern, text)
        for match in problem_matches:
            if len(match) == 3:
                dimension = match[1].strip()
                score_text = match[2].strip()
                # 提取数字分数
                score_match = re.search(r'(\d+)', score_text)
                if score_match and dimension in dimensions:
                    try:
                        score = int(score_match.group(1))
                        scores[dimension] = score
                    except ValueError:
                        continue

        return scores


class ScoreCalculator:
    """分数计算器。

    提供多种分数归一化和聚合策略，将原始评分(0-4)转换为归一化分数(0-1)。
    """

    @staticmethod
    def normalize_scores(all_scores: Dict[str, float]) -> Dict[str, Any]:
        """归一化分数并计算加权平均值。

        Args:
            all_scores: 各维度的原始分数字典(0-4分)。

        Returns:
            包含归一化分数、加权平均分和最小分的字典。
        """
        # 归一化：将0-4分转换为0-1分
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }

        # 计算最小分（用于保守评估）
        rm_final_score_min = float(min(normalized.values())) if normalized else 0.0

        # 计算加权平均分（根据维度权重）
        rm_final_score = 0.0
        for dim, score in normalized.items():
            weight = RewardConfig.DIM_WEIGHT.get(dim, 0.0)
            rm_final_score += score * weight

        return {
            "all_scores": normalized,
            "rm_final_score": rm_final_score,
            "rm_final_score_min": rm_final_score_min
        }

    @staticmethod
    def normalize_scores_min(all_scores: Dict[str, float]) -> Dict[str, Any]:
        """使用最小值归一化分数（保守策略）。

        Args:
            all_scores: 各维度的原始分数字典(0-4分)。

        Returns:
            包含归一化分数和最小分的字典。
        """
        # 归一化：将0-4分转换为0-1分
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }

        # 使用最小分作为最终分数（木桶效应）
        rm_final_score_min = float(min(normalized.values())) if normalized else 0.0

        return {
            "all_scores": normalized,
            "rm_final_score": rm_final_score_min,
            "rm_final_score_min": rm_final_score_min
        }

    @staticmethod
    def normalize_scores_mean(all_scores: Dict[str, float]) -> Dict[str, Any]:
        """使用算术平均值归一化分数。

        Args:
            all_scores: 各维度的原始分数字典(0-4分)。

        Returns:
            包含归一化分数、平均分和最小分的字典。
        """
        # 归一化：将0-4分转换为0-1分
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }

        # 计算最小分
        rm_final_score_min = float(min(normalized.values())) if normalized else 0.0

        # 计算算术平均分
        rm_final_score = (
            float(sum(normalized.values()) / len(normalized))
            if normalized else 0.0
        )

        return {
            "all_scores": normalized,
            "rm_final_score": rm_final_score,
            "rm_final_score_min": rm_final_score_min
        }

    @staticmethod
    def calculate_turns_reward(
        rm_final_score: float,
        num_turns: int,
        max_turns: Optional[int]
    ) -> float:
        """计算搜索轮次奖励（鼓励用更少轮次达到完美分数）。

        Args:
            rm_final_score: RM最终分数。
            num_turns: 实际使用的搜索轮次。
            max_turns: 最大允许的搜索轮次。

        Returns:
            轮次奖励分数(0-1之间)。
        """
        # 只有在达到完美分数（1.0）且设置了最大轮次时才给予轮次奖励
        if max_turns and rm_final_score == 1.0:
            return 1.0 - num_turns / max_turns
        return 0.0


class RewardOrchestrator:
    """奖励计算协调器。

    协调整个奖励评估流程，包括答案质量评估、轨迹质量评估和分数聚合。
    使用线程池并发执行多维度评估以提升效率。
    """

    def __init__(self, prompt_dir: Optional[str] = None, model_name: Optional[str] = None):
        """初始化奖励计算协调器。

        Args:
            prompt_dir: 包含prompt文件的目录（为None时使用默认值）。
            model_name: 评估模型名称（为None时使用默认值）。
        """
        self.prompt_dir = prompt_dir or RewardConfig.DEFAULT_PROMPT_DIR
        self.model_name = model_name or RewardConfig.DEFAULT_MODEL_NAME

        # 初始化答案评估器和轨迹评估器
        self.answer_evaluator = AnswerEvaluator(self.model_name, self.prompt_dir)
        self.trajectory_evaluator = TrajectoryEvaluator("gpt-5", self.prompt_dir)

        # 加载DeepSeek V3.1的summary prompt（用于两阶段评估）
        self.dsv31_summary_prompt = self._load_prompt("prompt_dsv31_summary.txt")

    def _load_prompt(self, filename: str) -> str:
        """从文件加载prompt模板。

        Args:
            filename: Prompt模板文件名。

        Returns:
            Prompt模板内容，失败返回空字符串。
        """
        try:
            filepath = os.path.join(self.prompt_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载prompt文件失败 {filename}: {e}")
            return ""

    def get_rm_scores(self, config: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, float]:
        """并发获取所有答案质量维度的RM评分。

        使用线程池并发评估多个维度，提升评估效率。

        Args:
            config: 配置字典。
            item: 评估数据项。

        Returns:
            各维度的原始分数字典(0-4分)。

        Raises:
            ValueError: 当config或item为空，或item缺少必需字段时。
        """
        # 参数验证
        if not config:
            raise ValueError("config不能为空")
        if not item:
            raise ValueError("item不能为空")
        if 'user_question' not in item or 'answer' not in item:
            raise ValueError("item缺少必需字段: user_question或answer")

        with ThreadPoolExecutor() as executor:
            # 提交所有维度的评估任务
            future_to_dim = {
                executor.submit(
                    self.answer_evaluator.evaluate_dimension,
                    config, item, dim
                ): dim
                for dim in RewardConfig.DIMENSIONS
            }

            # 初始化分数字典
            all_scores = {dim: 0.0 for dim in RewardConfig.DIMENSIONS}

            # 收集评估结果
            for future in as_completed(future_to_dim):
                dim = future_to_dim[future]
                try:
                    score = future.result()
                    all_scores[dim] = score
                    logger.debug(f"{dim} 评估完成: score={score}")
                except Exception as e:
                    logger.error(f"{dim} 评估失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    all_scores[dim] = 0.0

        return all_scores

    def get_trajectory_scores(
        self,
        config: Dict[str, Any],
        item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """并发获取所有轨迹质量维度的评分。

        Args:
            config: 配置字典。
            item: 评估数据项。

        Returns:
            包含归一化分数和平均轨迹分数的字典。

        Raises:
            ValueError: 当config或item为空时。
        """
        # 参数验证
        if not config:
            raise ValueError("config不能为空")
        if not item:
            raise ValueError("item不能为空")

        with ThreadPoolExecutor() as executor:
            # 提交所有轨迹维度的评估任务
            future_to_dim = {
                executor.submit(
                    self.trajectory_evaluator.evaluate_dimension,
                    config, item, dim
                ): dim
                for dim in RewardConfig.TRAJECTORY_DIMENSIONS
            }

            # 初始化分数字典
            all_scores = {dim: 0.0 for dim in RewardConfig.TRAJECTORY_DIMENSIONS}

            # 收集评估结果
            for future in as_completed(future_to_dim):
                dim = future_to_dim[future]
                try:
                    score = future.result()
                    all_scores[dim] = score
                    logger.debug(f"{dim} 评估完成: score={score}")
                except Exception as e:
                    logger.error(f"{dim} 评估失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    all_scores[dim] = 0.0

        # 归一化并计算平均轨迹分数
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }
        trajectory_score = (
            float(sum(normalized.values()) / len(normalized))
            if normalized else 0.0
        )

        return {
            "all_scores": normalized,
            "trajectory_score": trajectory_score
        }

    def parse_ground_truth_scores(self, ground_truth: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """从ground truth中解析参考答案RM分数。

        Args:
            ground_truth: Ground truth数据字典。

        Returns:
            各维度的参考分数字典。
        """
        try:
            if not ground_truth:
                return {dim: 0.0 for dim in RewardConfig.DIMENSIONS}

            rm_scores = ground_truth.get("rm_scores")
            if rm_scores is None or len(rm_scores) == 0:
                return {dim: 0.0 for dim in RewardConfig.DIMENSIONS}

            return json.loads(rm_scores[0])
        except Exception as e:
            logger.error(f"解析ground_truth rm_scores失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {dim: 0.0 for dim in RewardConfig.DIMENSIONS}


def my_reward_function_mean(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """使用平均分作为奖励的函数（推荐用于训练）。

    评估策略：
    - 使用所有维度的算术平均分作为主要奖励
    - 对达到完美分数的样本给予轮次奖励
    - 返回最小分用于验证集评估

    Args:
        config: 配置字典，包含模型和API设置。
        prompt_str: 输入的prompt字符串。
        sequences_str: 模型生成的序列字符串。
        ground_truth: 参考答案数据（可选）。
        data_source: 数据来源标识。
        messages: 对话消息列表。
        task_extra_info: 额外任务信息。

    Returns:
        包含train_final_score和val_final_score的字典。
    """
    try:
        # 初始化协调器
        orchestrator = RewardOrchestrator()

        # 构建评估数据项
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str
        item["sequences_str_len"] = len(sequences_str)
        item["prompt_str_len"] = len(prompt_str)

        # 获取RM原始评分（0-4分）
        raw_scores = orchestrator.get_rm_scores(config, item)

        # 归一化并计算平均分
        model_rm_scores = ScoreCalculator.normalize_scores_mean(raw_scores)

        all_scores = model_rm_scores["all_scores"]
        rm_final_score = model_rm_scores["rm_final_score"]
        rm_final_score_min = model_rm_scores["rm_final_score_min"]

        # 计算搜索轮次奖励（鼓励用更少轮次达到高分）
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            rm_final_score, item.get("num_turns", 0), max_turns
        )

        # 训练分数 = RM平均分 + 轮次奖励
        final_score = rm_final_score + turns_reward
        # 验证分数 = RM最小分（保守评估）
        val_final_score = rm_final_score_min

        logger.info(
            f"[REWARD_MEAN] all_scores: {all_scores}, "
            f"rm_final_score: {rm_final_score}, "
            f"turns_reward: {turns_reward}, final_score: {final_score}"
        )

    except Exception as e:
        logger.error(f"奖励函数执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_min(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """使用最小分作为奖励的函数（保守策略）。

    评估策略：
    - 使用所有维度中的最小分作为主要奖励（木桶效应）
    - 确保答案在所有维度上都达到一定质量
    - 适用于对答案质量要求严格的场景

    Args:
        config: 配置字典。
        prompt_str: 输入prompt。
        sequences_str: 生成序列。
        ground_truth: 参考答案（可选）。
        data_source: 数据来源。
        messages: 对话消息。
        task_extra_info: 额外信息。

    Returns:
        包含训练和验证分数的字典。
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # 获取RM评分并使用最小值策略
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores_min(raw_scores)

        rm_final_score = model_rm_scores["rm_final_score"]
        rm_final_score_min = model_rm_scores["rm_final_score_min"]

        # 计算轮次奖励
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            rm_final_score, item.get("num_turns", 0), max_turns
        )

        final_score = rm_final_score + turns_reward
        val_final_score = rm_final_score_min

        logger.info(
            f"[REWARD_MIN] rm_final_score: {rm_final_score}, "
            f"turns_reward: {turns_reward}, "
            f"final_score: {final_score}"
        )

    except Exception as e:
        logger.error(f"奖励函数执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v1(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """参考答案奖励函数v1。

    评估策略：
    - 将模型输出与参考答案进行比较
    - 当模型分数超过参考答案时，给予额外奖励比例
    - 结合搜索轮次奖励进行综合评估
    - 适用于需要与参考答案对比的场景

    Args:
        config: 配置字典。
        prompt_str: 输入prompt。
        sequences_str: 生成序列。
        ground_truth: 参考答案数据（可选）。
        data_source: 数据来源。
        messages: 对话消息。
        task_extra_info: 额外信息。

    Returns:
        包含训练和验证分数的字典。
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # 获取模型预测答案RM评分
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        # 获取参考答案RM评分
        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        # 计算参考答案奖励比例
        if ref_rm_scores["rm_final_score"] != 0 and model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward_ratio = model_rm_scores["rm_final_score"] / ref_rm_scores["rm_final_score"]
        else:
            ref_reward_ratio = 1.0

        # 计算搜索轮次奖励
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            model_rm_scores["rm_final_score"], item.get("num_turns", 0), max_turns
        )

        # 最终分数 = 模型分数 × 参考答案比例 + 轮次奖励
        final_score = model_rm_scores["rm_final_score"] * ref_reward_ratio + turns_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V1] 模型分数: {model_rm_scores['rm_final_score']}, "
            f"参考答案分数: {ref_rm_scores['rm_final_score']}, "
            f"奖励比例: {ref_reward_ratio}, "
            f"轮次奖励: {turns_reward}, "
            f"最终分数: {final_score}"
        )

    except Exception as e:
        logger.error(f"参考答案奖励函数v1执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v1_2stage(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """参考答案奖励函数v1（两阶段版本）。

    评估策略：
    - 第一阶段：使用DeepSeek V3.1生成答案
    - 第二阶段：将生成的答案与参考答案进行比较
    - 当模型分数超过参考答案时，给予额外奖励比例
    - 结合搜索轮次奖励进行综合评估
    - 适用于需要高质量答案生成的场景

    Args:
        config: 配置字典。
        prompt_str: 输入prompt。
        sequences_str: 生成序列。
        ground_truth: 参考答案数据（可选）。
        data_source: 数据来源。
        messages: 对话消息。
        task_extra_info: 额外信息。

    Returns:
        包含训练和验证分数的字典。
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # 第一阶段：使用DeepSeek V3.1生成答案
        try:
            dsv31_prompt_formatted = orchestrator.dsv31_summary_prompt.format(
                TIME=item["current_time"],
                DOCS=item["reference_doc"],
                QUERY=item["user_question"]
            )

            ds_messages = _trans_rollout_messages_to_ds(messages)
            ds_messages = ds_messages[:-1]
            ds_messages.append({"role": "user", "content": dsv31_prompt_formatted})

            v31_answer, _ = ModelAPIClient.call_deepseek_v31(ds_messages)
            item["answer"] = v31_answer
            logger.info(f"[REWARD_REF_V1_2STAGE] DeepSeek V3.1生成答案: {v31_answer[:100]}...")
        except Exception as e:
            logger.error(f"DeepSeek V3.1调用失败: {str(e)}")

        # 第二阶段：参考答案对比评估
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        # 计算参考答案奖励比例
        if ref_rm_scores["rm_final_score"] != 0 and model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward_ratio = model_rm_scores["rm_final_score"] / ref_rm_scores["rm_final_score"]
        else:
            ref_reward_ratio = 1.0

        # 计算搜索轮次奖励
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            model_rm_scores["rm_final_score"], item.get("num_turns", 0), max_turns
        )

        # 最终分数 = 模型分数 × 参考答案比例 + 轮次奖励
        final_score = model_rm_scores["rm_final_score"] * ref_reward_ratio + turns_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V1_2STAGE] 模型分数: {model_rm_scores['rm_final_score']}, "
            f"参考答案分数: {ref_rm_scores['rm_final_score']}, "
            f"奖励比例: {ref_reward_ratio}, "
            f"轮次奖励: {turns_reward}, "
            f"最终分数: {final_score}"
        )

    except Exception as e:
        logger.error(f"参考答案奖励函数v1（两阶段）执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v2(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """参考答案奖励函数v2。

    评估策略：
    - 当模型分数超过参考答案时，给予与轮次相关的奖励
    - 当模型分数低于参考答案时，给予与轮次相关的惩罚
    - 奖励/惩罚与搜索轮次成反比，鼓励用更少轮次达到更好效果
    - 适用于需要平衡质量和效率的场景

    Args:
        config: 配置字典。
        prompt_str: 输入prompt。
        sequences_str: 生成序列。
        ground_truth: 参考答案数据（可选）。
        data_source: 数据来源。
        messages: 对话消息。
        task_extra_info: 额外信息。

    Returns:
        包含训练和验证分数的字典。
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # 获取模型预测答案RM评分
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        # 获取参考答案RM评分
        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        # v2奖励计算逻辑
        max_turns = config.get("max_assistant_turns", 10)
        num_turns = item.get("num_turns", 1)

        # 当模型分数超过参考答案时：奖励 = (模型分-参考分) / 轮次
        # 当模型分数低于参考答案时：奖励 = (模型分-参考分) + (1-轮次)/最大轮次
        if model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) / num_turns
        else:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) \
                        + (1 - num_turns) / max_turns

        final_score = ref_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V2] 模型分数: {model_rm_scores['rm_final_score']}, "
            f"参考答案分数: {ref_rm_scores['rm_final_score']}, "
            f"搜索轮次: {num_turns}, "
            f"最大轮次: {max_turns}, "
            f"最终分数: {final_score}"
        )

    except Exception as e:
        logger.error(f"参考答案奖励函数v2执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v2_2stage(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """参考答案奖励函数v2（两阶段版本）。

    评估策略：
    - 第一阶段：使用DeepSeek V3.1生成高质量答案
    - 第二阶段：使用v2策略与参考答案进行比较
    - 当模型分数超过参考答案时，给予与轮次相关的奖励
    - 当模型分数低于参考答案时，给予与轮次相关的惩罚
    - 适用于需要高质量答案生成并与参考答案对比的场景

    Args:
        config: 配置字典。
        prompt_str: 输入prompt。
        sequences_str: 生成序列。
        ground_truth: 参考答案数据（可选）。
        data_source: 数据来源。
        messages: 对话消息。
        task_extra_info: 额外信息。

    Returns:
        包含训练和验证分数的字典。
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # 第一阶段：使用DeepSeek V3.1生成答案
        try:
            dsv31_prompt_formatted = orchestrator.dsv31_summary_prompt.format(
                TIME=item["current_time"],
                DOCS=item["reference_doc"],
                QUERY=item["user_question"]
            )

            ds_messages = _trans_rollout_messages_to_ds(messages)
            ds_messages = ds_messages[:-1]
            ds_messages.append({"role": "user", "content": dsv31_prompt_formatted})

            v31_answer, _ = ModelAPIClient.call_deepseek_v31(ds_messages)
            item["answer"] = v31_answer
            logger.info(f"[REWARD_REF_V2_2STAGE] DeepSeek V3.1生成答案: {v31_answer[:100]}...")
        except Exception as e:
            logger.error(f"DeepSeek V3.1调用失败: {str(e)}")

        # 第二阶段：v2策略参考答案对比评估
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        max_turns = config.get("max_assistant_turns", 10)
        num_turns = item.get("num_turns", 1)

        # v2奖励计算逻辑
        if model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) / num_turns
        else:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) \
                        + (1 - num_turns) / max_turns

        final_score = ref_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V2_2STAGE] 模型分数: {model_rm_scores['rm_final_score']}, "
            f"参考答案分数: {ref_rm_scores['rm_final_score']}, "
            f"搜索轮次: {num_turns}, "
            f"最大轮次: {max_turns}, "
            f"最终分数: {final_score}"
        )

    except Exception as e:
        logger.error(f"参考答案奖励函数v2（两阶段）执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_trajectory(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """包含轨迹质量评估的奖励函数。

    评估策略：
    - 评估答案质量（8个维度）：内容组织、意图理解、拒答、阅读体验、忠实度、时效性、可信度、其他
    - 评估轨迹质量（2个维度）：逻辑性、检索词重复冗余性
    - 综合答案分数、轨迹分数和轮次奖励
    - 适用于需要优化推理过程和答案质量的场景

    Args:
        config: 配置字典。
        prompt_str: 输入prompt字符串。
        sequences_str: 模型生成的序列字符串。
        ground_truth: 参考答案数据（可选）。
        data_source: 数据来源标识。
        messages: 对话消息列表。
        task_extra_info: 额外任务信息。

    Returns:
        包含训练和验证分数的字典。
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # 获取答案质量RM评分（8个维度）
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        # 获取轨迹质量评分（2个维度）
        trajectory_scores = orchestrator.get_trajectory_scores(config, item)

        rm_final_score = model_rm_scores["rm_final_score"]
        rm_final_score_min = model_rm_scores["rm_final_score_min"]
        trajectory_score = trajectory_scores["trajectory_score"]

        # 计算搜索轮次奖励（鼓励用更少轮次达到高分）
        max_turns = config.get("max_assistant_turns", 1)
        turns_reward = ScoreCalculator.calculate_turns_reward(
            rm_final_score, item.get("num_turns", 0), max_turns
        )

        # 最终分数 = 答案质量分数 + 轨迹质量分数 + 轮次奖励
        final_score = rm_final_score + trajectory_score + turns_reward
        val_final_score = rm_final_score_min

        logger.info(
            f"[REWARD_TRAJECTORY] 答案质量分数: {rm_final_score}, "
            f"轨迹质量分数: {trajectory_score}, "
            f"轮次奖励: {turns_reward}, "
            f"最终分数: {final_score}"
        )

    except Exception as e:
        logger.error(f"轨迹质量奖励函数执行失败: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def _trans_rollout_messages_to_ds(messages: Dict[str, Any]) -> List[Dict[str, Any]]:
    """转换rollout消息格式为DeepSeek API格式。

    将内部消息对象转换为标准字典格式，包括处理tool_calls等特殊字段。

    Args:
        messages: 包含消息列表的字典 {"messages": [msg1, msg2, ...]}。

    Returns:
        DeepSeek API格式的消息列表。
    """
    new_messages = []

    for msg in messages.get("messages", []):
        # 构建基本消息结构
        new_msg = {
            "role": msg.role,
            "content": msg.content
        }

        # 处理工具调用（如果存在）
        if msg.tool_calls is not None:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
            if tool_calls:
                new_msg["tool_calls"] = tool_calls

        new_messages.append(new_msg)

    return new_messages
