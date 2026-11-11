#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS AI平台搜索工具实现 - 搜索工具管理、批量处理和分布式执行

该模块提供了DS AI平台的搜索工具功能，支持搜索工具管理、批量查询处理、
Ray分布式执行和速率限制控制等高级特性。

主要功能：
- ✅ 搜索工具实例管理和生命周期控制
- ✅ 批量查询并行处理
- ✅ Ray分布式执行框架集成
- ✅ 令牌桶算法速率限制
- ✅ 搜索API调用封装和错误处理
- ✅ 搜索结果格式化和元数据收集

核心特性：
- **分布式架构**: 基于Ray框架实现分布式搜索执行，支持大规模并发
- **智能速率限制**: 使用令牌桶算法实现精确的请求速率控制
- **生命周期管理**: 完整的工具实例创建、执行、释放生命周期管理
- **批量处理优化**: 支持多查询并行搜索，提高搜索效率
- **错误恢复机制**: 内置重试逻辑和异常处理，确保服务稳定性
- **性能监控**: 详细的执行指标和性能统计
- **可配置性**: 支持线程/进程模式切换，灵活适应不同场景

使用示例：
    from search_tool_vendor_search import SearchTool, init_search_execution_pool

    # 初始化搜索执行池
    execution_pool = init_search_execution_pool(
        num_workers=120,
        enable_global_rate_limit=True,
        rate_limit=120,
        mode=PoolMode.ThreadMode
    )

    # 创建搜索工具实例
    search_tool = SearchTool(
        config={
            "retrieval_service_url": "https://search.example.com/api",
            "num_workers": 120,
            "rate_limit": 120,
            "timeout": 30,
            "topk": 20
        },
        tool_schema=openai_tool_schema
    )

    # 执行搜索
    tool_response, reward, metrics = await search_tool.execute(
        instance_id="test_instance",
        parameters={"query_list": ["量化新规", "流动性风险"]},
        doc_start_idx=0
    )

    # 获取搜索结果
    result_text = tool_response.text
    print(f"搜索结果: {result_text}")
    print(f"执行指标: {metrics}")

架构说明：
- **SearchTool**: 主工具类，负责搜索工具的生命周期管理
- **TokenBucketWorker**: 令牌桶速率限制器，基于Ray Actor实现
- **SearchExecutionWorker**: 搜索执行工作器，封装API调用逻辑
- **PoolMode**: 执行池模式枚举，支持线程和进程模式

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import re

import ray
import ray.actor

from verl.tools.utils.search_utils import perform_single_search_batch
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


def extract_prompt_str(prompt_str):
    """从prompt字符串中提取用户问题。

    该函数尝试从prompt字符串中提取用户的原始问题，支持两种格式：
    1. "Question: <问题>\nassistant" 格式
    2. "The user's message is:<问题>assistant" 格式

    Args:
        prompt_str: 包含用户问题的prompt字符串

    Returns:
        str: 提取出的用户问题，如果提取失败则返回None
    """
    q = None
    try:
        # 尝试第一种格式："Question: <问题>\nassistant"
        q_pattern = r"Question: (.*?)\nassistant"
        match = re.findall(q_pattern, prompt_str, re.DOTALL)

        # 如果匹配数量不足，返回None
        if len(match) <= 1:
            q = None
        # 如果有2个或更多匹配，取第一个
        q = match[0].strip()
    except Exception:
        # 尝试第二种格式："The user's message is:<问题>assistant"
        q_pattern = r"The user's message is:(.*?)assistant"
        match = re.findall(q_pattern, prompt_str, re.DOTALL)

        # 如果没有匹配，返回None
        if len(match) < 1:
            return None
        # 取第一个匹配结果
        q = match[0].strip()

    return q


def extract_response_str(response_str):
    """从响应字符串中提取历史搜索查询。

    该函数从模型的响应字符串中提取所有历史的搜索查询（search query），
    用于避免重复搜索相同的内容。

    Args:
        response_str: 模型的完整响应字符串，包含<tool_call>标签

    Returns:
        list: 历史搜索查询列表，如果提取失败则返回空列表

    Note:
        - 从第3个tool_call开始提取（前2个通常是系统调用）
        - 排除最后一个tool_call（当前轮次的调用）
    """
    try:
        # 提取所有<tool_call>标签中的内容
        tool_pattern = r"<tool_call>(.*?)</tool_call>"
        match = re.findall(tool_pattern, response_str, re.DOTALL)
        querys = []

        if match:
            # 从第3个开始到倒数第2个（排除前2个系统调用和最后1个当前调用）
            for _, m in enumerate(match[2:-1]):
                try:
                    # 解析JSON并提取query_list
                    query_list = json.loads(m.strip())["arguments"]["query_list"]
                    querys.extend(query_list)
                except Exception:
                    # 忽略解析失败的tool_call
                    pass
        return querys
    except Exception:
        return []


def test_extract_response_str():
    response_str = '<|im_start|>system\nYou are a helpful and harmless assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "search", "description": "Search the web for information. The result will be a string.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "description": "Generate up to 5 search queries based on the requirements below:\\n\\nAll search queries should use generalized, search engine-friendly, and easily retrievable keywords to improve search efficiency and the relevance of results. Avoid directly extracting phrases from the question that are difficult to retrieve effectively. For example: Avoid using relative or ambiguous terms such as \'third author\' or \'the following year.\' - Instead, use more universal expressions like \'author list\' or \'award history.\'\\nFor a single search query, keep them short - remove unnecessary auxiliary words, conjunctions, and interrogative words. Primarily use keyword combinations (separated by spaces). Ideally, they should not exceed 15 characters. Avoid using special punctuation marks.\\nWhen seeking information from non-English primary sources, adjust search query to retrieve content in the original language (e.g., chinese), even if your initial question is in English. This improves accuracy and avoids translation-related errors.\\nYou can search 1 to 5 search queries in parallel each time. If the question involves comparing multiple entities (e.g., people, places, books, movies) or contains two or more sub-questions, please split it into separate queries for each entity or sub-question.\\nDo not re-search for queries that have already been searched.", "enum": null}}, "required": ["query_list"]}, "strict": false}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nThe current date is 2025-09-19. Your primary task is to solve the user\'s questions, leveraging the appropriate tools as needed.\n[Core Instruction: Language Consistency]\nYou MUST write your entire response in the same language as the user\'s question.\nIf the user asks in Chinese, you must think and write in Chinese.\nIf the user asks in French, you must think and write in French.\nDo not switch to English unless the user\'s question was in English. This applies to your reasoning, explanations, and the final answer.\n[Core Instruction: Citation Formatting]\nIf the search tool was used and you are referencing a specific webpage in your final answer, you must use the format "[citation:x]", where x is the corresponding webpage number.\nYou must first engage in step-by-step, rigorous thinking and reasoning before using the appropriate tool or providing the final answer.\n[Core Instruction: Search Budget Management]\nYou have a maximum of 10 search attempts total. To use this budget efficiently:\n1. If you reach the 10-search limit, provide an answer based on available results and clearly indicate the limitations of your response due to search constraints.\n2. Plan your search strategy before executing to prevent wasting search budget on one task at the expense of others.\n3. Avoid searching 3 or more similar queries to prevent search budget waste.\n4. During your thinking process, you must explicitly track and state how many searches you have already used (e.g., "Search count: 2/10") to ensure you stay within the 10-search budget limit.\n\n# The user\'s message is: of155基因标记是刘邦独有的吗？<|im_end|>\n<|im_start|>assistant\n<think>\n好的，用户问的是“of155基因标记是刘邦独有的吗？”。首先，我需要确认这个基因标记的正确性，可能用户指的是Y染色体上的某个单倍型，比如O-F155。接下来，我需要查找这个标记是否与刘邦有关联。刘邦是汉朝的开国皇帝，属于汉族，可能属于O-F155的下游分支，比如O1a1-F11。然后，我需要确定这个标记是否只有刘邦有，还是其他人群也有。根据已有的知识，O-F155主要分布在东亚，尤其是汉族和一些其他东亚民族中，所以可能不是刘邦独有的。但为了准确回答，我需要使用搜索工具来查找相关研究或数据库信息。首先生成几个搜索关键词，比如“O-F155 刘邦”、“O-F155 基因分布”、“刘邦 Y染色体标记”等。然后执行搜索，查看是否有可靠的资料支持。假设搜索结果显示O-F155在汉族中较为常见，且有多个历史人物可能携带该标记，那么可以推断它并非刘邦独有。同时，要注意引用来源，确保信息的准确性。最后，整理信息，用中文回答用户的问题，并正确标注引用来源。\n</think>\n\n<tool_call>\n{"name": "search", "arguments": {"query_list": ["O-F155 刘邦", "O-F155 基因分布", "刘邦 Y染色体标记", "汉族基因标记O-F155", "O-F155历史人物"]}}\n</tool_call><|im_end|>\n'

    print(extract_response_str(response_str))
    print(extract_prompt_str(response_str))


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """执行池模式枚举。

    定义了两种执行池模式：
    - ThreadMode: 线程模式，适合I/O密集型任务
    - ProcessMode: 进程模式，适合CPU密集型任务（暂未实现）
    """

    ThreadMode = 1  # 线程模式
    ProcessMode = 2  # 进程模式（未实现）


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """基于令牌桶算法的速率限制Ray Actor。

    使用令牌桶算法实现并发控制，防止API请求过载。
    - acquire并发组：限制为1，确保获取令牌的串行化
    - release并发组：限制为10，允许更高的释放并发度

    Attributes:
        rate_limit: 最大并发数（令牌桶容量）
        current_count: 当前已获取的令牌数（用于监控）
        _semaphore: 信号量，用于控制并发
    """

    def __init__(self, rate_limit: int):
        """初始化令牌桶。

        Args:
            rate_limit: 最大并发数限制
        """
        self.rate_limit = rate_limit
        self.current_count = 0  # 当前令牌使用数，用于可观测性
        self._semaphore = threading.Semaphore(rate_limit)  # 创建信号量

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """从令牌桶中获取一个令牌。

        该方法会阻塞，直到有可用令牌。
        使用acquire并发组确保串行执行。
        """
        self._semaphore.acquire()  # 获取信号量（可能阻塞）
        self.current_count += 1  # 增加计数器

    @ray.method(concurrency_group="release")
    def release(self):
        """将令牌释放回令牌桶。

        释放令牌后，其他等待的请求可以继续执行。
        使用release并发组允许更高的并发度。
        """
        self._semaphore.release()  # 释放信号量
        self.current_count -= 1  # 减少计数器

    def get_current_count(self):
        """获取当前已获取的令牌数量。

        Returns:
            int: 当前正在使用的令牌数
        """
        return self.current_count


class SearchExecutionWorker:
    """搜索执行工作器，支持可选的速率限制。

    该类负责执行搜索操作，并可选地应用全局速率限制。
    通过Ray的TokenBucketWorker实现并发控制。

    Attributes:
        rate_limit_worker: 速率限制器实例（如果启用）
    """

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        """初始化搜索执行工作器。

        Args:
            enable_global_rate_limit: 是否启用全局速率限制
            rate_limit: 速率限制值（最大并发数）
        """
        # 如果启用速率限制，则初始化限制器
        self.rate_limit_worker = (
            self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        )

    def _init_rate_limit(self, rate_limit):
        """初始化单例速率限制器。

        使用Ray的命名Actor机制确保全局只有一个限制器实例。

        Args:
            rate_limit: 速率限制值

        Returns:
            TokenBucketWorker: 速率限制器Actor引用
        """
        return TokenBucketWorker.options(
            name="rate-limiter", get_if_exists=True
        ).remote(rate_limit)

    def ping(self):
        """健康检查方法。

        Returns:
            bool: 始终返回True，表示工作器正常运行
        """
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """执行函数，可选地应用速率限制。

        如果启用了速率限制，会先获取令牌，执行完成后自动释放令牌。
        使用ExitStack确保即使发生异常也能正确释放令牌。

        Args:
            fn: 要执行的函数
            *fn_args: 函数的位置参数
            **fn_kwargs: 函数的关键字参数

        Returns:
            T: 函数的返回值
        """
        if self.rate_limit_worker:
            # 启用了速率限制
            with ExitStack() as stack:
                # 注册回调，确保退出时释放令牌
                stack.callback(self.rate_limit_worker.release.remote)
                # 获取令牌（可能阻塞）
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    # 执行函数
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO(developer): 应该将错误信息传递给工具调用者
                    logger.warning(f"执行搜索时发生错误: {e}")
        else:
            # 未启用速率限制，直接执行
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int,
    enable_global_rate_limit=True,
    rate_limit=10,
    mode: PoolMode = PoolMode.ThreadMode,
):
    """初始化搜索执行池。

    创建一个Ray Actor池用于执行搜索操作，支持并发控制和速率限制。

    Args:
        num_workers: 工作线程数（最大并发数）
        enable_global_rate_limit: 是否启用全局速率限制
        rate_limit: 速率限制值
        mode: 执行池模式（线程或进程）

    Returns:
        Ray Actor引用: 搜索执行工作器的Actor引用

    Raises:
        NotImplementedError: 如果使用进程模式（当前未实现）
    """
    if mode == PoolMode.ThreadMode:
        # 使用线程模式创建Ray Actor
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)  # 设置最大并发数
            .remote(
                enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit
            )
        )
    else:
        # 进程模式当前未实现
        raise NotImplementedError("进程模式尚未实现")


class SearchTool(BaseTool):
    """搜索工具，用于使用外部检索服务获取信息。

    该工具通过Ray提供带有速率限制和并发执行支持的搜索功能。
    它集成了外部检索服务来执行语义搜索操作。

    主要功能：
    - 支持多查询并行搜索
    - 自动去重和结果截断
    - 速率限制和并发控制
    - 支持历史查询跟踪，避免重复搜索

    方法：
        get_openai_tool_schema: 返回OpenAI格式的工具schema
        create: 为轨迹创建工具实例
        execute: 执行搜索工具
        calc_reward: 计算工具状态的奖励
        release: 释放工具实例
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """初始化搜索工具。

        Args:
            config: 配置字典，包含工具设置
                - num_workers: 工作线程数（默认120）
                - rate_limit: 速率限制（默认120）
                - timeout: 请求超时时间（默认30秒）
                - enable_global_rate_limit: 是否启用全局速率限制（默认True）
                - retrieval_service_url: 检索服务URL（必需）
                - topk: 返回结果数量（默认-1表示全部）
                - max_tool_response_length: 工具响应最大长度（可选）
                - tool_response_truncate_side: 截断方向（left/right/both）
            tool_schema: OpenAI函数工具schema定义

        示例tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "根据查询搜索相关信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "搜索查询列表"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}  # 存储工具实例的字典

        # 工作线程和速率限制配置
        self.num_workers = config.get("num_workers", 120)  # 工作线程数
        self.rate_limit = config.get("rate_limit", 120)  # 速率限制
        self.timeout = config.get("timeout", 30)  # 请求超时

        # 启用全局速率限制
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        # 初始化执行池
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,  # 使用线程模式
        )

        # 检索服务配置
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "配置必须包含 'retrieval_service_url'"
        self.topk = config.get("topk", -1)  # -1表示返回所有结果
        self.max_tool_response_length = config.get(
            "max_tool_response_length", None
        )  # 响应最大长度
        self.tool_response_truncate_side = config.get(
            "tool_response_truncate_side", "left"
        )  # 截断方向

        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url 未设置")

        # logger.info(f"初始化SearchTool，配置: {config}")
        print(f"初始化SearchTool，配置: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """返回OpenAI工具schema。

        Returns:
            OpenAIFunctionToolSchema: 工具的schema定义
        """
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """创建工具实例。

        为每个搜索轨迹创建一个独立的工具实例，用于跟踪搜索历史和奖励。

        Args:
            instance_id: 工具实例的ID，如果为None则自动生成
            **kwargs: 其他参数（当前未使用）

        Returns:
            tuple: (instance_id, tool_creation_response)
                - instance_id: 工具实例的ID
                - tool_creation_response: 创建实例时的响应
        """
        if instance_id is None:
            instance_id = str(uuid4())  # 生成唯一ID

        # 初始化实例数据
        self._instance_dict[instance_id] = {
            "response": "",  # 存储响应内容
            "reward": [],  # 存储奖励列表
            "docs_hashmap": {},  # 轨迹级别的文档去重哈希表，防止重复搜索相同文档
            "docs_start_index": 0,  # 轨迹级别的文档索引起始值，用于递增标注
        }
        return instance_id, ToolResponse()
    def execute_search(
        self,
        instance_id: str,
        query_list: list,
        retrieval_service_url: str,
        timeout: int,
        docs_hashmap: dict,
        docs_start_index: int,
        query: str,
    ):
        """使用检索服务执行搜索操作。

        该方法调用外部检索服务执行实际的搜索操作，支持多查询并行搜索。
        同一个轨迹内共享docs_hashmap和docs_start_index，实现文档去重和索引递增。

        Args:
            instance_id: 工具实例ID
            query_list: 搜索查询列表
            retrieval_service_url: 检索服务的URL
            timeout: 请求超时时间（秒）
            docs_hashmap: 轨迹级别的文档去重哈希表（会被修改）
            docs_start_index: 当前轨迹的文档起始索引
            query: 原始用户问题

        Returns:
            tuple: (result_text, metadata, updated_docs_hashmap, updated_docs_start_index)
                - result_text: 搜索结果JSON字符串
                - metadata: 元数据字典，包含搜索统计信息
                - updated_docs_hashmap: 更新后的文档去重哈希表
                - updated_docs_start_index: 更新后的文档起始索引
        """
        # 调用搜索工具函数执行批量搜索，传递轨迹级别的状态
        result_text, metadata, updated_docs_hashmap, updated_docs_start_index = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            concurrent_semaphore=None,  # Ray已经处理并发控制
            timeout=timeout,
            docs_hashmap=docs_hashmap,  # 传递轨迹级别的去重哈希表
            docs_start_index=docs_start_index,  # 传递当前索引
            query=query,
        )
        # logger.debug(f"实例 {instance_id} 的搜索结果: {result_text}")
        # print(f"实例 {instance_id} 的搜索结果: {result_text}")
        return result_text, metadata, updated_docs_hashmap, updated_docs_start_index


    @rollout_trace_op  
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行搜索工具。

        该方法是搜索工具的主入口，负责：
        1. 解析和验证输入参数
        2. 提取用户问题
        3. 获取轨迹级别的状态（docs_hashmap和docs_start_index）
        4. 调用检索服务执行搜索
        5. 更新轨迹状态
        6. 处理搜索结果和错误
        7. 返回格式化的响应和指标

        Args:
            instance_id: 工具实例ID
            parameters: 工具参数，包含query_list和可选的timeout
                - query_list: 搜索查询列表（必需）
            **kwargs: 其他参数
                - prompt_str: 完整的输入字符串，用于提取用户问题

        Returns:
            tuple: (tool_response, tool_reward_score, tool_metrics)
                - tool_response: 工具的响应字符串
                - tool_reward_score: 工具的步骤奖励分数（当前为0.0）
                - tool_metrics: 工具的指标字典
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")

        # 从输入中提取用户问题
        prompt_str = kwargs.get("prompt_str", "")
        query = extract_prompt_str(prompt_str)  # 提取原始问题
    
        # 验证query_list参数
        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "错误：参数中缺少'query_list'、为空或不是列表类型。"
            # logger.error(f"[SearchTool] {error_msg} 收到的参数: {parameters}")
            return (
                ToolResponse(
                    text=json.dumps({"result": error_msg}, ensure_ascii=False)
                ),
                0.0,
                {},
            )

        # 获取当前轨迹的状态
        instance_data = self._instance_dict[instance_id]
        docs_hashmap = instance_data["docs_hashmap"]
        docs_start_index = instance_data["docs_start_index"]

        # 使用Ray执行池执行搜索
        try:
            # 异步调用搜索执行方法，传递轨迹级别的状态
            result_text, metadata, updated_docs_hashmap, updated_docs_start_index = await self.execution_pool.execute.remote(
                self.execute_search,
                instance_id,
                query_list_from_params,
                self.retrieval_service_url,
                timeout,
                docs_hashmap,  # 传递轨迹级别的去重哈希表
                docs_start_index,  # 传递当前文档索引
                query,
            )

            # 更新轨迹状态：文档索引递增和去重哈希表更新
            self._instance_dict[instance_id]["docs_start_index"] = updated_docs_start_index
            self._instance_dict[instance_id]["docs_hashmap"] = updated_docs_hashmap  # 更新去重哈希表

            # 将结果存储到实例字典中（用于奖励计算）
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # 将元数据转换为指标
            metrics = {
                "query_count": metadata.get("query_count", 0),  # 查询数量
                "status": metadata.get("status", "unknown"),  # 搜索状态
                "total_results": metadata.get("total_results", 0),  # 总结果数
                "api_request_error": metadata.get("api_request_error"),  # API错误信息
                "docs_start_index": updated_docs_start_index,  # 更新后的文档起始索引
                "docs_hashmap_size": len(docs_hashmap),  # 去重哈希表大小
            }

            # Extract search_call_stats from api_response if available
            api_resp = metadata.get("api_response")
            if api_resp and isinstance(api_resp, dict) and "search_call_stats" in api_resp:
                metrics["search_call_stats"] = api_resp["search_call_stats"]

            # 如果设置了响应长度限制，则进行截断
            if self.max_tool_response_length:
                # 解析响应文本
                tool_response_text = json.loads(result_text)["result"]

                # 如果响应超过最大长度，根据配置进行截断
                if (
                    tool_response_text
                    and len(tool_response_text) > self.max_tool_response_length
                ):
                    if self.tool_response_truncate_side == "left":
                        # 保留左侧（前部）内容
                        tool_response_text = (
                            tool_response_text[: self.max_tool_response_length]
                            + "...(已截断)"
                        )
                    elif self.tool_response_truncate_side == "right":
                        # 保留右侧（后部）内容
                        tool_response_text = (
                            "(已截断)..."
                            + tool_response_text[-self.max_tool_response_length :]
                        )
                    else:
                        # 保留两端内容
                        length = self.max_tool_response_length // 2
                        tool_response_text = (
                            tool_response_text[:length]
                            + "...(已截断)..."
                            + tool_response_text[-length:]
                        )

                return (
                    ToolResponse(
                        text=json.dumps(
                            {"result": tool_response_text}, ensure_ascii=False
                        )
                    ),
                    0.0,
                    metrics,
                )
            else:
                # 不需要截断，直接返回
                return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            # 搜索执行失败，返回错误信息
            error_result = json.dumps(
                {"result": f"搜索执行失败: {e}"}, ensure_ascii=False
            )
            # logger.error(f"[SearchTool] 执行失败: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """计算工具实例的奖励。

        Args:
            instance_id: 工具实例ID
            **kwargs: 其他参数（当前未使用）

        Returns:
            str: 奖励列表，包含所有搜索结果
        """
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """释放工具实例。

        清理实例数据，释放内存。

        Args:
            instance_id: 工具实例ID
            **kwargs: 其他参数（当前未使用）
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
