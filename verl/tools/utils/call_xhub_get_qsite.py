#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询站点推荐模块

该模块提供了基于查询内容推荐高权威站点的功能，
使用（匿名模型）AI模型分析查询意图，返回最相关的官方或权威站点列表。

主要功能：
- ✅ 支持单个或批量查询
- ✅ 并发处理提高效率
- ✅ 自动重试机制
- ✅ 智能站点解析
- ✅ 多模型支持
- ✅ 完整的错误处理

核心特性：
- **智能推荐**: 基于（匿名模型）AI模型分析查询意图，推荐高权威性站点
- **批量处理**: 支持批量查询并发处理，提高处理效率
- **重试机制**: 内置指数退避重试策略，提高调用成功率
- **结果解析**: 支持JSON、Python列表等多种格式的结果解析
- **类型安全**: 完整的类型提示和参数验证
- **易于集成**: 提供简洁的API接口，易于集成到其他系统

使用示例：
    from call_vendor_search_get_qsite import get_qsite_info

    # 单个查询
    query = "北京旅游局长"
    sites = get_qsite_info(query)
    print(f"推荐站点: {sites[0]}")

    # 批量查询
    queries = ["北京旅游局长", "上海美食推荐"]
    all_sites = get_qsite_info(queries, max_workers=5)
    for query, sites in zip(queries, all_sites):
        print(f"{query}: {sites}")

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""

import json
import uuid
import re
import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_qsite_info(sub_queries, max_workers=10, model="qsite_0.6b_v8.4"):
    """
    获取查询对应的推荐站点。

    该函数使用（匿名模型）AI模型分析查询内容，推荐能够解决该问题的高权威性或官方站点。
    支持批量查询并发处理，提高处理效率。

    Args:
        sub_queries: 查询列表或单个查询字符串
            - 列表示例：["北京旅游局长", "上海美食推荐"]
            - 字符串示例："北京旅游局长"
        max_workers: 并发线程数，默认10
            - 建议根据查询数量和系统资源调整
        model: 使用的模型名称，默认 "qsite_0.6b_v8.4"
            - 可选其他版本的qsite模型

    Returns:
        list: 每个查询对应的站点列表
            - 示例：[["site1.com", "site2.com"], ["site3.com"]]
            - 每个站点列表最多包含10个站点
            - 如果无推荐站点，返回空列表[]

    Example:
        >>> # 单个查询
        >>> sites = get_qsite_info("北京旅游局长")
        >>> print(sites[0])  # ['beijing.gov.cn', ...]

        >>> # 批量查询
        >>> queries = ["北京旅游局长", "上海美食推荐"]
        >>> all_sites = get_qsite_info(queries, max_workers=5)
        >>> for query, sites in zip(queries, all_sites):
        >>>     print(f"{query}: {sites}")
    """

    # ========================================================================
    # 系统提示词：定义AI助手的角色和输出格式
    # ========================================================================
    system = """
# 高权威站点推荐任务

## 任务描述
你是一个专业的信息检索助手，需要根据用户提出的问题，推荐能够解决该问题的高权威性或官方站点列表。若无推荐站点，输出[]

## 输出格式要求
- 最多给出10个站点，且站点不能重复
- 只输出域名，不包含协议（如http://、https://）
- 不包含具体路径或参数
- 严格按照以下JSON数组格式输出：

["example1.com", "example2.com", "example3.com"]

## 示例

输入：如何申请护照？

输出：["nia.gov.cn", "gaj.gz.gov.cn"]
"""

    def crawl_hyaide_result(message_log, model="planner"):
        """
        调用（匿名模型）AI模型接口。

        该函数封装了（匿名模型）AI模型的HTTP调用逻辑，包括请求构造、
        错误处理和自动重试机制。

        Args:
            message_log: 消息列表，包含system和user角色的对话
                格式：[{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."}]
            model: 模型名称，默认"planner"

        Returns:
            str: 模型返回的文本内容

        Raises:
            requests.RequestException: HTTP请求失败
            Exception: 其他异常（在重试3次后抛出）
        """
        # （匿名模型）AI服务配置
        url = os.getenv("KNOWGPO_LEGACY_QSITE_URL", "https://example.invalid/legacy-api")
        wsid = "10143"  # 服务组名称
        enable_stream = False  # 不使用流式输出

        # HTTP请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("KNOWGPO_SEARCH_API_KEY", ""),  # API密钥
            "Wsid": wsid,
        }
        # 构造请求数据
        req_data = {
            "model": model,  # 模型名称
            "query_id": "test_query_id_" + str(uuid.uuid4()),  # 唯一请求ID
            "messages": message_log,  # 对话消息
            "temperature": 0.1,  # 温度参数（0.1较低，输出更确定）
            "top_p": 0.8,  # 核采样参数
            "top_k": 20,  # Top-K采样
            "repetition_penalty": 1.0,  # 重复惩罚
            "output_seq_len": 128,  # 最大输出长度
            "max_input_seq_len": 10240,  # 最大输入长度
            "openai_infer": True,  # 使用OpenAI推理格式
            "stream": enable_stream,  # 是否流式输出
        }

        # 重试机制配置
        max_retries = 3  # 最大重试次数
        backoff = 0.5  # 初始退避时间（秒）
        last_exc = None  # 记录最后一次异常

        # 尝试最多max_retries次
        for attempt in range(max_retries):
            try:
                # 发送POST请求
                rspo = requests.post(url, json=req_data, headers=headers, stream=False)

                # 检查HTTP状态码
                if rspo.status_code != 200:
                    raise requests.RequestException(f"HTTP {rspo.status_code}")

                # 解析响应
                data = json.loads(rspo.text)
                choices = data["choices"]
                return choices[0]["message"]["content"]

            except Exception as e:
                last_exc = e
                # 如果还有重试机会，等待后重试
                if attempt < max_retries - 1:
                    wait_time = backoff * (2**attempt)  # 指数退避
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试都失败，抛出异常
                    raise last_exc

    def remove_think_tag(content: str) -> str:
        """
        去除模型输出中的思考标签。

        某些模型会在输出中包含<think>...</think>标签来展示思考过程，
        该函数用于清理这些标签，只保留最终结果。

        Args:
            content: 原始模型输出内容

        Returns:
            str: 清理后的内容

        Example:
            >>> text = "<think>分析中...</think>[\"site1.com\"]"
            >>> remove_think_tag(text)
            '["site1.com"]'
        """
        # 使用正则表达式移除<think>标签及其内容
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()

    def parse_predict_sites(predict_text: str) -> list:
        """
        解析模型预测结果中的站点列表。

        该函数尝试多种方式解析模型输出的站点列表：
        1. JSON解析
        2. Python eval解析
        3. 单个域名识别

        Args:
            predict_text: 模型输出的文本

        Returns:
            list: 站点域名列表，解析失败返回空列表

        Example:
            >>> parse_predict_sites('["site1.com", "site2.com"]')
            ['site1.com', 'site2.com']

            >>> parse_predict_sites('site1.com')
            ['site1.com']
        """
        # 检查输入是否为空
        if not predict_text or predict_text.strip() == "":
            return []

        predict_text = str(predict_text).strip()

        # 方法1：尝试JSON解析（最常见的格式）
        try:
            sites = json.loads(predict_text)
            if isinstance(sites, list):
                return sites
        except json.JSONDecodeError:
            pass  # JSON解析失败，尝试其他方法

        # 方法2：尝试Python eval解析（处理单引号等情况）
        try:
            sites = eval(predict_text)
            if isinstance(sites, list):
                return sites
        except Exception:
            pass  # eval解析失败，尝试其他方法

        # 方法3：检查是否为单个域名（无特殊字符）
        if re.match(r"^[a-zA-Z0-9.-]+$", predict_text):
            return [predict_text]

        # 所有解析方法都失败
        logger.error(f"[ERROR]无法解析的站点: {predict_text}")
        return []

    def process_single_query(query: str) -> list:
        """
        处理单个查询并返回推荐站点。

        该函数是单个查询的完整处理流程：
        1. 构造消息日志
        2. 调用AI模型
        3. 清理响应
        4. 解析站点列表

        Args:
            query: 用户查询字符串

        Returns:
            list: 推荐站点列表，失败返回空列表
        """
        try:
            # 构造对话消息（添加/no_think指令禁用思考过程输出）
            message_log = [
                {"role": "system", "content": system},
                {"role": "user", "content": query + " /no_think"},
            ]

            # 调用（匿名模型）AI模型
            response = crawl_hyaide_result(message_log, model)

            # 清理响应（移除思考标签）
            response = remove_think_tag(response)

            # 解析站点列表
            sites = parse_predict_sites(response)

            return sites

        except Exception as e:
            # 捕获异常，返回空列表
            logger.exception(f"[ERROR]处理查询 '{query}' 时发生错误: {e}")
            return []

    # ========================================================================
    # 主处理流程
    # ========================================================================

    # 如果输入是单个字符串，转换为列表
    if isinstance(sub_queries, str):
        sub_queries = [sub_queries]

    # 初始化结果列表（保持原始顺序）
    results = [[] for _ in range(len(sub_queries))]

    # 使用线程池并发处理所有查询
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        future_to_index = {
            executor.submit(process_single_query, query): i
            for i, query in enumerate(sub_queries)
        }

        # 收集完成的任务结果
        for future in as_completed(future_to_index):
            try:
                sites = future.result()  # 获取站点列表
                original_index = future_to_index[future]  # 获取原始索引
                results[original_index] = sites  # 按原始顺序存储结果
            except Exception as e:
                logger.exception(f"获取结果时发生错误: {e}")

    return results

# ============================================================================
# 使用示例和测试
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("查询站点推荐模块测试")
    print("=" * 80)

    # 测试1：单个查询
    print("\n【测试1】单个查询")
    query = "北京旅游局长"
    sites = get_qsite_info(query)
    print(f"查询: {query}")
    print(f"推荐站点: {sites[0]}")

    # 测试2：批量查询
    print("\n【测试2】批量查询")
    queries = ["中华联合财产保险分支机构法人资格"]
    all_sites = get_qsite_info(queries)

    for i, (query, sites) in enumerate(zip(queries, all_sites)):
        print(f"\n查询 {i+1}: {query}")
        print(f"推荐站点: {sites}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
