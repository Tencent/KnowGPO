#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索工具工具集 - 搜索结果的去重、格式化和API调用

该模块提供了搜索相关的核心工具函数，包括搜索结果去重、引文格式化、
API调用封装和批量搜索处理等功能。

主要功能：
- ✅ 搜索结果去重和长度控制
- ✅ 引文格式化输出
- ✅ 搜索API调用封装（带重试机制）
- ✅ 批量搜索处理
- ✅ 权威性级别映射

核心特性：
- **智能去重**: 基于标题、域名和内容前100字符进行文档去重
- **长度控制**: 自动控制文档总长度，防止token超限
- **统一格式**: 提供标准的网页引文格式化模板
- **重试机制**: 对API调用失败进行自动重试，支持递增延迟
- **权威性评估**: 支持文档权威性级别的中英文映射
- **并发安全**: 支持信号量控制的并发搜索

使用示例：
    from search_utils import perform_single_search_batch, deduplicate_and_fill_docs

    # 批量搜索示例
    result_text, metadata = perform_single_search_batch(
        retrieval_service_url="https://search.example.com/api",
        query_list=["量化新规", "流动性风险"],
        topk=20,
        doc_start_idx=0
    )

    # 文档去重示例
    docs, hashmap, stats = deduplicate_and_fill_docs(
        docs_candidate=search_results,
        docs_hashmap={},
        total_budget=24576
    )

    # 引文格式化示例
    formatted_result, index = format_search_results(docs_list, global_index=0)

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""

import json
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Optional, List, Tuple, Dict
import os

import requests
import concurrent.futures

# from .call_vendor_search_toolcall import call_search_query_group
from .call_as_vendor_search_toolcall import call_as_search_subquery_list
from .call_refiner_agent import refine_docs_batch, SHORT_CONTENT_THRESHOLD

# ============================================================================
# 常量配置
# ============================================================================

DEFAULT_TIMEOUT = 30  # 默认搜索请求超时时间（秒）
MAX_RETRIES = 5  # 最大重试次数
INITIAL_RETRY_DELAY = 1  # 初始重试延迟（秒）
API_TIMEOUT = 10  # API超时时间（秒）
NO_SUMMARY_THRESHOLD  = 300  # passage长度小于此值直接返回原文
MAX_SUBQ_PER_ROUND = 5      # 每轮subq数量限制
TOP_K_DOCS_PER_SUBQ = 10    # 每个subq截取top10
REFINE_MODEL = 'gemini-2.5-flash-lite'  # 摘要模型
FILTER_ZERO_RELEVANCE = True  # 过滤relevance=0的文档
MAX_WORKERS = 6

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ============================================================================
# 网页引文模板
# ============================================================================

# 网页引文格式化模板，用于将搜索结果格式化为统一的引文格式
WEB_TPL = """[webpage {idx} begin]
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
[webpage {idx} end]"""


def deduplicate_and_fill_docs_as(
    results: Dict[str, Tuple[List[Any], List[Any]]],
    docs_hashmap: dict,
    total_budget: int = 24576
) -> Tuple[Dict[str, Tuple[List[Any], List[Any]]], dict, dict]:
    """
    对每个subquery的docs进行去重，返回去重后的results

    Args:
        results: {subquery: (docs, docs_expanded)} 格式的搜索结果
        docs_hashmap: 用于去重的hashmap
        total_budget: 每个subquery的总长度预算

    Returns:
        Tuple[去重后的results, 更新后的docs_hashmap, 统计信息]
    """
    deduplicated_results = {}
    total_stats = {
        'original_count': 0,
        'added_count': 0,
        'removed_duplicates': 0,
        'final_count': 0,
        'total_length': 0
    }

    for subquery, (docs, docs_expanded) in results.items():
        # 合并 docs 和 docs_expanded 进行去重
        all_docs = docs + docs_expanded

        if len(all_docs) == 0:
            deduplicated_results[subquery] = ([], [])
            continue

        final_docs = []
        final_docs_expanded = []
        current_total_length = 0
        removed_duplicates = 0
        added_count = 0

        # 先处理 docs
        for doc in docs:
            doc_key = f"{doc.get('title', '')}__{doc.get('domain', '')}__{doc.get('content', '')[:100]}"
            if doc_key in docs_hashmap:
                removed_duplicates += 1
                continue

            doc_length = len(doc.get('content', ''))
            if current_total_length + doc_length <= total_budget:
                docs_hashmap[doc_key] = True
                final_docs.append(doc)
                current_total_length += doc_length
                added_count += 1
            else:
                break

        # 再处理 docs_expanded
        for doc in docs_expanded:
            doc_key = f"{doc.get('title', '')}__{doc.get('domain', '')}__{doc.get('content', '')[:100]}"
            if doc_key in docs_hashmap:
                removed_duplicates += 1
                continue

            doc_length = len(doc.get('content', ''))
            if current_total_length + doc_length <= total_budget:
                docs_hashmap[doc_key] = True
                final_docs_expanded.append(doc)
                current_total_length += doc_length
                added_count += 1
            else:
                break

        deduplicated_results[subquery] = (final_docs, final_docs_expanded)

        # 累计统计
        total_stats['original_count'] += len(all_docs)
        total_stats['added_count'] += added_count
        total_stats['removed_duplicates'] += removed_duplicates
        total_stats['final_count'] += len(final_docs) + len(final_docs_expanded)
        total_stats['total_length'] += current_total_length

    return deduplicated_results, docs_hashmap, total_stats


def format_search_results_as(
    query_list: List[str],
    results: Dict[str, Tuple[List[Any], List[Any]]],
    top_k: int = 10,
    global_index: int = 0
) -> Tuple[str, int]:
    """
    遍历query_list，取results里对应的docs的top_k，用template拼接

    Args:
        query_list: 查询列表
        results: {subquery: (docs, docs_expanded)} 格式的搜索结果
        top_k: 每个query取的文档数量，默认10
        global_index: 全局索引起始值

    Returns:
        Tuple[拼接后的搜索结果字符串, 更新后的global_index]
    """
    all_search_results = []

    for query in query_list:
        if query not in results:
            all_search_results.append(f"No results found for query: '{query}'. Use a less specific query.")
            continue

        docs, docs_expanded = results[query]
        # 合并docs，优先取docs，再取docs_expanded
        all_docs = docs # + docs_expanded
        top_docs = all_docs[:top_k]

        if not top_docs:
            all_search_results.append(f"No results found for query: '{query}'. Use a less specific query.")
            continue

        # 拼接该query的搜索结果
        query_result = f"A search for {query} found the following results:\n\n"

        for doc in top_docs:
            query_result += WEB_TPL.format(
                idx=global_index + 1,
                title=doc.get('title', ''),
                domain=doc.get('url', ''),
                date=doc.get('date', ''),
                auth=doc.get('auth', ''),
                passage=doc.get('snippet', doc.get('content', ''))  # 优先使用snippet，如果不存在则fallback到content
            ) + '\n\n'
            global_index += 1

        all_search_results.append(query_result.strip())

    # 用 \n\n 拼接所有query的结果
    final_result = "\n\n".join(all_search_results).strip()

    return final_result, global_index

# ============================================================================
# 权威性级别映射
# ============================================================================

# 文档权威性级别中文映射表
AUTH_LEVEL_CH = {
    0: "极低权威性",  # 权威性最低
    1: "低权威性",  # 权威性较低
    2: "中等权威性",  # 权威性中等
    3: "高权威性",  # 权威性较高
    4: "极高权威性",  # 权威性最高
    -1: "暂无权威性信息",  # 未知权威性
}


def _aggregate_refine_stats(stats_list: list) -> dict:
    """Aggregate refine stats from multiple subqueries into one summary dict."""
    if not stats_list:
        return {}
    agg = {
        "total_docs": 0,
        "output_docs": 0,
        "refine_mode": stats_list[0].get("refine_mode", "unknown"),
        "refine_retries": 0,
        "refine_api_duration": 0.0,
        "refine_local_duration": 0.0,
        "refine_total_duration": 0.0,
        "short_content_preserved_count": 0,
        "refined_count": 0,
        "error_count": 0,
        "empty_content_count": 0,
        "zero_relevance_filtered_count": 0,
        "avg_compression_ratio": 0.0,
        "avg_relevance": 0.0,
        "status_distribution": {},
        "relevance_distribution": {"0-0.1": 0, "0.2-0.4": 0, "0.5-0.7": 0, "0.8-1.0": 0},
    }
    total_compression = 0.0
    total_relevance = 0.0
    count_compression = 0
    count_relevance = 0

    for s in stats_list:
        agg["total_docs"] += s.get("total_docs", 0)
        agg["output_docs"] += s.get("output_docs", 0)
        agg["refine_retries"] += s.get("refine_retries", 0)
        agg["refine_api_duration"] = max(agg["refine_api_duration"], s.get("refine_api_duration", 0.0))
        agg["refine_local_duration"] = max(agg["refine_local_duration"], s.get("refine_local_duration", 0.0))
        agg["refine_total_duration"] = max(agg["refine_total_duration"], s.get("refine_total_duration", 0.0))
        agg["short_content_preserved_count"] += s.get("short_content_preserved_count", 0)
        agg["refined_count"] += s.get("refined_count", 0)
        agg["error_count"] += s.get("error_count", 0)
        agg["empty_content_count"] += s.get("empty_content_count", 0)
        agg["zero_relevance_filtered_count"] += s.get("zero_relevance_filtered_count", 0)
        # Accumulate for avg
        n = s.get("total_docs", 0)
        if n > 0 and s.get("avg_compression_ratio", 0) > 0:
            total_compression += s["avg_compression_ratio"] * n
            count_compression += n
        if n > 0 and s.get("avg_relevance", 0) > 0:
            total_relevance += s["avg_relevance"] * n
            count_relevance += n
        # Merge status_distribution
        for k, v in s.get("status_distribution", {}).items():
            agg["status_distribution"][k] = agg["status_distribution"].get(k, 0) + v
        # Merge relevance_distribution
        for k, v in s.get("relevance_distribution", {}).items():
            agg["relevance_distribution"][k] = agg["relevance_distribution"].get(k, 0) + v

    if count_compression > 0:
        agg["avg_compression_ratio"] = round(total_compression / count_compression, 4)
    if count_relevance > 0:
        agg["avg_relevance"] = round(total_relevance / count_relevance, 4)

    return agg


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    timeout: int = DEFAULT_TIMEOUT,
    docs_hashmap: dict = None,
    docs_start_index: int = 0,
    query: str = "",
) -> tuple[Optional[dict[str, Any]], Optional[str], dict, int]:
    """调用远程搜索API执行检索，带有重试逻辑。

    该函数实现了完整的搜索流程：
    1. 调用外部搜索服务获取原始结果
    2. 对结果进行去重和长度控制（使用轨迹级别的docs_hashmap）
    3. 根据轮次进行结果截断（首轮全部，后续top10）
    4. 对服务器错误进行自动重试，使用递增延迟

    Args:
        retrieval_service_url: 检索服务API的URL
        query_list: 搜索查询列表
        timeout: 请求超时时间（秒）
        docs_hashmap: 轨迹级别的文档去重哈希表（会被修改以实现去重）
        docs_start_index: 当前轨迹的文档起始索引
        query: 原始用户问题

    Returns:
        tuple: (response_json, error_message, updated_docs_hashmap, updated_docs_start_index)
            - response_json: API返回的JSON对象，成功时不为None
            - error_message: 错误信息，成功时为None
            - updated_docs_hashmap: 更新后的文档去重哈希表
            - updated_docs_start_index: 更新后的文档起始索引
    Note:
        - 对500/502/503/504错误进行自动重试
        - docs_hashmap会被修改以实现轨迹内的文档去重
    """
    # 生成唯一请求ID用于日志跟踪
    request_id = str(uuid.uuid4())
    log_prefix = f"[搜索请求ID: {request_id}] "
    # 限制每轮subq数量
    queries = query_list[:MAX_SUBQ_PER_ROUND]

    # 如果没有传入docs_hashmap，则创建一个新的（用于本次搜索的去重）
    if docs_hashmap is None:
        docs_hashmap = {}

    last_error = None  # 记录最后一次错误
    current_docs_start_index = docs_start_index  # 当前文档索引

    # Timing and stats tracking for this call
    search_call_stats = {
        "search_api_duration": 0.0,
        "search_api_retries": 0,
        "dedup_duration": 0.0,
        "refine_duration": 0.0,
        "refine_stats": {},
        "format_duration": 0.0,
        "total_call_duration": 0.0,
    }
    call_total_start = time.time()

    # 尝试最多MAX_RETRIES次
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}尝试 {attempt + 1}/{MAX_RETRIES}: 调用搜索API {retrieval_service_url}"
            )

            # 调用搜索服务获取文档
            # 注意: 新版 call_as_vendor_search_toolcall 走 PromptSearch 后端, 强依赖 prompt
            # (用户原始问题) 来做查询改写和相关性排序; 必须把 query 传下去, 否则
            # 服务端会用 subquery 拼接做 fallback prompt, 检索质量会显著下降.
            search_api_start = time.time()
            as_search_results = call_as_search_subquery_list(queries, prompt=query)
            search_call_stats["search_api_duration"] = time.time() - search_api_start
            search_call_stats["search_api_retries"] = attempt


            # 对文档进行去重和长度控制（200000字符限制）
            # 传入轨迹级别的docs_hashmap，实现跨轮次去重
            dedup_start = time.time()
            as_search_results, docs_hashmap, stats = deduplicate_and_fill_docs_as(
                    as_search_results, docs_hashmap, 200000,
                )
            search_call_stats["dedup_duration"] = time.time() - dedup_start

            # ============ 核心改动：每个subq分别取top10，然后过refiner ============
             # 1. 按queries顺序收集每个subq的top_docs
            subq_top_docs_list = []  # [(subq, top_docs), ...] 保持顺序
            total_docs_before = 0
            for subq in queries:  # 按queries顺序遍历
                if subq in as_search_results:
                    docs, docs_expanded = as_search_results[subq]
                    top_docs = docs[:TOP_K_DOCS_PER_SUBQ]
                else:
                    top_docs = []
                subq_top_docs_list.append((subq, top_docs))
                total_docs_before += len(top_docs)
            
            # 2. 并发调用refiner处理每个subq的docs（使用全部queries）
            total_docs_after = 0
            
            def refine_subq_docs(idx, subq, top_docs):
                """并发处理单个subq的docs，返回idx保持顺序"""
                if not top_docs:
                    return idx, subq, [], {}
                refined_docs, r_stats = refine_docs_batch(
                    prompt=query,
                    subqueries=queries,  # 使用全部queries
                    docs=top_docs,
                    model=REFINE_MODEL,
                    short_content_threshold=NO_SUMMARY_THRESHOLD,
                    filter_zero_relevance=FILTER_ZERO_RELEVANCE
                )
                return idx, subq, refined_docs, r_stats
            
            # 并发执行，收集结果
            refine_start = time.time()
            results_by_idx = {}
            all_refine_stats = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(subq_top_docs_list)) as executor:
                futures = [
                    executor.submit(refine_subq_docs, idx, subq, top_docs)
                    for idx, (subq, top_docs) in enumerate(subq_top_docs_list)
                ]
                for future in concurrent.futures.as_completed(futures):
                    idx, subq, refined_docs, r_stats = future.result()
                    results_by_idx[idx] = (subq, refined_docs)
                    total_docs_after += len(refined_docs)
                    if r_stats:
                        all_refine_stats.append(r_stats)
            search_call_stats["refine_duration"] = time.time() - refine_start
            # Aggregate refine stats from all subqueries
            search_call_stats["refine_stats"] = _aggregate_refine_stats(all_refine_stats)
            
            # 3. 按原始顺序构建refined_results
            refined_results = {}
            for idx in range(len(subq_top_docs_list)):
                subq, refined_docs = results_by_idx[idx]
                refined_results[subq] = (refined_docs, [])
            
            print(f"[Refine] 各subq取top{TOP_K_DOCS_PER_SUBQ}后共 {total_docs_before} 个文档")
            print(f"[Refine] 摘要提取完成，保留 {total_docs_after} 个文档（过滤relevance=0）")
            
            # 使用format_search_results_as格式化，传入当前文档索引
            format_start = time.time()
            if total_docs_after > 0:
                as_search_results_str, updated_docs_start_index = format_search_results_as(
                    queries, refined_results, TOP_K_DOCS_PER_SUBQ, global_index=current_docs_start_index
                )
                search_call_stats["format_duration"] = time.time() - format_start
                status = "success"
                total_results = total_docs_after
            elif total_docs_before == 0:
                if stats['final_count'] == 0:
                    print('没搜到结果！query太差了！')
                    as_search_results_str = f"No results found for '{queries}'. Try with more general queries."
                else:
                    print('搜到了结果，但是去重后为空！')
                    as_search_results_str = f"deduplicated search results for '{queries}' are empty. Try with different queries."
                status = "no_results"
                total_results = 0
                updated_docs_start_index = current_docs_start_index  # 没有新文档，索引不变
            else:
                # 有文档但全部被过滤（relevance=0）
                print('搜到了结果，但refine后全部被过滤！')
                as_search_results_str = f"All {total_docs_before} documents were filtered out due to low relevance. Try with different queries."
                status = "no_results"
                total_results = 0
                updated_docs_start_index = current_docs_start_index  # 没有有效文档，索引不变

            # 构造模拟HTTP响应对象
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(
                {"result": as_search_results_str, "status": status, "total_results": total_results}, ensure_ascii=False
            ).encode("utf-8")

            # 检查服务器错误（500/502/503/504），需要重试
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API请求错误: 服务器错误 ({response.status_code}) "
                    f"尝试 {attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)

                # 如果还有重试机会，等待后重试
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # 递增延迟
                    # logger.info(f"{log_prefix}{delay}秒后重试...")
                    time.sleep(delay)
                continue

            # 检查其他HTTP错误（如4xx）
            response.raise_for_status()

            # 成功（状态码2xx）
            # logger.info(f"{log_prefix}搜索API调用成功，尝试 {attempt + 1} 次")
            search_call_stats["total_call_duration"] = time.time() - call_total_start
            resp_json = response.json()
            resp_json["search_call_stats"] = search_call_stats
            return resp_json, None, docs_hashmap, updated_docs_start_index
        except Exception as e:
            # 发生意外错误，记录并退出重试循环
            last_error = f"{log_prefix}意外错误: {e}"
            break  # 退出重试循环

    # 如果所有重试都失败，返回最后一次错误，索引不变
    logger.error(f"{log_prefix}搜索API调用失败。最后错误: {last_error}")
    return None, (
        last_error.replace(log_prefix, "API调用失败: ")
        if last_error
        else "重试后 API调用失败"
    ), docs_hashmap, current_docs_start_index



def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
    docs_hashmap: dict = None,
    docs_start_index: int = 0,
    query: str = "",
) -> tuple[str, dict[str, Any], dict, int]:
    """执行单次批量搜索（原始搜索工具行为）。

    该函数是搜索工具的主入口，负责：
    1. 调用搜索API获取原始结果
    2. 处理API响应和错误
    3. 格式化搜索结果为引文格式
    4. 收集和返回元数据
    5. 返回更新后的文档索引和去重哈希表

    Args:
        retrieval_service_url: 检索服务API的URL
        query_list: 搜索查询列表
        concurrent_semaphore: 可选的信号量，用于并发控制
        timeout: 请求超时时间（秒）
        docs_hashmap: 轨迹级别的文档去重哈希表（会被修改）
        docs_start_index: 当前轨迹的文档起始索引
        query: 原始用户问题

    Returns:
        tuple: (result_text, metadata, updated_docs_hashmap, updated_docs_start_index)
            - result_text: 搜索结果JSON字符串
            - metadata: 批量搜索的元数据字典
            - updated_docs_hashmap: 更新后的文档去重哈希表
            - updated_docs_start_index: 更新后的文档起始索引
    """
    # logger.info(f"开始批量搜索，共 {len(query_list)} 个查询。")

    # 如果没有传入docs_hashmap，则创建一个新的
    if docs_hashmap is None:
        docs_hashmap = {}

    api_response = None  # API响应
    error_msg = None  # 错误信息
    updated_docs_start_index = docs_start_index  # 初始化更新后的索引
    updated_docs_hashmap = docs_hashmap if docs_hashmap is not None else {}  # 初始化更新后的哈希表

    try:
        # 如果提供了信号量，使用它进行并发控制
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg, updated_docs_hashmap, updated_docs_start_index = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    timeout=timeout,
                    docs_hashmap=updated_docs_hashmap,
                    docs_start_index=docs_start_index,
                    query=query,
                )
        else:
            # 直接调用API，无并发控制
            api_response, error_msg, updated_docs_hashmap, updated_docs_start_index = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                timeout=timeout,
                docs_hashmap=updated_docs_hashmap,
                docs_start_index=docs_start_index,
                query=query,
            )
    except Exception as e:
        error_msg = f"批量搜索过程中API请求异常: {e}"
        # logger.error(f"批量搜索: {error_msg}")
        # logger.exception(e)

    # 初始化元数据
    metadata = {
        "query_count": len(query_list),  # 查询数量
        "queries": query_list,  # 查询列表
        "api_request_error": error_msg,  # API错误信息
        "api_response": None,  # API响应
        "status": "unknown",  # 状态
        "total_results": 0,  # 总结果数
        "formatted_result": None,  # 格式化后的结果
    }

    # 默认错误响应
    result_text = json.dumps(
        {"result": "搜索请求失败或重试后超时。"}, ensure_ascii=False
    )

    # 处理API响应
    if error_msg:
        # API调用失败
        metadata["status"] = "api_error"
        result_text = json.dumps(
            {"result": f"搜索错误: {error_msg}"}, ensure_ascii=False
        )
        # logger.error(f"批量搜索: 发生API错误: {error_msg}")
    elif api_response:
        # API调用成功
        # logger.debug(f"批量搜索: API响应: {api_response}")
        metadata["api_response"] = api_response

        try:
            # 提取原始结果
            search_status = api_response.get("status", "unknown")
            total_results = api_response.get("total_results", 0)
            search_result = api_response.get("result", "")

            if search_status == "success":
                # 格式化搜索结果为引文格式
                result_text = json.dumps({"result": search_result}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = search_result
                # logger.info(f"批量搜索: 成功，获得 {total_results} 个结果")
            else:
                # 没有搜索结果
                result_text = json.dumps(
                    {"result": search_result}, ensure_ascii=False
                )
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                # logger.info("批量搜索: 未找到结果")
        except Exception as e:
            # 处理结果时出错
            error_msg = f"处理搜索结果时出错: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            # logger.error(f"批量搜索: {error_msg}")
    else:
        # 未知API状态
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "未知API状态（无响应且无错误信息）。"}, ensure_ascii=False
        )
        logger.error("批量搜索: 未知API状态。")

    return result_text, metadata, updated_docs_hashmap, updated_docs_start_index


def main():
    """主函数：演示搜索工具的使用。

    该函数提供了搜索工具的演示用例，包括：
    1. 不同起始索引的搜索测试
    2. 错误处理演示
    3. 结果格式化展示

    Returns:
        bool: 执行是否成功
    """
    # 配置参数
    config = {
        "retrieval_service_url": "https://search.example.com/api",
        "test_queries": ["你好", "你是谁"],
        "test_cases": [
            {"name": "后续轮次搜索", "doc_start_idx": 20},
            {"name": "首轮搜索", "doc_start_idx": 0}
        ]
    }

    print("🚀 开始搜索工具演示")
    print(f"服务地址: {config['retrieval_service_url']}")
    print(f"测试查询: {config['test_queries']}")
    print("-" * 50)

    success_count = 0
    total_cases = len(config['test_cases'])

    for i, test_case in enumerate(config['test_cases'], 1):
        print(f"📋 测试用例 {i}/{total_cases}: {test_case['name']}")
        print(f"   起始索引: {test_case['doc_start_idx']}")

        try:
            # 执行搜索
            result_text, metadata = perform_single_search_batch(
                retrieval_service_url=config["retrieval_service_url"],
                query_list=config["test_queries"],
                doc_start_idx=test_case["doc_start_idx"],
            )

            # 解析结果
            result_data = json.loads(result_text)

            print(f"✅ 搜索状态: {metadata.get('status', 'unknown')}")
            print(f"   结果数量: {metadata.get('total_results', 0)}")

            # 显示结果摘要
            if metadata.get('total_results', 0) > 0:
                result_content = result_data.get('result', '')
                # 显示前200字符作为预览
                preview = result_content[:200] + "..." if len(result_content) > 200 else result_content
                print(f"   结果预览: {preview}")
            else:
                print("   无搜索结果")

            success_count += 1

        except Exception as e:
            print(f"❌ 测试用例失败: {e}")
            logger.error(f"测试用例 {test_case['name']} 失败: {e}")

        print("-" * 50)

    # 总结报告
    print("📊 测试总结")
    print(f"   总测试用例: {total_cases}")
    print(f"   成功用例: {success_count}")
    print(f"   失败用例: {total_cases - success_count}")

    if success_count == total_cases:
        print("🎉 所有测试用例执行成功！")
    else:
        print("⚠️  部分测试用例失败，请检查日志获取详细信息")

    return success_count == total_cases


if __name__ == "__main__":
    main()
