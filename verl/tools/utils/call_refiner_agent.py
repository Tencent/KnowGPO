#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Refine Agent: 对搜索结果进行摘要提取
目的：输入 prompt, subqueries, docs，通过远程 BatchRefineDocument API 批量处理，
     输出每个 doc 抽取摘要后的结果。

调用策略：
仅调用远程 BatchRefineDocument API（批量处理，更快更稳定）。
如果远程 API 不可用/失败，不再进行本地 LLM 回退，直接返回空结果。
"""

import os
import json
import logging
import time
import requests
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ============================================================================
# BatchRefineDocument 远程 API 配置
# ============================================================================

BATCH_REFINE_API_URL = os.getenv(
    "BATCH_REFINE_API_URL",
    "https://example.invalid/refine"
)

BATCH_REFINE_API_ENABLED = os.getenv(
    "BATCH_REFINE_API_ENABLED", "true"
).lower() in ["true", "1", "yes", "on"]

BATCH_REFINE_API_TIMEOUT = int(os.getenv("BATCH_REFINE_API_TIMEOUT", "120"))
BATCH_REFINE_API_MAX_RETRIES = int(os.getenv("BATCH_REFINE_API_MAX_RETRIES", "2"))

# 短内容阈值（供外部 import 使用）
SHORT_CONTENT_THRESHOLD = 300


# ============================================================================
# 远程 API 调用
# ============================================================================

def call_batch_refine_api(
    prompt: str,
    subqueries: List[str],
    documents: List[Dict[str, Any]],
    timeout: int = BATCH_REFINE_API_TIMEOUT,
    max_retries: int = BATCH_REFINE_API_MAX_RETRIES,
) -> Optional[List[Dict[str, Any]]]:
    """
    调用远程 BatchRefineDocument API 批量处理文档。

    Returns:
        (result_docs, total_retries, api_duration)
        成功时 result_docs 为处理后的文档列表；失败时 result_docs 为 None
    """
    request_body = {
        "prompt": prompt,
        "subqueries": subqueries if subqueries else [],
        "documents": documents,
    }

    last_error = None
    total_retries = 0
    api_start_time = time.time()
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                BATCH_REFINE_API_URL,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()

            data = resp.json()

            res_info = data.get("res_info", {})
            if res_info.get("code", -1) != 0:
                msg = res_info.get("msg", "unknown error")
                logger.warning(
                    f"[BatchRefineAPI] business error (attempt {attempt+1}): "
                    f"code={res_info.get('code')}, msg={msg}"
                )
                last_error = msg
                total_retries += 1
                continue

            result_docs = data.get("documents", [])
            if not isinstance(result_docs, list):
                logger.warning(
                    f"[BatchRefineAPI] invalid documents type: {type(result_docs)}"
                )
                last_error = "invalid documents type"
                total_retries += 1
                continue

            logger.info(
                f"[BatchRefineAPI] success: input {len(documents)}, "
                f"output {len(result_docs)}"
            )
            api_duration = time.time() - api_start_time
            return result_docs, total_retries, api_duration

        except requests.exceptions.Timeout:
            logger.warning(
                f"[BatchRefineAPI] timeout (attempt {attempt+1}/{max_retries})"
            )
            last_error = "timeout"
            total_retries += 1
        except requests.exceptions.ConnectionError as e:
            logger.warning(
                f"[BatchRefineAPI] connection error (attempt {attempt+1}/{max_retries}): {e}"
            )
            last_error = str(e)
            total_retries += 1
        except Exception as e:
            logger.warning(
                f"[BatchRefineAPI] exception (attempt {attempt+1}/{max_retries}): {e}"
            )
            last_error = str(e)
            total_retries += 1

    logger.error(
        f"[BatchRefineAPI] all retries failed (last_error={last_error})"
    )
    api_duration = time.time() - api_start_time
    return None, total_retries, api_duration


# ============================================================================
# 公开接口（保持与外部调用方兼容的函数签名与返回字段）
# ============================================================================

def refine_docs_batch(
    prompt: str,
    subqueries: List[str],
    docs: List[Dict[str, Any]],
    model: str = 'summary_model',
    max_workers: int = 10,
    short_content_threshold: int = SHORT_CONTENT_THRESHOLD,
    filter_zero_relevance: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    批量对文档列表进行摘要提取。

    仅调用远程 BatchRefineDocument API；如果远程 API 失败或未启用，
    则返回空文档列表（不再回退到本地 LLM 处理）。

    注意：`model` 和 `max_workers` 参数仅为保持对外签名兼容而保留，
    当前实现不再使用它们。
    """
    refine_stats = {
        "total_docs": len(docs),
        "refine_mode": "none",  # "api" or "none"
        "refine_retries": 0,
        "refine_api_duration": 0.0,
        "refine_local_duration": 0.0,
        "refine_total_duration": 0.0,
        # Refine effectiveness stats
        "short_content_preserved_count": 0,
        "refined_count": 0,
        "error_count": 0,
        "empty_content_count": 0,
        "zero_relevance_filtered_count": 0,
        "avg_compression_ratio": 0.0,
        "avg_relevance": 0.0,
        "status_distribution": {},
        "relevance_distribution": {"0-0.1": 0, "0.2-0.4": 0, "0.5-0.7": 0, "0.8-1.0": 0},
        "rerefine_count": 0,
        "rerefine_local_duration": 0.0,
        "output_docs": 0,
    }

    if not docs:
        return [], refine_stats

    refine_total_start = time.time()
    results: Optional[List[Dict[str, Any]]] = None

    # ========== 一级调用：远程 BatchRefineDocument API ==========
    if BATCH_REFINE_API_ENABLED:
        try:
            # 构造 API 请求的 documents 格式
            api_documents = []
            for doc in docs:
                api_doc = {
                    "title": doc.get("title", ""),
                    "content": (
                        doc.get("passage_long", "")
                        or doc.get("content", "")
                        or doc.get("passage", "")
                    ),
                    "url": doc.get("url", ""),
                    "date": doc.get("date", ""),
                    "auth": doc.get("auth", ""),
                }
                api_documents.append(api_doc)

            api_result_tuple = call_batch_refine_api(
                prompt=prompt,
                subqueries=subqueries if subqueries else [],
                documents=api_documents,
            )
            api_results, api_retries, api_duration = api_result_tuple
            refine_stats["refine_retries"] += api_retries
            refine_stats["refine_api_duration"] = api_duration

            if api_results is not None and len(api_results) == len(docs):
                # 远程 API 成功，将结果合并回原始 doc 字段
                refine_stats["refine_mode"] = "api"
                results = []
                for orig_doc, api_doc in zip(docs, api_results):
                    merged = {**orig_doc}
                    merged["snippet"] = api_doc.get(
                        "snippet", api_doc.get("content", "")
                    )
                    merged["relevance"] = api_doc.get("relevance", 0.5)
                    merged["refine_status"] = api_doc.get(
                        "refine_status", "api_success"
                    )
                    results.append(merged)

                logger.info(
                    f"[refine_docs_batch] remote API processed {len(results)} docs"
                )
            elif api_results is not None:
                logger.warning(
                    f"[refine_docs_batch] remote API returned count({len(api_results)})"
                    f" mismatch with input({len(docs)})"
                )
        except Exception as e:
            logger.warning(
                f"[refine_docs_batch] remote API exception: {e}"
            )

    # ========== 远程 API 失败/未启用：直接返回原文档（不摘要） ==========
    if results is None:
        logger.warning(
            f"[refine_docs_batch] remote API unavailable, returning original docs "
            f"without refinement ({len(docs)} docs)"
        )
        refine_stats["refine_mode"] = "passthrough"
        results = []
        for orig_doc in docs:
            orig_content = (
                orig_doc.get("passage_long", "")
                or orig_doc.get("content", "")
                or orig_doc.get("passage", "")
            )
            merged = {**orig_doc}
            merged["snippet"] = orig_content
            merged["relevance"] = 0.5
            merged["refine_status"] = "api_unavailable_preserved"
            results.append(merged)

    # ========== 统计 refine 效果指标 ==========
    compression_ratios = []
    relevance_scores = []
    for i, doc in enumerate(results):
        if doc is None:
            continue
        status = doc.get("refine_status", "unknown")
        refine_stats["status_distribution"][status] = refine_stats["status_distribution"].get(status, 0) + 1

        if "short_content" in status:
            refine_stats["short_content_preserved_count"] += 1
        elif "error" in status or "executor_error" in status:
            refine_stats["error_count"] += 1
        elif "empty_content" in status:
            refine_stats["empty_content_count"] += 1
        else:
            refine_stats["refined_count"] += 1

        # Compression ratio
        orig_content = (
            docs[i].get("passage_long", "") or docs[i].get("content", "") or docs[i].get("passage", "")
        )
        snippet = doc.get("snippet", "")
        if orig_content and len(orig_content) > 0:
            ratio = len(snippet) / len(orig_content)
            compression_ratios.append(ratio)

        # Relevance distribution
        rel = doc.get("relevance", 0)
        relevance_scores.append(rel)
        if rel <= 0.1:
            refine_stats["relevance_distribution"]["0-0.1"] += 1
        elif rel <= 0.4:
            refine_stats["relevance_distribution"]["0.2-0.4"] += 1
        elif rel <= 0.7:
            refine_stats["relevance_distribution"]["0.5-0.7"] += 1
        else:
            refine_stats["relevance_distribution"]["0.8-1.0"] += 1

    if compression_ratios:
        refine_stats["avg_compression_ratio"] = round(sum(compression_ratios) / len(compression_ratios), 4)
    if relevance_scores:
        refine_stats["avg_relevance"] = round(sum(relevance_scores) / len(relevance_scores), 4)

    # 过滤 relevance=0 的文档
    if filter_zero_relevance:
        before_count = len(results)
        results = [
            doc for doc in results if doc and doc.get("relevance", 0) > 0
        ]
        refine_stats["zero_relevance_filtered_count"] = before_count - len(results)

    refine_stats["refine_total_duration"] = time.time() - refine_total_start
    refine_stats["output_docs"] = len(results)

    return results, refine_stats


def filter_by_relevance(
    refined_docs: List[Dict[str, Any]],
    min_relevance: float = 0.0,
) -> List[Dict[str, Any]]:
    """根据 relevance 分数过滤文档"""
    return [
        doc
        for doc in refined_docs
        if doc.get("relevance", 0) >= min_relevance
    ]


def sort_by_relevance(
    refined_docs: List[Dict[str, Any]],
    descending: bool = True,
) -> List[Dict[str, Any]]:
    """根据 relevance 分数排序文档"""
    return sorted(
        refined_docs,
        key=lambda x: x.get("relevance", 0),
        reverse=descending,
    )


if __name__ == "__main__":
    test_prompt = '在首次提到"两面包夹芝士"的视频中，up主最后提到的干员是什么职业'
    test_subqueries = [
        "两面包夹芝士首次提及",
        "两面包夹芝士视频",
        "两面包夹芝士up主",
        "两面包夹芝士干员",
    ]
    test_docs = [
        {
            "title": "两面包夹芝士是什么梗啥意思",
            "content": (
                '这个梗来源于b站的up主:一只大哈鱼,发的"3000块全没了!各位喝醉了一定要'
                '管好自己,我真服了!"的视频。其中的经典句"形成一个两面包夹之势"成为了'
                '"两面包夹芝士"的梗开始快速传播。'
            ),
            "url": "http://m.mnw.cn/keji/internet/2421282.html",
            "date": "2021-05-07",
            "auth": "普通网站",
        },
        {
            "title": "短内容测试",
            "content": "这是一个短内容，少于300字符，应该直接保留原文。",
            "url": "http://example.com/short",
            "date": "2025-01-01",
        },
    ]

    print("=" * 50)
    print("测试 BatchRefineDocument API ...")
    print("=" * 50)

    api_docs = [
        {
            "title": d.get("title", ""),
            "content": d.get("content", ""),
            "url": d.get("url", ""),
            "date": d.get("date", ""),
            "auth": d.get("auth", ""),
        }
        for d in test_docs
    ]

    api_result_tuple = call_batch_refine_api(
        prompt=test_prompt,
        subqueries=test_subqueries,
        documents=api_docs,
    )
    api_result, api_retries, api_dur = api_result_tuple

    if api_result is not None:
        print(f"\n远程 API 成功，返回 {len(api_result)} 篇文档 (retries={api_retries}, duration={api_dur:.2f}s):")
        for i, doc in enumerate(api_result):
            print(f"\n【文档 {i+1}】")
            print(f"  标题: {doc.get('title', 'N/A')}")
            print(f"  Snippet: {doc.get('snippet', 'N/A')[:200]}")
            print(f"  Relevance: {doc.get('relevance', 'N/A')}")
            print(f"  Status: {doc.get('refine_status', 'N/A')}")
            print(f"  Source: {doc.get('source', 'N/A')}")
    else:
        print("\n远程 API 不可用。")

    # 测试完整流程
    print("\n" + "=" * 50)
    print("测试 refine_docs_batch (仅远程 API) ...")
    print("=" * 50)

    results, stats = refine_docs_batch(
        prompt=test_prompt,
        subqueries=test_subqueries,
        docs=test_docs,
    )
    print(f"\nRefine Stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    for i, doc in enumerate(results):
        print(f"\n【文档 {i+1}】")
        print(f"  标题: {doc.get('title', 'N/A')}")
        print(f"  Snippet: {doc.get('snippet', 'N/A')[:200]}")
        print(f"  Relevance: {doc.get('relevance', 'N/A')}")
        print(f"  Status: {doc.get('refine_status', 'N/A')}")
