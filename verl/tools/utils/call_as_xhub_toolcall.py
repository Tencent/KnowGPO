#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
新版搜索接口调用封装 — 基于 PromptSearch API (vr-mcp-server)

该模块通过统一的 PromptSearch 接口完成搜索，不再区分中英文走不同通道。
对外暴露的函数签名与旧版 call_as_vendor_search_tool_call_w_google.py 完全兼容。
"""
import re
import traceback
import json
import requests
import uuid
from typing import Tuple, Optional, Union, List, Any, Dict
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# PromptSearch API 配置
# ============================================================

PROMPT_SEARCH_ENDPOINT = (
    "https://search.example.com/api"
    "/trpc.tsbs.vr_mcp_server.VrMcpServerHttp/PromptSearch"
)

# 流量类型
YB_TRAFFIC_TYPE_NORMAL = 0    # 正常流量
YB_TRAFFIC_TYPE_TRAIN = 1     # 训练流量
YB_TRAFFIC_TYPE_EVAL = 2      # 评估流量

# 默认 source 标识
DEFAULT_SOURCE = "agentic_search"
# 训练流量 source 标识（训练流量务必单独标识）
TRAIN_SOURCE = "Rhino_Bird_Deepsearch"

# 默认 ext_params
DEFAULT_EXT_PARAMS = {
    "open_google_api": "1",
    "open_wx_video": "0",
    "filter_level": "0",
    "enable_full_content": "1",
}

# 单 prompt 最大支持的 sub_query 数量
MAX_SUBQ_PER_PROMPT = 5


# ============================================================
# Query 清洗（sanitize_query）
# ============================================================

# 搜索引擎高级语法 pattern（site:xxx, inurl:xxx, intitle:xxx, filetype:xxx 等）
_ADVANCED_SYNTAX_RE = re.compile(
    r'\b(?:site|inurl|intitle|allintitle|intext|allintext|filetype|ext|cache|related|info|define|link)'
    r':\S*',
    re.IGNORECASE,
)

# JSON 转义残留 pattern
_JSON_ESCAPE_RE = re.compile(r'\\[nrtbf/\\\"\']')

# 纯标点/符号 pattern
_ONLY_PUNCT_RE = re.compile(r'^[\s\W]*$')


def sanitize_query(query: str) -> Optional[str]:
    """
    清洗模型生成的搜索 query，移除无效字符、JSON 转义残留和搜索引擎高级语法。

    处理规则：
    1. strip 前后空白
    2. 移除 JSON 转义残留（双重转义序列）
    3. 移除反斜杠
    4. 移除引号 → 替换为空格
    5. 移除搜索引擎高级语法（site:xxx, inurl:xxx 等）
    6. 移除控制字符
    7. 合并连续空格
    8. 清洗后为空或仅含标点符号 → 返回 None

    Returns:
        清洗后的查询字符串，如果无效则返回 None
    """
    if not query:
        return None

    q = query.strip()

    # 移除 JSON 转义残留
    q = _JSON_ESCAPE_RE.sub(" ", q)

    # 移除所有反斜杠
    q = q.replace("\\", "")

    # 移除引号 → 替换为空格
    for ch in ("'", '"', "\u2018", "\u2019", "\u201c", "\u201d"):
        q = q.replace(ch, " ")

    # 移除搜索引擎高级语法
    q = _ADVANCED_SYNTAX_RE.sub("", q)

    # 移除控制字符
    q = re.sub(r'[\x00-\x1f]+', ' ', q)

    # 合并连续空格并 strip
    q = re.sub(r'\s+', ' ', q).strip()

    if not q:
        return None

    if _ONLY_PUNCT_RE.match(q):
        return None

    return q


# ============================================================
# 工具函数
# ============================================================

def extract_domain(url: str) -> str:
    """从 URL 中提取域名"""
    try:
        parsed_url = urlparse(url)
        return parsed_url.netloc
    except Exception:
        return ""


# 引文构建模版
YB_WEB_TPL = """[webpage {idx} begin]
[webpage title]
{title}
[webpage url]
{domain}
[webpage date published]
{time}
[webpage authoritativeness]
{auth}
[webpage content begin]
{passage}
[webpage content end]
[webpage {idx} end]"""

# 权威性映射
AUTH_LEVEL_MAP = {
    0: "极低权威性",
    1: "低权威性",
    2: "中等权威性",
    3: "高权威性",
    4: "极高权威性",
    -1: "暂无权威性信息",
}


def get_auth_level_str(qd_auth_level) -> str:
    """获取权威性字符串"""
    try:
        auth = int(qd_auth_level)
        return AUTH_LEVEL_MAP.get(auth, "未知")
    except Exception:
        return "未知"


# ============================================================
# PromptSearch API 调用
# ============================================================

def _build_prompt_search_request(
    prompt: str,
    subquery_list: List[str],
    traffic_type: int = YB_TRAFFIC_TYPE_TRAIN,
    source: Optional[str] = None,
    ext_params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    构建 PromptSearch API 请求体。

    Args:
        prompt: 用户原始 prompt
        subquery_list: 子查询列表（已清洗）
        traffic_type: 流量类型（0=正常, 1=训练, 2=评估）
        source: 业务标识
        ext_params: 额外扩展参数

    Returns:
        请求体字典
    """
    trace_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # 确定 source
    if source is None:
        source = TRAIN_SOURCE if traffic_type == YB_TRAFFIC_TYPE_TRAIN else DEFAULT_SOURCE

    # 构建 sub_query_info（限制最大数量）
    sub_query_info = [
        {"subq": sq, "search_type": 0}
        for sq in subquery_list[:MAX_SUBQ_PER_PROMPT]
    ]

    # 合并 ext_params
    _ext_params = DEFAULT_EXT_PARAMS.copy()
    if ext_params:
        _ext_params.update(ext_params)

    req_data = {
        "header": {
            "session_id": session_id,
            "uuid": session_id,
            "trace_id": trace_id,
            "yb_traffic_type": traffic_type,
        },
        "prompt": prompt,
        "source": source,
        "sub_query_info": sub_query_info,
        "params": {
            "ext_params": _ext_params,
        },
    }

    return req_data


def _parse_prompt_search_response(
    response_json: Dict[str, Any],
    subquery_list: List[str],
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """
    解析 PromptSearch API 响应，转换为与旧版兼容的格式。

    响应中的 docs 字段包含：
        title, url, publish_time, abstract, abstract_clean, content, site_name, site_icon

    转换为旧版格式：
        title, content, passage, passage_long, domain, url, date, res_type, auth, qd_auth_level, score, recall_by

    Args:
        response_json: API 响应 JSON
        subquery_list: 子查询列表（用于按 subquery 分组结果）

    Returns:
        Dict[str, Tuple[List[Dict], List[Dict]]]，key 为 subquery，value 为 (docs, docs_expanded)
    """
    results = {}

    # 尝试从响应中提取 docs
    # PromptSearch 可能返回：
    # 1. 按 sub_query_info 分组的结果
    # 2. 或者一个扁平的 docs 列表
    raw_docs = response_json.get("docs", [])
    sub_query_results = response_json.get("sub_query_results", [])

    if sub_query_results:
        # 按 sub_query 分组的结果
        for i, sq_result in enumerate(sub_query_results):
            if i < len(subquery_list):
                subq = subquery_list[i]
            else:
                subq = sq_result.get("subq", f"unknown_subq_{i}")

            sq_docs = sq_result.get("docs", [])
            converted_docs = _convert_docs(sq_docs)
            results[subq] = (converted_docs, [])
            print(f"[PromptSearch] 查询 '{subq}' 获得 {len(converted_docs)} 条文档")
    elif raw_docs:
        # 扁平的 docs 列表，均匀分配给各 subquery
        all_converted = _convert_docs(raw_docs)
        if len(subquery_list) == 1:
            results[subquery_list[0]] = (all_converted, [])
            print(f"[PromptSearch] 查询 '{subquery_list[0]}' 获得 {len(all_converted)} 条文档")
        else:
            # 多个 subquery 但返回扁平 docs → 每个 subquery 都拿到全部结果
            for subq in subquery_list:
                results[subq] = (all_converted, [])
                print(f"[PromptSearch] 查询 '{subq}' 获得 {len(all_converted)} 条文档（共享）")
    else:
        # 无结果
        for subq in subquery_list:
            results[subq] = ([], [])
            print(f"[PromptSearch] 查询 '{subq}' 无结果")

    return results


def _convert_docs(raw_docs: List[Dict]) -> List[Dict]:
    """
    将 PromptSearch 返回的 docs 转换为旧版格式。

    PromptSearch docs 字段:
        title, url, publish_time, abstract, abstract_clean, content, site_name, site_icon

    旧版 doc 字段:
        title, content, passage, passage_long, domain, url, date, res_type, auth, qd_auth_level, score, recall_by
    """
    converted = []
    for doc in raw_docs:
        try:
            url = doc.get("url", "")
            title = doc.get("title", "")
            publish_time = doc.get("publish_time", "")
            abstract = doc.get("abstract", "")
            abstract_clean = doc.get("abstract_clean", "")
            content = doc.get("content", "")
            site_name = doc.get("site_name", "")

            # 优先使用 content（文档总结），其次 abstract_clean，最后 abstract
            main_content = content if content else (abstract_clean if abstract_clean else abstract)
            # passage 使用 abstract_clean 或 abstract
            passage = abstract_clean if abstract_clean else abstract

            domain = extract_domain(url) if url else site_name

            doc_item = {
                "title": title,
                "content": main_content,
                "passage": passage,
                "passage_long": content,
                "domain": domain,
                "url": url,
                "date": publish_time,
                "res_type": "prompt_search",
                "auth": "中等权威性",       # 默认中等权威性
                "qd_auth_level": 2,
                "score": 0.0,
                "recall_by": "prompt_search",
                "site_name": site_name,      # 额外保留
                "site_icon": doc.get("site_icon", ""),
            }
            converted.append(doc_item)
        except Exception as e:
            print(f"[PromptSearch] 解析单条 doc 失败: {e}")
            continue

    return converted


def call_prompt_search(
    prompt: str,
    subquery_list: List[str],
    traffic_type: int = YB_TRAFFIC_TYPE_TRAIN,
    source: Optional[str] = None,
    ext_params: Optional[Dict[str, str]] = None,
    timeout: float = 15.0,
    max_retries: int = 3,
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """
    调用 PromptSearch API 执行搜索。

    Args:
        prompt: 用户原始问题
        subquery_list: 子查询列表（已清洗）
        traffic_type: 流量类型
        source: 业务标识
        ext_params: 额外扩展参数
        timeout: 请求超时
        max_retries: 最大重试次数

    Returns:
        Dict[str, Tuple[List[Dict], List[Dict]]]，与旧版 call_as_search_subquery_list 返回格式一致
    """
    req_data = _build_prompt_search_request(
        prompt=prompt,
        subquery_list=subquery_list,
        traffic_type=traffic_type,
        source=source,
        ext_params=ext_params,
    )

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                PROMPT_SEARCH_ENDPOINT,
                headers=headers,
                json=req_data,
                timeout=timeout,
            )

            if response.status_code != 200:
                print(
                    f"[PromptSearch] HTTP {response.status_code} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    continue
                # 最后一次重试仍失败
                return {subq: ([], []) for subq in subquery_list}

            result = response.json()
            return _parse_prompt_search_response(result, subquery_list)

        except requests.exceptions.Timeout:
            print(
                f"[PromptSearch] 超时 (attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                continue
        except Exception as e:
            print(f"[PromptSearch] 请求异常: {e} (attempt {attempt + 1}/{max_retries})")
            traceback.print_exc()
            if attempt < max_retries - 1:
                continue

    # 所有重试失败
    return {subq: ([], []) for subq in subquery_list}


# ============================================================
# 对外兼容接口（与 w_google 版本签名一致）
# ============================================================

def call_as_search_subquery_list(
    subquery_list: List[str],
    prompt: str = "",
    traffic_type: int = YB_TRAFFIC_TYPE_TRAIN,
    source: Optional[str] = None,
    ext_params: Optional[Dict[str, str]] = None,
    timeout: float = 15.0,
    max_retries: int = 3,
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """
    批量搜索入口 — 兼容旧版 call_as_search_subquery_list 接口。

    Args:
        subquery_list: 子查询列表
        prompt: 用户原始问题（可选，不传则用 subquery 拼接）
        traffic_type: 流量类型（0=正常, 1=训练, 2=评估）
        source: 业务标识
        ext_params: 额外扩展参数
        timeout: 请求超时
        max_retries: 最大重试次数

    Returns:
        Dict[str, Tuple[List[Dict], List[Dict]]]
        key 为 subquery，value 为 (docs, docs_expanded)
    """
    results = {}

    # 1. 清洗所有 query，过滤无效的
    cleaned_subquery_list = []
    for subquery in subquery_list:
        cleaned = sanitize_query(subquery)
        if cleaned is None:
            print(f"call_as_search_subquery_list: 清洗后 query 无效，跳过。原始 query='{subquery}'")
            continue
        if cleaned != subquery:
            print(f"call_as_search_subquery_list: query 已清洗 '{subquery}' -> '{cleaned}'")
        cleaned_subquery_list.append(cleaned)

    if not cleaned_subquery_list:
        print("call_as_search_subquery_list: 所有 query 清洗后均无效，返回空结果")
        return results

    # 2. 如果没有传入 prompt，用 subquery 拼接
    if not prompt:
        prompt = " ".join(cleaned_subquery_list)

    # 3. 调用 PromptSearch API（一次性传入所有 subquery）
    print(f"call_as_search_subquery_list: 调用 PromptSearch，"
          f"prompt='{prompt[:50]}...', subqueries={cleaned_subquery_list}")

    results = call_prompt_search(
        prompt=prompt,
        subquery_list=cleaned_subquery_list,
        traffic_type=traffic_type,
        source=source,
        ext_params=ext_params,
        timeout=timeout,
        max_retries=max_retries,
    )

    return results


# ============================================================
# 去重、格式化等兼容函数
# ============================================================

def deduplicate_and_fill_docs(
    results: Dict[str, Tuple[List[Dict], List[Dict]]],
    docs_hashmap: dict,
    total_budget: int = 24576,
) -> Tuple[Dict[str, Tuple[List[Dict], List[Dict]]], dict, dict]:
    """
    对搜索结果去重（兼容旧版接口）。

    Args:
        results: {subquery: (docs, docs_expanded)} 格式的搜索结果
        docs_hashmap: 用于去重的 hashmap
        total_budget: 每个 subquery 的总长度预算

    Returns:
        Tuple[去重后的 results, 更新后的 docs_hashmap, 统计信息]
    """
    deduplicated_results = {}
    total_stats = {
        'original_count': 0,
        'added_count': 0,
        'removed_duplicates': 0,
        'final_count': 0,
        'total_length': 0,
    }

    for subquery, (docs, docs_expanded) in results.items():
        all_docs = docs + docs_expanded

        if len(all_docs) == 0:
            deduplicated_results[subquery] = ([], [])
            continue

        final_docs = []
        final_docs_expanded = []
        current_total_length = 0
        removed_duplicates = 0
        added_count = 0

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

        total_stats['original_count'] += len(all_docs)
        total_stats['added_count'] += added_count
        total_stats['removed_duplicates'] += removed_duplicates
        total_stats['final_count'] += len(final_docs) + len(final_docs_expanded)
        total_stats['total_length'] += current_total_length

    return deduplicated_results, docs_hashmap, total_stats


def format_search_results(
    query_list: List[str],
    results: Dict[str, Tuple[List[Dict], List[Dict]]],
    top_k: int = 10,
    global_index: int = 0,
) -> Tuple[str, int]:
    """
    格式化搜索结果为引文字符串（兼容旧版接口）。

    Args:
        query_list: 查询列表
        results: {subquery: (docs, docs_expanded)} 格式的搜索结果
        top_k: 每个 query 取的文档数量
        global_index: 全局索引起始值

    Returns:
        Tuple[拼接后的搜索结果字符串, 更新后的 global_index]
    """
    all_search_results = []

    for query in query_list:
        if query not in results:
            all_search_results.append(
                f"No results found for query: '{query}'. Use a less specific query."
            )
            continue

        docs, docs_expanded = results[query]
        all_docs = docs + docs_expanded
        top_docs = all_docs[:top_k]

        if not top_docs:
            all_search_results.append(
                f"No results found for query: '{query}'. Use a less specific query."
            )
            continue

        query_result = f"A search for {query} found the following results:\n\n"

        for doc in top_docs:
            query_result += YB_WEB_TPL.format(
                idx=global_index + 1,
                title=doc.get('title', ''),
                domain=doc.get('url', ''),
                time=doc.get('date', ''),
                auth=doc.get('auth', ''),
                passage=doc.get('snippet', doc.get('content', '')),
            ) + '\n\n'
            global_index += 1

        all_search_results.append(query_result.strip())

    final_result = "\n\n".join(all_search_results)
    return final_result, global_index


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    test_queries_cn = ["（匿名机构）股价", "北京旅游攻略"]
    test_queries_en = ["what is reinforcement learning"]
    all_queries = test_queries_cn + test_queries_en

    print("=" * 60)
    print("测试 PromptSearch API")
    print("=" * 60)

    # 测试 sanitize_query
    test_dirty_queries = ["' \\", "\\", "site:state.gov \\", " host venue", "normal query"]
    for q in test_dirty_queries:
        cleaned = sanitize_query(q)
        print(f"  sanitize_query('{q}') = {cleaned!r}")

    print("\n" + "=" * 60)
    print("开始搜索...")
    print("=" * 60)

    results = call_as_search_subquery_list(
        all_queries,
        prompt="（匿名机构）今天股价怎么样，北京旅游攻略，什么是强化学习",
    )

    for subquery, (docs, docs_expanded) in results.items():
        print(f"\n【查询】: {subquery}")
        print(f"  召回文档数: {len(docs)}, 扩展文档数: {len(docs_expanded)}")
        if docs:
            for i, doc in enumerate(docs[:3]):
                print(f"  [{i+1}] {doc.get('title', 'N/A')[:50]}")
                print(f"      URL: {doc.get('url', 'N/A')[:60]}")
                print(f"      内容: {doc.get('content', 'N/A')[:]}...")
