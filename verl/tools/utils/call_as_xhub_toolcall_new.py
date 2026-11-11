#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
新版 vendor_search AS 接口调用封装
对齐 call_vendor_search_new.py 的调用方式
"""
import traceback
import json
import requests
import uuid
import base64
from enum import Enum
from typing import Tuple, Optional, Union, List, Any, Dict
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# 配置常量
# ============================================================

def encode(text: str) -> str:
    """Base64编码"""
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


# Endpoint
VENDOR_SEARCH_ENDPOINT_ONLINE = "https://search.example.com/api"
VENDOR_SEARCH_ENDPOINT_PRE = "https://search.example.com/api"

# Scene ID
VENDOR_SEARCH_DSAI_SCENE_ID = "8001"
VENDOR_SEARCH_AS_SCENE_ID = "8003"
VENDOR_SEARCH_DUMP_SCENE_ID = "9010"

# Debug logtrace
VENDOR_SEARCH_LOGTRACE = '{"business":"web","level":3,"modules":["vendor_search", "onebox_search_frame"],"nodes":[]}'

# 环境
VENDOR_SEARCH_ENV_ONLINE = "online"
VENDOR_SEARCH_ENV_PRE = "pre"


class RequestType(Enum):
    FROM_VENDOR_MODEL = 0      # 元宝
    FROM_NSP = 1          # 立知
    FROM_AI_AGENT = 2
    FROM_COMMON_SEARCH = 3  # 商业
    FROM_SEARCH_AGENT = 4
    FROM_DS_SEARCH = 5    # DSAI / Agentic Search


# 默认 extends 配置
DEFAULT_EXTENDS = {
    RequestType.FROM_VENDOR_MODEL: {
        "open_query_rewrite": encode("on"),
    },
    RequestType.FROM_NSP: {
        "close_cnt_filter": encode("1"),
    },
    RequestType.FROM_COMMON_SEARCH: {
        "open_expand": encode("on"),
        "open_query_rewrite": encode("on"),
    },
    (RequestType.FROM_DS_SEARCH, VENDOR_SEARCH_DSAI_SCENE_ID): {
        "cs_open_weixin": encode("on"),
        "open_query_rewrite": encode("on"),
    },
    (RequestType.FROM_DS_SEARCH, VENDOR_SEARCH_AS_SCENE_ID): {
        "cs_open_weixin": encode("on"),
        "open_wiki": encode("1"),
    },
}


def get_default_extends(request_type: Union[RequestType, int], scene_id: Optional[str] = None) -> Dict[str, str]:
    """根据 request_type 和 scene_id 获取默认 extends"""
    if isinstance(request_type, int):
        request_type = RequestType(request_type)
    k = (request_type, scene_id) if scene_id else request_type
    if k not in DEFAULT_EXTENDS:
        raise ValueError(f"unsupported request type: {request_type.name}, scene id: {scene_id}")
    return DEFAULT_EXTENDS[k].copy()


# 默认实验配置
DEFAULT_EXP_INFO = json.dumps([
    {
        "layerType": "rankab",
        "experimentId": 123456,
        "expKey": "finance_zh_new",
        "params": {
            "open_miniprogram_docs": "on",
            "sr_topk_ratio_zero": "1",
            "disable_l1_rank": "1",
            "open_teg_searchapi": "1",
            "use_vendor_model_selection_v2": "1",
            "merge_search_api_v2": "1",
            "recall_rank_proxy_name": "RoughRank6BertInt8",
            "l3l4_merge": "1",
            "open_fresh_signal_v3": "1",
            "vendor_search_new_finance_queue": "1",
            "tune_finance_qdauth": "1",
            "open_wx_ocr": "1",
            "use_selection": "1",
            "open_plugin_priority": "1",
            "open_dc_version": "1",
            "vendor_search_global_search_mtp_reranker": "1",
            "eng_dedup_passage": "1",
            "apply_context_doc_filter": "1",
            "apply_context_doc_cutoff": "1",
            "context_doc_cutoff_count": "8",
            "selection_expand_count": "5",
            "open_select_snippets": "1",
            "plugin_weather_enable_rerank": "1",
            "plugin_rank_rela_90000500": "1",
            "plugin_exchange_enable_rerank": "1",
            "plugin_rerank_threshold_90000500": "0.799545440813141",
            "plugin_rank_rela_90000800": "1",
            "plugin_rerank_threshold_90000800": "0.746345471674881",
            "open_fit_intent": "1",
            "vendor_search_l3_expand_exp": "1",
            "vendor_search_main_queue_exp": "1",
            "vendor_search_no_extract_exp": "1",
            "vendor_search_l4_dedup_topk": "200",
            "vendor_search_l4_dedup_passage_topk": "200",
            "vendor_search_l3_expand_params": json.dumps({
                "l3_total_cnt": 1200,
                "main_topk": -1,
                "main_normal_max": -1,
                "main_normal_min": 30,
                "main_sr_max": -1,
                "main_sr_min": 10,
                "main_meta_max": -1,
                "main_meta_min": 5,
                "main_fz_max": -1,
                "main_fz_min": 10,
                "main_off_max": -1,
                "main_off_min": 5,
                "main_pdf_max": 5,
                "main_pdf_min": 5,
                "main_3h_max": -1,
                "main_3h_min": 6,
                "main_3h_ver_max": -1,
                "main_3h_ver_min": 11,
                "main_queue": -1,
                "kd_queue": 10,
                "authority_queue": 30,
                "baike_queue": 5,
                "economic_unified_queue": 20,
                "normal_en_queue": 0,
                "meta_en_queue": 0
            }, ensure_ascii=False),
            "l4_formula_all": "1",
            "force_eng_l5_rank": "1"
        },
        "layer": "1"
    },
    {
        "layerType": "rankab",
        "experimentId": 123456,
        "expKey": "author_2_content",
        "params": {
            "med_hunpai": "1",
            "med_hunpai_cnt": "20"
        },
        "layer": "1"
    }
], ensure_ascii=False)


def is_chinese_query(text: str) -> bool:
    """判断查询是否主要为中文"""
    if not text:
        return False
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_chars > len(text) * 0.1  # 中文字符占比超过10%


# ============================================================
# 引文模板和工具函数
# ============================================================

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


def extract_domain(url: str) -> str:
    """从URL中提取域名"""
    parsed_url = urlparse(url)
    return parsed_url.netloc


# ============================================================
# 核心函数
# ============================================================

def parse_data(origin) -> Tuple[List[Dict], List[Dict]]:
    """
    解析 vendor_search 返回结果
    
    Args:
        origin: vendor_search 返回的 JSON（可以是 str 或 dict）
        
    Returns:
        (docs, docs_expanded) 元组
    """
    try:
        if isinstance(origin, str):
            res = json.loads(origin)
        else:
            res = origin
    except Exception as e:
        print(f"parse_data 解析错误: {e}")
        return [], []
    
    if not res:
        print('返回结果为空！')
        return [], []
    
    try:
        item = res["passage_group"]
    except Exception as e:
        print(f"parse_data: passage_group 解析错误 {e}")
        return [], []
    
    if not item:
        print('返回结果为空！')
        return [], []
    
    docs_info = item["doc_infos"]
    docs = []
    docs_expanded = []
    
    for d_index, passage_info in enumerate(item["passage_infos"]):
        try:
            passage = passage_info["passage"]
            passage_long = passage_info["passage_long"]
            score = float(passage_info["score"])
            
            features = passage_info.get("features", {})
            date_str = features.get("date", "")
            recall_by = features.get("recall_by", "")
            res_type = features.get("res_type", "")
            is_expand = features.get("expand", "0")
            qd_auth_level = features.get("qd_auth_level", -1)
            
            url = docs_info[d_index]["url"]
            title = docs_info[d_index]["title"]
            
            # 微信文章 URL 处理
            if res_type == 'wechat':
                url = url + '/' + features.get("new_sn", "")
            
            # 医疗内容特殊处理
            if res_type == 'medical':
                passage = f"[medical] 该网页内容来自医院权威医生\n{passage}"
                qd_auth_level = 4
            
            domain = extract_domain(url)
            auth_string = get_auth_level_str(qd_auth_level)
            content = passage_long if passage_long else passage
            
            doc_item = {
                "title": title,
                "content": content,
                "passage": passage,
                "passage_long": passage_long,
                "domain": domain,
                "url": url,
                "date": date_str,
                "res_type": res_type,
                "auth": auth_string,
                "qd_auth_level": qd_auth_level,
                "score": score,
                "recall_by": recall_by
            }
            
            if is_expand == "1":
                docs_expanded.append(doc_item)
            else:
                docs.append(doc_item)
                
        except Exception as e:
            traceback.print_exc()
            print(f"过滤掉的 passage: {passage_info}")
            continue
    
    return docs, docs_expanded


def get_docs(
    query: str,
    env: str = VENDOR_SEARCH_ENV_ONLINE,
    req_type: Union[RequestType, int] = RequestType.FROM_DS_SEARCH,
    scene_id: str = VENDOR_SEARCH_AS_SCENE_ID,
    dump_scene_id: str = VENDOR_SEARCH_DUMP_SCENE_ID,
    exp_info: Optional[str] = None,
    extends: Optional[dict] = None,
    enable_debug: bool = False,
    timeout: float = 10.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    vendor_search 召回接口
    
    Args:
        query: 查询字符串
        env: 环境，"online" 或 "pre"
        req_type: 请求类型
        scene_id: 场景ID（用于选择 extends 配置）
        dump_scene_id: dump 场景ID（用于请求头）
        exp_info: 实验配置（不传则使用 DEFAULT_EXP_INFO）
        extends: 额外扩展参数
        enable_debug: 是否开启调试
        timeout: 超时时间
        
    Returns:
        (docs, docs_expanded) 元组
    """
    # 英文查询自动切换到 PRE 环境（PRE 对英文召回更好）
    _env = env
    if not is_chinese_query(query) and env == VENDOR_SEARCH_ENV_ONLINE:
        _env = VENDOR_SEARCH_ENV_PRE

    # 选择 endpoint
    if _env == VENDOR_SEARCH_ENV_ONLINE:
        endpoint = VENDOR_SEARCH_ENDPOINT_ONLINE
    elif _env == VENDOR_SEARCH_ENV_PRE:
        endpoint = VENDOR_SEARCH_ENDPOINT_PRE
    else:
        raise ValueError(f"invalid env: {_env}")
    
    # 获取 request_type 的值
    _req_type = req_type.value if isinstance(req_type, RequestType) else req_type
    
    # 处理 exp_info：如果未传入则使用默认实验配置
    _exp_info_str = exp_info if exp_info else DEFAULT_EXP_INFO
    _exp_info = encode(_exp_info_str)
    
    # 获取默认 extends 并合并自定义 extends
    _extends = get_default_extends(req_type, scene_id)
    if extends:
        _extends.update(extends)
    
    # 构建请求ID
    req_id = f"{dump_scene_id}_{uuid.uuid4()}_dump"
    
    # 构建请求数据
    req_data = {
        "request_header": {
            "request_id": req_id,
            "session_id": req_id,
            "guid": req_id,
            "scene": {"id": dump_scene_id},
            "enable_full_content": True,
            "enable_debug": True,
        },
        "request_type": _req_type,
        "extends": _extends,
        "query_group": {
            "origin_query": query,
            "all_queries": [{"query": query}] if is_chinese_query(query) else [
                {"query": query, "query_type": 0, "extends": {"en_search": "b24="}, "index": 0}
            ],
        },
        "exp_info": _exp_info,
    }
    
    if enable_debug:
        req_data["logtrace"] = VENDOR_SEARCH_LOGTRACE
    
    try:
        response = requests.post(endpoint, json=req_data, timeout=timeout)
        if response.status_code != 200:
            print(f"response.status_code != 200, status_code: {response.status_code}")
            return [], []
        
        result = response.json()
        docs, docs_expanded = parse_data(result)
        return docs, docs_expanded
        
    except Exception as e:
        print(f"get_docs 执行错误: {e}")
        traceback.print_exc()
        return [], []


def call_as_search_query_group(
    subquery: str,
    env: str = VENDOR_SEARCH_ENV_ONLINE,
    req_type: Union[RequestType, int] = RequestType.FROM_DS_SEARCH,
    scene_id: str = VENDOR_SEARCH_AS_SCENE_ID,
    dump_scene_id: str = VENDOR_SEARCH_DUMP_SCENE_ID,
    exp_info: Optional[str] = None,
    extends: Optional[dict] = None,
    enable_debug: bool = False,
    timeout: float = 10.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    单查询搜索
    
    Args:
        subquery: 子查询
        其他参数同 get_docs
        
    Returns:
        (docs, docs_expanded) 元组
    """
    try:
        print(f"call_as_search_query_group: query={subquery}, env={env}, scene_id={scene_id}")
        return get_docs(
            query=subquery,
            env=env,
            req_type=req_type,
            scene_id=scene_id,
            dump_scene_id=dump_scene_id,
            exp_info=exp_info,
            extends=extends,
            enable_debug=enable_debug,
            timeout=timeout,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"call_as_search_query_group 执行错误: {e}")
        return [], []


def call_as_search_subquery_list(
    subquery_list: List[str],
    env: str = VENDOR_SEARCH_ENV_ONLINE,
    req_type: Union[RequestType, int] = RequestType.FROM_DS_SEARCH,
    scene_id: str = VENDOR_SEARCH_AS_SCENE_ID,
    dump_scene_id: str = VENDOR_SEARCH_DUMP_SCENE_ID,
    exp_info: Optional[str] = None,
    extends: Optional[dict] = None,
    enable_debug: bool = False,
    timeout: float = 10.0,
    max_workers: int = 8,
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """
    并发批量搜索
    
    Args:
        subquery_list: 子查询列表
        max_workers: 最大并发数
        其他参数同 get_docs
        
    Returns:
        字典，key 为 subquery，value 为 (docs, docs_expanded) 元组
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subquery = {
            executor.submit(
                call_as_search_query_group,
                subquery,
                env=env,
                req_type=req_type,
                scene_id=scene_id,
                dump_scene_id=dump_scene_id,
                exp_info=exp_info,
                extends=extends,
                enable_debug=enable_debug,
                timeout=timeout,
            ): subquery
            for subquery in subquery_list
        }
        
        for future in as_completed(future_to_subquery):
            subquery = future_to_subquery[future]
            try:
                docs, docs_expanded = future.result()
                results[subquery] = (docs, docs_expanded)
            except Exception as e:
                print(f"查询 '{subquery}' 执行失败: {e}")
                results[subquery] = ([], [])
    
    return results


def deduplicate_and_fill_docs(
    results: Dict[str, Tuple[List[Dict], List[Dict]]],
    docs_hashmap: dict,
    total_budget: int = 24576
) -> Tuple[Dict[str, Tuple[List[Dict], List[Dict]]], dict, dict]:
    """
    对搜索结果去重
    
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
        'total_length': 0
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
    global_index: int = 0
) -> Tuple[str, int]:
    """
    格式化搜索结果为引文字符串
    
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
            all_search_results.append(f"No results found for query: '{query}'. Use a less specific query.")
            continue
        
        docs, docs_expanded = results[query]
        all_docs = docs + docs_expanded
        top_docs = all_docs[:top_k]
        
        if not top_docs:
            all_search_results.append(f"No results found for query: '{query}'. Use a less specific query.")
            continue
        
        query_result = f"A search for {query} found the following results:\n\n"
        
        for doc in top_docs:
            query_result += YB_WEB_TPL.format(
                idx=global_index + 1,
                title=doc.get('title', ''),
                domain=doc.get('url', ''),
                time=doc.get('date', ''),
                auth=doc.get('auth', ''),
                passage=doc.get('snippet',doc.get('content', ''))
            ) + '\n\n'
            global_index += 1
        
        all_search_results.append(query_result.strip())
    
    final_result = "\n\n".join(all_search_results)
    return final_result, global_index


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    test_queries_cn = ["王鹤棣 百科", "北京旅游攻略"]
    test_queries_en = ["what is reinforcement learning"]
    all_queries = test_queries_cn + test_queries_en

    print("=" * 60)
    print("测试新版 vendor_search AS 接口")
    print("=" * 60)

    # 测试 is_chinese_query
    for q in all_queries:
        print(f"  is_chinese_query('{q}') = {is_chinese_query(q)}")

    print("\n" + "=" * 60)
    print("开始搜索...")
    print("=" * 60)

    results = call_as_search_subquery_list(all_queries)

    for subquery, (docs, docs_expanded) in results.items():
        print(f"\n【查询】: {subquery} (中文={is_chinese_query(subquery)})")
        print(f"  召回文档数: {len(docs)}, 扩展文档数: {len(docs_expanded)}")
        if docs:
            for i, doc in enumerate(docs[:3]):
                print(f"  [{i+1}] {doc.get('title', 'N/A')[:50]}")
                print(f"      URL: {doc.get('url', 'N/A')[:60]}")
                print(f"      权威性: {doc.get('auth', 'N/A')}")
