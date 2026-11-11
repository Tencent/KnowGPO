#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Refine Agent: 对搜索结果进行摘要提取
目的：输入prompt, subquery, docs，输出每个doc抽取摘要后的结果
策略：
1. 尽可能保召回，防止信息遗漏
2. 保准确，防止模型幻觉编造内容

字段对齐 call_as_vendor_search_toolcall.py:
- title: 文档标题
- content: 文档内容 (passage_long 或 passage)
- passage: 原始passage
- passage_long: 长passage
- domain: 域名
- url: 完整URL
- date: 发布日期
- res_type: 资源类型
- auth: 权威性字符串
- qd_auth_level: 权威性等级
- score: 召回分数
- recall_by: 召回方式
"""

import sys
import os
import re
import json
import json_repair
import concurrent.futures
import hashlib
import threading
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import call_model_api_with_messages


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ============================================================================
# URL缓存机制配置
# ============================================================================

# 缓存目录：优先使用VENDOR_MODEL_CKPT_PATH环境变量，否则使用默认路径
_VENDOR_MODEL_CKPT_PATH = os.getenv("CKPT_PATH", "")
if _VENDOR_MODEL_CKPT_PATH:
    _DEFAULT_CACHE_DIR = os.path.join(_VENDOR_MODEL_CKPT_PATH, "refine_cache")
else:
    _DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "refine_cache")
REFINE_CACHE_DIR = os.getenv("REFINE_CACHE_DIR", _DEFAULT_CACHE_DIR)

# 是否启用缓存（默认启用）
REFINE_CACHE_ENABLED = os.getenv("REFINE_CACHE_ENABLED", "true").lower() in ["true", "1", "yes", "on"]

# 线程锁，用于保证缓存操作线程安全
_cache_lock = threading.Lock()

# 内存缓存（URL -> 结果）
_memory_cache: Dict[str, Dict[str, Any]] = {}


def _get_cache_key(url: str, prompt: str) -> str:
    """
    根据URL和prompt生成缓存key
    
    Args:
        url: 网页URL
        prompt: 用户问题（不同问题对同一URL可能有不同的摘要）
        
    Returns:
        缓存key（MD5哈希）
    """
    combined = f"{url}|||{prompt}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def _get_cache_file_path(cache_key: str) -> str:
    """获取缓存文件路径"""
    return os.path.join(REFINE_CACHE_DIR, f"{cache_key}.json")


def get_from_cache(url: str, prompt: str) -> Optional[Dict[str, Any]]:
    """
    从缓存中获取结果（一级缓存）
    
    Args:
        url: 网页URL
        prompt: 用户问题
        
    Returns:
        缓存的结果，如果未命中返回None
    """
    if not REFINE_CACHE_ENABLED:
        return None
        
    cache_key = _get_cache_key(url, prompt)
    
    # 先检查内存缓存
    with _cache_lock:
        if cache_key in _memory_cache:
            logger.debug(f"[Cache] 内存缓存命中: {url[:50]}...")
            return _memory_cache[cache_key].copy()
    
    # 再检查文件缓存
    cache_file = _get_cache_file_path(cache_key)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
                # 同时加入内存缓存
                with _cache_lock:
                    _memory_cache[cache_key] = cached_result
                logger.debug(f"[Cache] 文件缓存命中: {url[:50]}...")
                return cached_result.copy()
        except Exception as e:
            logger.warning(f"[Cache] 读取缓存文件失败: {e}")
    
    return None


def save_to_cache(url: str, prompt: str, result: Dict[str, Any]) -> None:
    """
    保存结果到缓存
    
    Args:
        url: 网页URL
        prompt: 用户问题
        result: 处理结果
    """
    if not REFINE_CACHE_ENABLED:
        return
        
    cache_key = _get_cache_key(url, prompt)
    
    # 保存到内存缓存
    with _cache_lock:
        _memory_cache[cache_key] = result.copy()
    
    # 保存到文件缓存
    try:
        os.makedirs(REFINE_CACHE_DIR, exist_ok=True)
        cache_file = _get_cache_file_path(cache_key)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[Cache] 保存缓存文件失败: {e}")


# ============================================================================
# 三级调用机制
# ============================================================================

# summary_model 最大重试次数
SUMMARY_MODEL_MAX_RETRIES = 3

# 兜底模型名称
FALLBACK_MODEL = 'gemini-2.5-flash-lite'


def call_llm_with_fallback(
    messages: List[Dict[str, str]],
    primary_model: str = 'summary_model',
    fallback_model: str = FALLBACK_MODEL,
    max_retries: int = SUMMARY_MODEL_MAX_RETRIES
) -> Tuple[str, str]:
    """
    带兜底策略的LLM调用（二级+三级调用）
    
    调用策略：
    1. 首先使用 chat_with_model 调用 primary_model (summary_model)
    2. 如果 primary_model 失败（重试 max_retries 次后仍失败），
       则使用 call_model_api_with_messages 调用 fallback_model (gemini-2.5-flash-lite)
    
    Args:
        messages: 消息列表
        primary_model: 主模型名称，默认 summary_model
        fallback_model: 兜底模型名称，默认 gemini-2.5-flash-lite
        max_retries: 主模型最大重试次数，默认3
        
    Returns:
        (response_text, call_source) 元组
        call_source: "primary" | "fallback" | "failed"
    """
    # 二级调用：尝试使用 chat_with_model 调用 summary_model
    last_error = None
    for retry in range(max_retries):
        try:
            _, response_text, error = chat_with_model(
                primary_model, messages, tools=None, enable_thinking=False
            )
            if error:
                logger.warning(f"[LLM] chat_with_model 第{retry+1}次尝试错误: {error}")
                last_error = error
                continue
            if response_text:
                return response_text, "primary"
            else:
                logger.warning(f"[LLM] chat_with_model 第{retry+1}次返回空响应")
                last_error = "empty_response"
        except Exception as e:
            logger.warning(f"[LLM] chat_with_model 第{retry+1}次异常: {e}")
            last_error = str(e)
    
    # 三级调用：使用 call_model_api_with_messages 调用兜底模型
    logger.info(f"[LLM] summary_model 失败 ({last_error})，使用兜底模型 {fallback_model}")
    try:
        response_text, _ = call_model_api_with_messages(
            messages, model=fallback_model, max_retries=3
        )
        if response_text:
            return response_text, "fallback"
        else:
            logger.warning(f"[LLM] 兜底模型也返回空响应")
            return "", "failed"
    except Exception as e:
        logger.error(f"[LLM] 兜底模型调用失败: {e}")
        return "", "failed"

# ============================================================================
# Search Tool 保存机制配置
# ============================================================================

# 保存目录：支持环境变量配置，默认为项目根目录下的search_tool_logs
_DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "search_tool_logs")
SEARCH_TOOL_SAVE_DIR = os.getenv("SEARCH_TOOL_SAVE_DIR", _DEFAULT_SAVE_DIR)
SEARCH_TOOL_SAVE_FILE = os.path.join(SEARCH_TOOL_SAVE_DIR, "refine_results.jsonl")

# 是否启用保存功能（默认启用，可通过环境变量关闭）
SEARCH_TOOL_SAVE_ENABLED = os.getenv("SEARCH_TOOL_SAVE_ENABLED", "true").lower() in ["true", "1", "yes", "on"]


def _safe_get_str(obj: Any, key: str, default: str = "") -> str:
    """
    安全获取字符串值，处理None和非字符串类型
    
    Args:
        obj: 字典对象
        key: 键名
        default: 默认值
    
    Returns:
        字符串值
    """
    if obj is None:
        return default
    try:
        value = obj.get(key) if isinstance(obj, dict) else default
        if value is None:
            return default
        return str(value) if not isinstance(value, str) else value
    except Exception:
        return default


def _safe_get_content(webpage_info: Dict[str, Any]) -> str:
    """
    安全获取网页内容，按优先级尝试多个字段
    
    Args:
        webpage_info: 网页信息字典
    
    Returns:
        内容字符串，如果都不存在则返回空字符串
    """
    if not isinstance(webpage_info, dict):
        return ""
    
    # 按优先级尝试多个字段
    for field in ["passage_long", "content", "passage", "text", "body"]:
        try:
            content = webpage_info.get(field)
            if content and isinstance(content, str):
                return content
        except Exception:
            continue
    return ""


def _safe_get_number(obj: Any, key: str, default: float = 0.0) -> float:
    """
    安全获取数字值
    
    Args:
        obj: 字典对象
        key: 键名
        default: 默认值
    
    Returns:
        数字值
    """
    if obj is None:
        return default
    try:
        value = obj.get(key) if isinstance(obj, dict) else default
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def save_refine_result(
    query: str,
    subqueries: List[str],
    webpage_info: Dict[str, Any],
    refiner_result: Dict[str, Any]
) -> None:
    """
    实时保存refine结果到文件（增强鲁棒性版本）
    
    该函数会安全处理各种异常情况：
    - None值处理
    - 类型错误处理
    - 文件IO错误处理
    - 编码错误处理
    - 目录权限问题
    
    可通过环境变量配置：
    - SEARCH_TOOL_SAVE_ENABLED: 是否启用保存（默认true）
    - SEARCH_TOOL_SAVE_DIR: 保存目录路径
    
    Args:
        query: 用户原始问题
        subqueries: 子查询列表
        webpage_info: 网页信息（包含title, url, content等）
        refiner_result: refine后的结果（包含snippet, relevance等）
    """
    # 0. 检查是否启用保存功能
    if not SEARCH_TOOL_SAVE_ENABLED:
        return
    
    try:
        # 1. 参数校验和容错处理
        if query is None:
            query = ""
        if not isinstance(query, str):
            query = str(query)
            
        if subqueries is None:
            subqueries = []
        if not isinstance(subqueries, list):
            subqueries = [str(subqueries)]
        # 确保subqueries中所有元素都是字符串
        subqueries = [str(sq) if sq is not None else "" for sq in subqueries]
            
        if webpage_info is None:
            webpage_info = {}
        if not isinstance(webpage_info, dict):
            webpage_info = {}
            
        if refiner_result is None:
            refiner_result = {}
        if not isinstance(refiner_result, dict):
            refiner_result = {}
        
        # 2. 确保目录存在（带权限检查）
        try:
            os.makedirs(SEARCH_TOOL_SAVE_DIR, exist_ok=True)
        except PermissionError:
            logger.error(f"保存目录权限不足: {SEARCH_TOOL_SAVE_DIR}")
            return
        except Exception as e:
            logger.error(f"创建保存目录失败: {e}")
            return
        
        # 3. 安全获取各个字段
        content = _safe_get_content(webpage_info)
        snippet = _safe_get_str(refiner_result, "snippet")
        
        # 获取原始的passage字段
        passage = _safe_get_str(webpage_info, "passage")
        passage_long = _safe_get_str(webpage_info, "passage_long")
        content_field = _safe_get_str(webpage_info, "content")
        
        # 4. 构建保存记录（所有字段都做安全处理）
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "query": query[:2000],  # 限制长度防止超长query
                "subqueries": [sq[:1000] for sq in subqueries[:20]],  # 限制数量和长度
                "webpage_info": {
                    "title": _safe_get_str(webpage_info, "title")[:5000],
                    "url": _safe_get_str(webpage_info, "url")[:1000],
                    "domain": _safe_get_str(webpage_info, "domain")[:200],
                    "date": _safe_get_str(webpage_info, "date")[:50],
                    "auth": _safe_get_str(webpage_info, "auth")[:100],
                    "qd_auth_level": _safe_get_number(webpage_info, "qd_auth_level", -1),
                    "score": _safe_get_number(webpage_info, "score", 0.0),
                    "recall_by": _safe_get_str(webpage_info, "recall_by")[:50],
                    "res_type": _safe_get_str(webpage_info, "res_type")[:50],
                    # 保存完整的内容字段
                    "content": content_field,  # 原始content字段
                    "passage": passage,  # 原始passage字段
                    "passage_long": passage_long,  # 原始passage_long字段
                    "content_length": len(content) if content else 0,
                    "content_preview": content[:200] if content else ""
                },
                "refiner_result": {
                    "snippet": snippet,  # 保存完整snippet，不截断
                    "relevance": _safe_get_number(refiner_result, "relevance", 0.0),
                    "refine_status": _safe_get_str(refiner_result, "refine_status")[:100],
                    "snippet_length": len(snippet) if snippet else 0,
                    "has_parse_error": bool(refiner_result.get("parse_error", False))
                }
            }
        except Exception as e:
            logger.warning(f"构建保存记录失败: {e}")
            return
        
        # 5. 追加写入JSONL文件（带编码和IO错误处理）
        try:
            with open(SEARCH_TOOL_SAVE_FILE, "a", encoding="utf-8") as f:
                json_str = json.dumps(record, ensure_ascii=False)
                f.write(json_str + "\n")
                f.flush()  # 立即刷新到磁盘
        except UnicodeEncodeError as e:
            logger.warning(f"编码错误，尝试使用ASCII模式: {e}")
            try:
                with open(SEARCH_TOOL_SAVE_FILE, "a", encoding="utf-8") as f:
                    json_str = json.dumps(record, ensure_ascii=True)
                    f.write(json_str + "\n")
                    f.flush()
            except Exception as e2:
                logger.error(f"ASCII模式保存也失败: {e2}")
        except IOError as e:
            logger.error(f"文件IO错误: {e}")
        except Exception as e:
            logger.error(f"写入文件失败: {e}")
            
    except Exception as e:
        # 最外层兜底异常处理
        logger.error(f"保存refine结果发生未预期错误: {e}")
        try:
            # 尝试记录错误信息到日志
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
        except Exception:
            pass  # 连日志都记录不了，静默失败


# Refine Agent 的 System Prompt
REFINE_SYSTEM_PROMPT = """You are a precise information extraction assistant. Your task is to extract useful information from webpage content that can help answer a user's question.

Core Principles:
1. MAXIMIZE RECALL: Extract ALL potentially useful information. When uncertain, INCLUDE rather than exclude. Information may be useful even if not directly related to subqueries.
2. ENSURE ACCURACY: Only extract information explicitly stated in the content. NEVER add, infer, or fabricate any information.
3. PRESERVE ORIGINAL WORDING: For dates, numbers, names, quotes, and key facts, keep the EXACT original text. Do NOT paraphrase or summarize these details."""

# Refine Agent 的 User Prompt 模板
REFINE_USER_PROMPT_TEMPLATE = """Extract information from the webpage content that may help answer the user's question.

<instructions>
1. EXTRACTION RULES:
   - Extract ALL potentially useful information, not just what directly matches the question
   - PRESERVE EXACT ORIGINAL WORDING for: dates, numbers, names, titles, quotes, specific facts
   - Do NOT summarize or paraphrase key details - copy them verbatim
   - Include context around key facts (who said what, when, where)
   - Keep up to 200 words if the content is highly relevant

2. WHAT TO INCLUDE:
   - Information directly answering the question
   - Background context that might be useful
   - Related facts mentioned in the same content (they may answer follow-up parts of the question)
   - Specific details: names, dates, numbers, quotes - in their ORIGINAL form

3. If NO relevant information exists:
   - Set relevance to 0
   - Provide a 1-sentence summary of what the content is about
   - Do NOT mention that it doesn't answer the question

4. Relevance scoring (0 to 1):
   - 0.8-1.0: Directly answers or provides key information
   - 0.5-0.7: Contains useful supporting/background information
   - 0.2-0.4: Tangentially related, may provide context
   - 0-0.1: No relevant information

CRITICAL:
- NEVER fabricate information. Extract ONLY from the given content.
- PRESERVE original wording for specific facts to avoid attribution errors.
- Output ONLY valid JSON.
</instructions>

<content>
{text}
</content>

<question>
{query}
</question>

Output: {{"snippet": "extracted text preserving original wording", "relevance": 0.X}}"""

# 短内容阈值：低于此长度直接保留原文，不调用模型
SHORT_CONTENT_THRESHOLD = 300

# 单文档信息抽取 Prompt 模板
EXTRACT_SINGLE_DOC_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rationale", "evidence", "summary" fields**"""


def parse_refine_response(response_text: str) -> Dict[str, Any]:
    """
    解析模型返回的JSON结果

    Args:
        response_text: 模型返回的文本

    Returns:
        解析后的字典，包含snippet和relevance
    """
    try:
        # 尝试直接解析JSON
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        pass

    try:
        # 尝试使用json_repair修复并解析
        result = json_repair.loads(response_text)
        return result
    except Exception:
        pass

    # 尝试从文本中提取JSON
    import re
    json_pattern = r'\{[^{}]*"snippet"[^{}]*"relevance"[^{}]*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        try:
            result = json_repair.loads(match.group())
            return result
        except Exception:
            pass

    # 解析失败，返回默认值
    return {
        "snippet": response_text if response_text else "",
        "relevance": 0.5,
        "parse_error": True
    }


def refine_single_doc(
    prompt: str,
    doc: Dict[str, Any],
    model: str = 'summary_model',
    short_content_threshold: int = SHORT_CONTENT_THRESHOLD
) -> Dict[str, Any]:
    """
    对单个文档进行摘要提取（三级调用策略）
    
    调用策略：
    1. 一级缓存：检查URL是否命中缓存，命中则直接返回
    2. 二级调用：使用chat_with_model调用summary_model，最多重试3次
    3. 三级兜底：如果二级失败，使用call_model_api_with_messages调用gemini-2.5-flash-lite

    Args:
        prompt: 用户原始问题（用于判断相关性）
        doc: 文档字典，包含content, title, url等字段
        model: 使用的模型，默认summary_model
        short_content_threshold: 短内容阈值，低于此长度直接保留原文

    Returns:
        包含原始doc信息和提取结果的字典
    """
    content = doc.get('content', '') or doc.get('passage', '')
    url = doc.get('url', '')

    if not content:
        return {
            **doc,
            "snippet": "",
            "relevance": 0,
            "refine_status": "empty_content"
        }

    # ========== 一级缓存：检查URL缓存 ==========
    if url:
        cached_result = get_from_cache(url, prompt)
        if cached_result:
            logger.debug(f"[Refine] 缓存命中: {url[:50]}...")
            return {
                **doc,
                **cached_result,
                "refine_status": "cache_hit"
            }

    # 短内容优化：直接保留原文，不调用模型
    if len(content) <= short_content_threshold:
        result = {
            **doc,
            "snippet": content,  # 保留原文
            "relevance": 0.5,    # 默认中等相关度，让后续流程决定
            "refine_status": "short_content_preserved"
        }
        # 保存到缓存
        if url:
            save_to_cache(url, prompt, {"snippet": content, "relevance": 0.5})
        return result

    # 截断过长的内容，避免超出模型限制
    max_content_length = 8000
    if len(content) > max_content_length:
        content = content[:max_content_length]

    # 构建prompt - 使用用户原始问题而不是subquery
    user_prompt = REFINE_USER_PROMPT_TEMPLATE.format(
        text=content,
        query=prompt
    )

    messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # ========== 二级+三级调用：带兜底策略的LLM调用 ==========
    try:
        response_text, call_source = call_llm_with_fallback(
            messages, 
            primary_model=model,
            fallback_model=FALLBACK_MODEL,
            max_retries=SUMMARY_MODEL_MAX_RETRIES
        )

        if response_text:
            parsed_result = parse_refine_response(response_text)
            result = {
                **doc,
                "snippet": parsed_result.get("snippet", ""),
                "relevance": parsed_result.get("relevance", 0),
                "refine_status": f"refined_by_{call_source}",
                "refine_raw_response": response_text,
                "call_source": call_source
            }
            # 保存到缓存
            if url:
                save_to_cache(url, prompt, {
                    "snippet": parsed_result.get("snippet", ""),
                    "relevance": parsed_result.get("relevance", 0)
                })
            return result
        else:
            # 模型返回空，保留原文
            return {
                **doc,
                "snippet": content,
                "relevance": 0.5,
                "refine_status": "empty_response_preserved",
                "call_source": call_source
            }
    except Exception as e:
        # 出错时保留原文，避免信息丢失
        return {
            **doc,
            "snippet": content,
            "relevance": 0.5,
            "refine_status": f"error_preserved: {str(e)}"
        }


def call_refine_agent(
    prompt: str,
    docs: List[Dict[str, Any]],
    model: str = 'summary_model',
    max_workers: int = 10,
    short_content_threshold: int = SHORT_CONTENT_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    并发调用refine agent对所有文档进行摘要提取

    Args:
        prompt: 用户原始问题（用于判断相关性，比subquery更全面）
        docs: 文档列表
        model: 使用的模型，默认summary_model
        max_workers: 最大并发数，默认10
        short_content_threshold: 短内容阈值，低于此长度直接保留原文

    Returns:
        带有snippet和relevance的文档列表
    """
    if not docs:
        return []

    refined_docs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_doc_idx = {
            executor.submit(refine_single_doc, prompt, doc, model, short_content_threshold): idx
            for idx, doc in enumerate(docs)
        }

        # 收集结果，保持原始顺序
        results = [None] * len(docs)
        for future in concurrent.futures.as_completed(future_to_doc_idx):
            idx = future_to_doc_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"文档 {idx} 处理失败: {e}")
                # 出错时保留原文
                original_content = docs[idx].get('content', '') or docs[idx].get('passage', '')
                results[idx] = {
                    **docs[idx],
                    "snippet": original_content,
                    "relevance": 0.5,
                    "refine_status": f"executor_error_preserved: {str(e)}"
                }

        refined_docs = results

    return refined_docs


def call_refine_agent_with_subqueries(
    prompt: str,
    subquery_results: Dict[str, Tuple[List[Any], List[Any]]],
    model: str = 'summary_model',
    max_workers: int = 10,
    short_content_threshold: int = SHORT_CONTENT_THRESHOLD
) -> Dict[str, Tuple[List[Any], List[Any]]]:
    """
    对call_as_vendor_search_toolcall返回的结果进行refine处理
    输入格式与call_as_search_subquery_list返回格式一致

    Args:
        prompt: 用户原始问题（用于判断相关性）
        subquery_results: {subquery: (docs, docs_expanded)} 格式的搜索结果
        model: 使用的模型
        max_workers: 最大并发数
        short_content_threshold: 短内容阈值

    Returns:
        {subquery: (refined_docs, refined_docs_expanded)} 格式的结果
    """
    refined_results = {}

    # 收集所有需要处理的任务
    all_tasks = []
    for subquery, (docs, docs_expanded) in subquery_results.items():
        for doc in docs:
            all_tasks.append((subquery, doc, 'docs'))
        for doc in docs_expanded:
            all_tasks.append((subquery, doc, 'docs_expanded'))

    if not all_tasks:
        return subquery_results

    # 并发处理所有文档 - 使用 prompt 而不是 subquery
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(refine_single_doc, prompt, task[1], model, short_content_threshold): task
            for task in all_tasks
        }

        # 初始化结果结构
        temp_results = {
            subquery: {'docs': [], 'docs_expanded': []}
            for subquery in subquery_results.keys()
        }

        # 收集结果
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            subquery, original_doc, doc_type = task

            try:
                result = future.result()
                temp_results[subquery][doc_type].append(result)
            except Exception as e:
                print(f"处理失败 [{subquery}]: {e}")
                # 出错时保留原文
                original_content = original_doc.get('content', '') or original_doc.get('passage', '')
                temp_results[subquery][doc_type].append({
                    **original_doc,
                    "snippet": original_content,
                    "relevance": 0.5,
                    "refine_status": f"error_preserved: {str(e)}"
                })

    # 转换为最终格式
    for subquery in subquery_results.keys():
        refined_results[subquery] = (
            temp_results[subquery]['docs'],
            temp_results[subquery]['docs_expanded']
        )

    return refined_results


def extract_single_doc(
    doc: Dict[str, Any],
    goal: str,
    model: str = 'summary_model'
) -> str:
    """
    对单个文档进行信息抽取，输出结构化的useful_information（三级调用策略）
    
    调用策略：
    1. 一级缓存：检查URL是否命中缓存，命中则直接返回
    2. 二级调用：使用chat_with_model调用summary_model，最多重试3次
    3. 三级兜底：如果二级失败，使用call_model_api_with_messages调用gemini-2.5-flash-lite

    Args:
        doc: 文档字典，包含content, url等字段
        goal: 用户目标/问题
        model: 使用的模型，默认summary_model

    Returns:
        格式化的useful_information字符串
    """
    content = doc.get('content', '') or doc.get('passage', '')
    url = doc.get('url', 'unknown')

    if not content:
        return f"The useful information in {url} for user goal {goal} as follows: \n\nEvidence in page: \nNo content available\n\nSummary: \nEmpty document\n\n"

    # ========== 一级缓存：检查URL缓存 ==========
    cache_key_suffix = f"_extract_{goal[:100]}"  # 区分不同类型的缓存
    if url and url != 'unknown':
        cached_result = get_from_cache(url, f"extract:{goal}")
        if cached_result and "useful_information" in cached_result:
            logger.debug(f"[Extract] 缓存命中: {url[:50]}...")
            return cached_result["useful_information"]

    # 截断过长的内容
    max_content_length = 8000
    if len(content) > max_content_length:
        content = content[:max_content_length]

    # 构建prompt
    user_prompt = EXTRACT_SINGLE_DOC_PROMPT.format(
        webpage_content=content,
        goal=goal
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    # ========== 二级+三级调用：带兜底策略的LLM调用 ==========
    try:
        response_text, call_source = call_llm_with_fallback(
            messages,
            primary_model=model,
            fallback_model=FALLBACK_MODEL,
            max_retries=SUMMARY_MODEL_MAX_RETRIES
        )

        if response_text:
            # 解析JSON响应
            raw = parse_extract_response(response_text)

            # 格式化输出
            useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            useful_information += "Evidence in page: \n" + str(raw.get("evidence", "")) + "\n\n"
            useful_information += "Summary: \n" + str(raw.get("summary", "")) + "\n\n"

            # 保存到缓存
            if url and url != 'unknown':
                save_to_cache(url, f"extract:{goal}", {"useful_information": useful_information})

            return useful_information
        else:
            return f"The useful information in {url} for user goal {goal} as follows: \n\nEvidence in page: \n{content[:500]}\n\nSummary: \nFailed to extract summary\n\n"

    except Exception as e:
        return f"The useful information in {url} for user goal {goal} as follows: \n\nEvidence in page: \n{content[:500]}\n\nSummary: \nExtraction error: {str(e)}\n\n"


def parse_extract_response(response_text: str) -> Dict[str, Any]:
    """
    解析extract_single_doc的JSON响应

    Args:
        response_text: 模型返回的文本

    Returns:
        解析后的字典，包含rationale, evidence, summary
    """
    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        pass

    try:
        result = json_repair.loads(response_text)
        return result
    except Exception:
        pass

    # 尝试从文本中提取JSON
    import re
    json_pattern = r'\{[^{}]*"evidence"[^{}]*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        try:
            result = json_repair.loads(match.group())
            return result
        except Exception:
            pass

    # 尝试提取更复杂的嵌套JSON
    try:
        # 找到第一个{和最后一个}
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
            result = json_repair.loads(json_str)
            return result
    except Exception:
        pass

    # 解析失败，返回原文作为evidence
    return {
        "rationale": "",
        "evidence": response_text,
        "summary": "Parse error - raw response preserved",
        "parse_error": True
    }


def extract_single_doc_raw(
    doc: Dict[str, Any],
    goal: str,
    model: str = 'summary_model'
) -> Dict[str, Any]:
    """
    对单个文档进行信息抽取，返回原始字典结果（三级调用策略）
    
    调用策略：
    1. 一级缓存：检查URL是否命中缓存，命中则直接返回
    2. 二级调用：使用chat_with_model调用summary_model，最多重试3次
    3. 三级兜底：如果二级失败，使用call_model_api_with_messages调用gemini-2.5-flash-lite

    Args:
        doc: 文档字典，包含content, url等字段
        goal: 用户目标/问题
        model: 使用的模型，默认summary_model

    Returns:
        包含rationale, evidence, summary的字典
    """
    content = doc.get('content', '') or doc.get('passage', '')
    url = doc.get('url', 'unknown')

    if not content:
        return {
            "url": url,
            "goal": goal,
            "rationale": "",
            "evidence": "",
            "summary": "Empty document",
            "status": "empty_content"
        }

    # ========== 一级缓存：检查URL缓存 ==========
    if url and url != 'unknown':
        cached_result = get_from_cache(url, f"extract_raw:{goal}")
        if cached_result and "rationale" in cached_result:
            logger.debug(f"[ExtractRaw] 缓存命中: {url[:50]}...")
            return {**cached_result, "status": "cache_hit"}

    # 截断过长的内容
    max_content_length = 8000
    if len(content) > max_content_length:
        content = content[:max_content_length]

    # 构建prompt
    user_prompt = EXTRACT_SINGLE_DOC_PROMPT.format(
        webpage_content=content,
        goal=goal
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    # ========== 二级+三级调用：带兜底策略的LLM调用 ==========
    try:
        response_text, call_source = call_llm_with_fallback(
            messages,
            primary_model=model,
            fallback_model=FALLBACK_MODEL,
            max_retries=SUMMARY_MODEL_MAX_RETRIES
        )

        if response_text:
            raw = parse_extract_response(response_text)
            result = {
                "url": url,
                "goal": goal,
                "rationale": raw.get("rationale", ""),
                "evidence": raw.get("evidence", ""),
                "summary": raw.get("summary", ""),
                "status": f"success_by_{call_source}",
                "raw_response": response_text,
                "call_source": call_source
            }
            # 保存到缓存
            if url and url != 'unknown':
                save_to_cache(url, f"extract_raw:{goal}", {
                    "url": url,
                    "goal": goal,
                    "rationale": raw.get("rationale", ""),
                    "evidence": raw.get("evidence", ""),
                    "summary": raw.get("summary", "")
                })
            return result
        else:
            return {
                "url": url,
                "goal": goal,
                "rationale": "",
                "evidence": content[:500],
                "summary": "Empty response from model",
                "status": "empty_response"
            }

    except Exception as e:
        return {
            "url": url,
            "goal": goal,
            "rationale": "",
            "evidence": content[:500],
            "summary": f"Error: {str(e)}",
            "status": f"error: {str(e)}"
        }


# ============ 实验用函数 ============

# 实验用 Prompt 模板（支持 subqueries）
REFINE_USER_PROMPT_TEMPLATE_WITH_SUBQUERIES = """Extract information from the webpage content that may help answer the user's question.

<instructions>
1. EXTRACTION RULES:
   - Extract ALL potentially useful information, not just what directly matches the question
   - PRESERVE EXACT ORIGINAL WORDING for: dates, numbers, names, titles, quotes, specific facts
   - Do NOT summarize or paraphrase key details - copy them verbatim，Do NOT paraphrase or summarize these details.
   - Include context around key facts (who said what, when, where)
   - Keep up to 300 words if the content is highly relevant

2. WHAT TO INCLUDE:
   - Information directly answering the question
   - Background context that might be useful
   - Related facts mentioned in the same content (they may answer follow-up parts of the question)
   - Specific details: names, dates, numbers, quotes - in their ORIGINAL form

3. If NO relevant information exists:
   - Set relevance to 0
   - Provide a 1-sentence summary of what the content is about
   - Do NOT mention that it doesn't answer the question

4. Relevance scoring (0 to 1):
   - 0.8-1.0: Directly answers or provides key information
   - 0.5-0.7: Contains useful supporting/background information
   - 0.2-0.4: Tangentially related, may provide context
   - 0-0.1: No relevant information

CRITICAL:
- NEVER fabricate information. Extract ONLY from the given content.
- PRESERVE original wording for specific facts to avoid attribution errors.
- Output ONLY valid JSON.
</instructions>

<content>
{text}
</content>

<question>
{query}
</question>
Output:
```json
{{"snippet": "extracted text preserving original wording", "relevance": 0.X}}
```"""


def refine_single_doc_with_subqueries(
    prompt: str,
    subqueries: List[str],
    doc: Dict[str, Any],
    model: str = 'summary_model',
    short_content_threshold: int = SHORT_CONTENT_THRESHOLD
) -> Dict[str, Any]:
    """
    对单个文档进行摘要提取（支持 subqueries，三级调用策略）
    
    调用策略：
    1. 一级缓存：检查URL是否命中缓存，命中则直接返回
    2. 二级调用：使用chat_with_model调用summary_model，最多重试3次
    3. 三级兜底：如果二级失败，使用call_model_api_with_messages调用gemini-2.5-flash-lite

    Args:
        prompt: 用户原始问题
        subqueries: 当前轮次的subquery列表
        doc: 文档字典，包含content, title, url等字段
        model: 使用的模型，默认summary_model
        short_content_threshold: 短内容阈值，低于此长度直接保留原文

    Returns:
        包含原始doc信息和提取结果的字典
    """
    content = doc.get('passage_long', '') or doc.get('content', '')
    url = doc.get('url', '')

    if not content:
        result = {
            **doc,
            "snippet": "",
            "relevance": 0,
            "refine_status": "empty_content"
        }
        # 保存结果
        save_refine_result(prompt, subqueries, doc, result)
        return result

    # ========== 一级缓存：检查URL缓存 ==========
    if url:
        cached_result = get_from_cache(url, prompt)
        if cached_result and "snippet" in cached_result:
            logger.debug(f"[RefineWithSubqueries] 缓存命中: {url[:50]}...")
            result = {
                **doc,
                **cached_result,
                "refine_status": "cache_hit"
            }
            # 即使缓存命中也保存结果用于分析
            save_refine_result(prompt, subqueries, doc, result)
            return result

    # 短内容优化：直接保留原文，不调用模型
    if len(content) <= short_content_threshold:
        result = {
            **doc,
            "snippet": content,
            "relevance": 0.5,
            "refine_status": "short_content_preserved"
        }
        # 保存到缓存
        if url:
            save_to_cache(url, prompt, {"snippet": content, "relevance": 0.5})
        # 保存结果
        save_refine_result(prompt, subqueries, doc, result)
        return result

    # 截断过长的内容
    max_content_length = 8000
    if len(content) > max_content_length:
        content = content[:max_content_length]

    # 构建query：结合prompt和subqueries
    combined_query = f"User Question: {prompt}\nRelated Search Queries: {', '.join(subqueries)}"

    user_prompt = REFINE_USER_PROMPT_TEMPLATE_WITH_SUBQUERIES.format(
        text=content,
        query=combined_query
    )

    messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # ========== 二级+三级调用：带兜底策略的LLM调用 ==========
    try:
        response_text, call_source = call_llm_with_fallback(
            messages,
            primary_model=model,
            fallback_model=FALLBACK_MODEL,
            max_retries=SUMMARY_MODEL_MAX_RETRIES
        )

        if response_text:
            parsed_result = parse_refine_response(response_text)
            result = {
                **doc,
                "snippet": parsed_result.get("snippet", content),
                "relevance": parsed_result.get("relevance", 0.5),
                "refine_status": f"refined_by_{call_source}",
                "call_source": call_source
            }
            # 保存到缓存
            if url:
                save_to_cache(url, prompt, {
                    "snippet": parsed_result.get("snippet", content),
                    "relevance": parsed_result.get("relevance", 0.5)
                })
            # 保存结果
            save_refine_result(prompt, subqueries, doc, result)
            return result
        else:
            result = {
                **doc,
                "snippet": content,
                "relevance": 0.5,
                "refine_status": "empty_response_preserved"
            }
            # 保存结果
            save_refine_result(prompt, subqueries, doc, result)
            return result
    except Exception as e:
        result = {
            **doc,
            "snippet": content,
            "relevance": 0.5,
            "refine_status": f"error_preserved: {str(e)}"
        }
        # 保存结果
        save_refine_result(prompt, subqueries, doc, result)
        return result


def refine_docs_batch(
    prompt: str,
    subqueries: List[str],
    docs: List[Dict[str, Any]],
    model: str = 'summary_model',
    max_workers: int = 10,
    short_content_threshold: int = SHORT_CONTENT_THRESHOLD,
    filter_zero_relevance: bool = False
) -> List[Dict[str, Any]]:
    """
    并发对文档列表进行摘要提取（实验用）

    Args:
        prompt: 用户原始问题
        subqueries: 当前轮次的subquery列表
        docs: 文档列表
        model: 摘要模型
        max_workers: 最大并发数
        short_content_threshold: 短内容阈值
        filter_zero_relevance: 是否过滤relevance=0的文档

    Returns:
        处理后的文档列表
    """
    if not docs:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(refine_single_doc_with_subqueries, prompt, subqueries, doc, model, short_content_threshold): idx
            for idx, doc in enumerate(docs)
        }

        results = [None] * len(docs)
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"文档 {idx} 处理失败: {e}")
                results[idx] = {
                    **docs[idx],
                    "snippet": docs[idx].get('content', ''),
                    "relevance": 0.5,
                    "refine_status": f"executor_error: {str(e)}"
                }

    # 根据配置决定是否过滤relevance=0的文档
    if filter_zero_relevance:
        original_count = len(results)
        results = [doc for doc in results if doc and doc.get('relevance', 0) > 0]
        filtered_count = original_count - len(results)
        if filtered_count > 0:
            # logger.debug(f"[Refine] 过滤掉 {filtered_count} 个relevance=0的文档")
            pass

    return results


def filter_by_relevance(
    refined_docs: List[Dict[str, Any]],
    min_relevance: float = 0.0
) -> List[Dict[str, Any]]:
    """
    根据relevance分数过滤文档

    Args:
        refined_docs: refine后的文档列表
        min_relevance: 最小relevance阈值，默认0（保留所有）

    Returns:
        过滤后的文档列表
    """
    return [
        doc for doc in refined_docs
        if doc.get('relevance', 0) >= min_relevance
    ]


def sort_by_relevance(
    refined_docs: List[Dict[str, Any]],
    descending: bool = True
) -> List[Dict[str, Any]]:
    """
    根据relevance分数排序文档

    Args:
        refined_docs: refine后的文档列表
        descending: 是否降序排列，默认True

    Returns:
        排序后的文档列表
    """
    return sorted(
        refined_docs,
        key=lambda x: x.get('relevance', 0),
        reverse=descending
    )


if __name__ == "__main__":
    # 测试代码
    test_prompt = '在首次提到"两面包夹芝士"的视频中，up主最后提到的干员是什么职业'
    test_subqueries = ['两面包夹芝士首次提及', '两面包夹芝士视频', '两面包夹芝士up主', '两面包夹芝士干员', '两面包夹芝士职业']
    test_docs = [
        {
            "title": "两面包夹芝士是什么梗啥意思",
            "content": '这个梗来源于b站的up主:一只大哈鱼,发的"3000块全没了!各位喝醉了一定要管好自己,我真服了!"的视频。其中的经典句"形成一个两面包夹之势"成为了"两面包夹芝士"的梗开始快速传播。其中也有许多白开水怎么没味、小羊脏了、我永远喜欢森蚺、ocean cat等搞笑梗也被传播开来。',
            "url": "http://m.mnw.cn/keji/internet/2421282.html",
            "date": "2021-05-07"
        },
        {
            "title": "瑞士大孔芝士夹面包的做法",
            "content": "瑞士大孔芝士夹面包的做法_菜谱_豆果美食。法棍切片备用 两片面包夹两片瑞士太孔芝士 黄油小火融化 放入面包煎酥脆 煎至两边金黄即可",
            "url": "http://mwx.douguo.com/recipe/imgs/2517255",
            "date": "2025-11-28"
        }
    ]

    print("=" * 50)
    print("测试 Refine Agent...")
    print("=" * 50)
    print(f"问题: {test_prompt}")
    print(f"子查询: {test_subqueries}")
    print(f"文档数: {len(test_docs)}")

    # 实际运行测试（需要配置好API）
    # refined_docs = call_refine_agent(test_prompt, test_docs)
    # for i, doc in enumerate(refined_docs):
    #     print(f"\n【文档 {i+1}】")
    #     print(f"  标题: {doc.get('title', 'N/A')}")
    #     print(f"  Snippet: {doc.get('snippet', 'N/A')}")
    #     print(f"  Relevance: {doc.get('relevance', 'N/A')}")
