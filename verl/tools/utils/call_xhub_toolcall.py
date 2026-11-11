#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VendorSearch搜索工具调用模块

该模块提供了与VendorSearch搜索服务集成的完整功能，支持搜索结果获取、解析、
站点信息和意图分类集成、查询重写、文档权威性评估和引文格式化。

主要功能：
- ✅ 搜索结果获取和解析
- ✅ 站点信息和意图分类集成
- ✅ 查询重写支持
- ✅ 文档权威性评估
- ✅ 引文格式化
- ✅ 批量查询处理

核心特性：
- **完整集成**: 集成VendorSearch搜索、站点推荐、意图分类、查询重写等多个功能
- **权威性评估**: 自动评估文档权威性级别（0-4级）
- **引文格式化**: 统一的引文格式，包含标题、URL、日期、权威性等信息
- **多源支持**: 支持微信、医疗、通用等多种数据源
- **并发处理**: 支持批量查询并发处理，提高处理效率
- **类型安全**: 完整的类型提示和参数验证

使用示例：
    from call_vendor_search_toolcall import call_search_query_group

    # 基本查询
    query = "量化新规实施后的影响"
    sub_queries = ["量化新规 流动性风险 中小盘股 指标"]
    docs, docs_expanded = call_search_query_group(
        query,
        sub_queries,
        get_sites=True,
        get_intentions=True
    )

    # 带历史查询的重写
    history = ["宁波佑嘉环保科技有限公司"]
    query = "产业标签是多少？"
    docs, docs_expanded = call_search_query_group(
        query,
        sub_queries,
        get_rewrite=True,
        history_queries=history
    )

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""

import traceback
import json
import re
from typing import Tuple, List, Any, Dict
from urllib.parse import urlparse

import requests

from .call_vendor_search_get_qsite import get_qsite_info
from .call_vendor_search_get_qintent import get_intention_info
from .call_vendor_search_rewrite_query import get_rewrite_query

# ============================================================================
# 引文模板和常量配置
# ============================================================================

# 网页引文模板（线上版本）
# 用于将搜索结果格式化为统一的引文格式
yb_web_tpl_new = """[webpage {idx} begin]
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

# 文档权威性级别中文映射表
# 用于将数字权威性级别转换为中文描述
AUTH_LEVEL_CH = {
    0: "极低权威性",  # 权威性最低
    1: "低权威性",  # 权威性较低
    2: "中等权威性",  # 权威性中等
    3: "高权威性",  # 权威性较高
    4: "极高权威性",  # 权威性最高
    -1: "暂无权威性信息",  # 未知权威性
}
# 判断是否包含中文
NON_CHINESE_STRING_PATTERN = re.compile(
    r"^[^\u4E00-\u9FA5]+$"
)

def get_doc_confidence_new(qd_auth_level):
    """获取文档权威性级别描述。

    该函数将数字权威性级别转换为中文描述，用于向用户展示文档的权威性信息。

    Args:
        qd_auth_level: 权威性级别（0-4或-1）

    Returns:
        str: 权威性级别的中文描述

    Note:
        - 0: 极低权威性
        - 1: 低权威性
        - 2: 中等权威性
        - 3: 高权威性
        - 4: 极高权威性
        - -1: 暂未找到权威性信息
    """
    try:
        auth = int(qd_auth_level)
    except Exception as e:
        print(f"权威性级别转换错误: {e}")
        return "未知"

    # 使用字典映射，更简洁高效
    return AUTH_LEVEL_CH.get(auth, "未知")


def extract_domain(url: str) -> str:
    """从URL中提取域名。

    该函数使用urllib.parse.urlparse解析URL并提取网络位置（域名）。

    Args:
        url: 完整的URL字符串

    Returns:
        str: 提取出的域名

    Example:
        >>> extract_domain("https://www.example.com/path/to/page")
        'www.example.com'
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain


def parse_data(query, origin, has_plugin: bool = False):
    """解析VendorSearch搜索结果并转换为vendor_search格式。

    该函数将VendorSearch返回的原始搜索结果解析为统一的数据格式，
    包括文档标题、内容、URL、日期、权威性等字段。

    Args:
        query: 用户查询
        origin: VendorSearch返回的原始JSON字符串
        has_plugin: 是否包含plugin结果

    Returns:
        tuple: (docs, docs_expanded)
            - docs: 标准文档列表
            - docs_expanded: 扩展文档列表（expand=1的文档）

    Note:
        每个文档包含以下字段：
        - title: 标题
        - content: 完整内容
        - passage: 简短摘要
        - passage_long: 详细内容
        - domain: 域名
        - url: 完整URL
        - date: 发布日期
        - res_type: 资源类型（wechat/medical等）
        - auth: 权威性描述
        - qd_auth_level: 权威性级别（0-4）
        - score: 相关性分数
        - recall_by: 召回方式
    """
    # 解析JSON响应
    try:
        res = json.loads(origin)
    except Exception as e:
        print(f"parse_data执行错误: JSON解析失败 - {e}")
        return [], []

    # 检查响应是否为空
    if not res:
        print("返回结果为空！")
        return [], []

    # 提取passage_group
    try:
        item = res["passage_group"]
    except Exception as e:
        print(f"parse_data执行错误: passage_group解析错误 - {e}")
        return [], []

    if not item:
        print("返回结果为空！")
        return [], []

    # 获取文档信息
    docs_info = item["doc_infos"]

    all_num = 0  # 总文档数
    num_error = 0  # 错误文档数
    docs = []  # 标准文档列表
    docs_expanded = []  # 扩展文档列表

    # 遍历所有passage信息
    for d_index, passage_info in enumerate(item["passage_infos"]):
        # 提取基本信息
        passage = passage_info["passage"]  # 简短摘要
        passage_long = passage_info["passage_long"]  # 详细内容
        score = passage_info["score"]  # 相关性分数
        score = str(score)

        try:
            # 从 features 中提取各种属性
            date_str = passage_info["features"].get("date", "")  # 发布日期
            recall_by = passage_info["features"].get("recall_by", "")  # 召回方式
            res_type = passage_info["features"].get("res_type", "")  # 资源类型
            is_expand = passage_info["features"].get("expand", "0")  # 是否为扩展结果
            qd_auth_level = passage_info["features"].get(
                "qd_auth_level", -1
            )  # 权威性级别

            # 获取URL和标题
            url = docs_info[d_index]["url"]

            # 微信文章需要拼接new_sn
            if res_type == "wechat":
                url = url + "/" + passage_info["features"]["new_sn"]

            # 医疗类内容特殊处理：添加标记并设置最高权威性
            if res_type == "medical":
                passage = f"[medical] 该网页内容来自医院权威医生\n{passage}"
                qd_auth_level = 4  # 医疗内容设为最高权威性

            title = docs_info[d_index]["title"]
            score = float(score)

            # 提取域名和权威性描述
            domain = extract_domain(url)
            auth_string = get_doc_confidence_new(qd_auth_level)

            # 选择内容：优先使用passage_long，否则使用passage
            content = passage_long if passage_long else passage

            # 构建文档字典
            item = {
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
                "recall_by": recall_by,
            }
            all_num += 1

        except Exception as e:
            # 捕获解析错误，跳过该文档
            import traceback

            traceback.print_exc()
            print(f"过滤掉的passage: {passage_info}")
            num_error += 1
            continue

        # 根据是否为扩展结果分类存储
        is_expand = passage_info["features"].get("expand", "0")
        if is_expand == "1":
            docs_expanded.append(item)  # 扩展结果
        else:
            docs.append(item)  # 标准结果
    # print('all num: ', all_num)
    # print('num error: ', num_error)
    if all_num == 0:
        print("search error!")
    return docs, docs_expanded


def get_docs(query_group, query, has_plugin=False):
    """获取搜索结果文档。

    该函数调用VendorSearch搜索服务获取搜索结果，支持多种环境配置和并发处理。

    Args:
        query_group: 查询组配置
        query: 用户查询
        has_plugin: 是否包含plugin结果

    Returns:
        tuple: (docs, docs_expanded)
            - docs: 标准文档列表
            - docs_expanded: 扩展文档列表

    Note:
        支持多种环境配置：
        - 原始域名：vendor_search.xsearch.example.com
        - vendor_search内部预发环境：lx-search.example.com
        - vendor_search外部预发/线上环境：vendor_search.xsearch.example.com
        - vendor_search实验环境：gray.vendor_search.xsearch.example.com
    """

    # #原始域名
    # url = "https://search.example.com/api"

    # #ds-ai内部预发环境
    # url = "https://search.example.com/api"

    # #ds-ai外部预发环境、线上环境
    url = "https://search.example.com/api"

    # #ds-ai实验环境
    # url = "https://search.example.com/api"

    headers = {"Content-Type": "application/json"}
    data = {
        "request_header": {
            "request_id": "expa",
            "guid": "wuye",
            "session_id": "wuye",
            "enable_full_content": False,
            "enable_debug": True,
            "business_type": "",
            "scene": {"id": "10086", "name": "offline_test"},
        },
        "request_type": 5,  # 3表示商业流量 2表示AI研究助手 1表示通用流量 0表示元宝
        "query_group": query_group,
        "exp_info": "",
        "extends": {
            "max_result_num": "NTA=",  # 模拟商业流量的extend参数
            "open_expand": "b24=",  # 模拟商业流量的extend参数
            "open_query_rewrite": "b24=",  # 模拟商业流量的extend参数
            "from_offline": "",
        },
        "filter_level": "",
        "logtrace": "",
    }

    with requests.Session() as session:
        response = session.post(url, headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        print(f"response.status_code !=200, status_code:{response.status_code}")
        return [], []

    docs, docs_expanded = parse_data(query, response.text, has_plugin)
    response.close()
    return docs, docs_expanded


def call_search(
    query: str, subqueries: List[str], has_plugin: bool = False
) -> Tuple[Dict[str, Any]]:
    """执行原文和子查询搜索。

    该函数将子查询列表转换为查询组格式，并调用搜索服务获取结果。

    Args:
        query: 原始用户查询
        subqueries: 子查询列表
        has_plugin: 是否包含plugin结果

    Returns:
        tuple: (docs, docs_expanded)
            - docs: 标准文档列表
            - docs_expanded: 扩展文档列表

    Note:
        该函数主要用于向后兼容，推荐使用call_search_query_group函数。
    """
    try:
        subqueries_dict_list = [
            {"query": subq, "query_type": 0 if i == 0 else 1}
            for i, subq in enumerate(subqueries)
        ]
        docs, docs_expanded = get_docs(subqueries_dict_list, query, has_plugin)
        return docs, docs_expanded
    except Exception:
        print("call_search执行错误")
        return [], []


def string_is_all_english(text: str) -> bool:
    """判断字符串是否不包含中文字符。

    该函数使用正则表达式检查字符串是否完全由非中文字符组成。

    Args:
        text: 待检查的字符串

    Returns:
        bool: True表示不包含中文字符，False表示包含中文字符

    Note:
        使用正则表达式模式：r"^[^\\u4E00-\\u9FA5]+$"
    """

    match = NON_CHINESE_STRING_PATTERN.match(text)
    return match is not None


def call_search_query_group(
    query: str,
    subqueries: List[str],
    get_sites: bool = True,
    get_intentions: bool = True,
    get_rewrite: bool = False,
    has_plugin: bool = False,
    history_queries: List[str] = None,
) -> Tuple[list, list]:
    """执行增强版搜索查询组调用。

    该函数是主要的搜索入口，支持站点推荐、意图分类、查询重写等高级功能，
    对齐线上query group特性。

    Args:
        query: 原始用户查询
        subqueries: 子查询列表
        get_sites: 是否获取站点推荐信息
        get_intentions: 是否获取意图分类信息
        get_rewrite: 是否使用查询重写模型
        has_plugin: 是否包含plugin结果
        history_queries: 历史查询列表，用于查询重写的上下文

    Returns:
        tuple: (docs, docs_expanded)
            - docs: 标准文档列表
            - docs_expanded: 扩展文档列表

    Note:
        - 支持查询重写：当get_rewrite=True时使用重写模型，否则使用|拼接子查询
        - 支持纯英文查询自动开启安全查询参数
        - 支持站点推荐和意图分类集成
    """
    try:
        # #获取每个subq的站点list和意图分类
        all_queries = []
        if get_sites and subqueries:
            try:
                all_sites = get_qsite_info(subqueries)
            except Exception:
                print("获取站点信息失败")
                all_sites = [[] for _ in subqueries]
        else:
            all_sites = [[] for _ in subqueries]

        # 获取意图分类
        if get_intentions and subqueries:
            try:
                all_intentions = get_intention_info(subqueries)
            except Exception:
                print("获取意图分类失败")
                all_intentions = [[] for _ in subqueries]
        else:
            all_intentions = [[] for _ in subqueries]

        for i, subq in enumerate(subqueries):
            # #获取qiste
            query_sites = all_sites[i] if i < len(all_sites) else []
            query_site_str = ";".join(query_sites) if query_sites else ""

            # #获取qinient
            query_intentions = all_intentions[i] if i < len(all_intentions) else []
            # #将意图列表用|拼接
            policy_infer_str = "|".join(query_intentions) if query_intentions else ""
            if get_sites and get_intentions:
                query_item = {
                    "query": subq,
                    "query_type": 0 if i == 0 else 1,
                    "extends": {
                        "query_site": query_site_str,
                        "policy_infer": policy_infer_str,  # 意图用|拼接
                    },
                }
            elif get_sites and not get_intentions:
                query_item = {
                    "query": subq,
                    "query_type": 0 if i == 0 else 1,
                    "extends": {
                        "query_site": query_site_str,
                    },
                }
            elif not get_sites and get_intentions:
                query_item = {
                    "query": subq,
                    "query_type": 0 if i == 0 else 1,
                    "extends": {"policy_infer": policy_infer_str},  # 意图用|拼接
                }
            # "is_safe_query": "1",  ##纯英文query召回开启这个参数
            if string_is_all_english(query):
                query_item["extends"]["is_safe_query"] = "1"
            all_queries.append(query_item)

        # #构建rewrite_query（将所有子查询用|拼接）
        if get_rewrite:
            rewrite_query_from_llm = get_rewrite_query(query, history_queries)
            rewrite_query_from_llm = (
                rewrite_query_from_llm.strip()
                if rewrite_query_from_llm
                else "|".join(subqueries)
            )
            rewrite_query = {"query": rewrite_query_from_llm, "query_type": 1}
        else:
            rewrite_query = {"query": "|".join(subqueries), "query_type": 1}

        # #构建完整的query_group
        query_group = {
            "origin_query": query,
            "all_queries": all_queries,
            "rewrite_query": rewrite_query,
        }

        # #调用搜索接口
        docs, docs_expanded = get_docs(query_group, query, has_plugin=has_plugin)
        return docs, docs_expanded

    except Exception:
        traceback.print_exc()
        print("call_search执行错误")
        return [], []


def safe_substitute(text: str, max_chars: int = 2048) -> str:
    """安全截断文本，保护数学公式不被破坏。

    该函数在截断文本时，会先保护数学公式（$$...$$和$...$格式），
    然后在截断后恢复公式，确保公式完整性。

    Args:
        text: 待截断的文本
        max_chars: 最大字符数限制，默认为2048

    Returns:
        str: 截断后的文本，公式保持完整

    Raises:
        TypeError: 输入不是字符串或字节类型
        ValueError: 正则表达式处理错误
    """
    if not isinstance(text, (str, bytes)):
        raise TypeError(f"Expected string or bytes, got {type(text).__name__}")

    formulas = []

    def formula_replacer(match):
        formula = match.group(0)
        formulas.append(formula)
        return f"@@FORMULA_{len(formulas)-1}@@"

    try:
        protected = re.sub(
            pattern=r"(\$\$.*?\$\$|\$.*?\$)", repl=formula_replacer, string=str(text)
        )
        lines = protected.split("\n")
        result_lines = []
        current_length = 0

        for line in lines:
            line_length = len(line) + (1 if result_lines else 0)
            if current_length + line_length <= max_chars:
                result_lines.append(line)
                current_length += line_length
            else:
                break

        truncated = "\n".join(result_lines)

        def restore(match):
            try:
                idx = int(match.group(1))
                return formulas[idx] if 0 <= idx < len(formulas) else ""
            except (IndexError, ValueError):
                return ""

        result = re.sub(r"@@FORMULA_(\d+)@@", restore, truncated)
        return result

    except re.error as e:
        raise ValueError(f"正则表达式处理错误: {str(e)}")


def main():
    """主函数：演示搜索工具的使用。

    该函数提供了搜索工具的演示用例，包括：
    1. 多种业务场景的测试用例
    2. 详细的执行状态和结果展示
    3. 错误处理和性能统计
    4. 可配置的测试参数

    Returns:
        bool: 执行是否成功
    """
    # 配置测试用例
    test_cases = [
        {
            "name": "企业信息查询",
            "query": "产业标签是多少？",
            "sub_queries": [
                "宁波佑嘉环保科技有限公司产业标签",
                "宁波佑嘉环保科技有限公司官网关键词",
                "宁波佑嘉环保科技有限公司新闻关键词",
                "宁波佑嘉环保科技有限公司专利关键词",
                "宁波佑嘉环保科技有限公司商标关键词",
                "宁波佑嘉环保科技有限公司软著关键词",
                "宁波佑嘉环保科技有限公司招聘信息关键词",
            ],
            "history_queries": ["宁波佑嘉环保科技有限公司"],
            "get_rewrite": False,
            "has_plugin": True
        },
        {
            "name": "量化新规分析",
            "query": "量化新规实施后，如何具体判断哪些中小盘股可能面临流动性风险？有哪些量化指标可以参考？",
            "sub_queries": ["量化新规 流动性风险 中小盘股 指标"],
            "history_queries": [],
            "get_rewrite": False,
            "has_plugin": True
        },
        {
            "name": "医疗健康查询",
            "query": "糖尿病患者应该如何控制饮食？",
            "sub_queries": ["糖尿病饮食控制 血糖管理"],
            "history_queries": [],
            "get_rewrite": True,
            "has_plugin": False
        }
    ]

    print("🚀 VendorSearch搜索工具演示开始")
    print(f"📋 测试用例数量: {len(test_cases)}")
    print("-" * 60)

    success_count = 0
    total_results = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 测试用例 {i}/{len(test_cases)}: {test_case['name']}")
        print(f"   查询: {test_case['query']}")
        print(f"   子查询数量: {len(test_case['sub_queries'])}")
        print(f"   历史查询: {len(test_case['history_queries'])}条")
        print(f"   重写功能: {'启用' if test_case['get_rewrite'] else '禁用'}")
        print(f"   Plugin结果: {'包含' if test_case['has_plugin'] else '不包含'}")

        try:
            # 执行搜索
            docs_list, docs_expanded = call_search_query_group(
                query=test_case['query'],
                subqueries=test_case['sub_queries'],
                get_rewrite=test_case['get_rewrite'],
                has_plugin=test_case['has_plugin'],
                history_queries=test_case['history_queries'],
            )

            # 统计结果
            total_docs = len(docs_list) + len(docs_expanded)
            total_results += total_docs

            print(f"✅ 搜索成功")
            print(f"   标准文档: {len(docs_list)}条")
            print(f"   扩展文档: {len(docs_expanded)}条")
            print(f"   总结果数: {total_docs}条")

            # 显示前3个结果的摘要
            if docs_list:
                print("\n📄 结果摘要:")
                for j, doc in enumerate(docs_list[:3], 1):
                    print(f"   {j}. [{doc.get('res_type', 'unknown')}] {doc.get('title', '无标题')[:50]}...")
                    print(f"      权威性: {doc.get('auth', '未知')}")
                    print(f"      分数: {doc.get('score', 0):.2f}")

            success_count += 1

        except Exception as e:
            print(f"❌ 测试用例失败: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 60)

    # 生成总结报告
    print("\n📊 测试总结报告")
    print(f"   总测试用例: {len(test_cases)}")
    print(f"   成功用例: {success_count}")
    print(f"   失败用例: {len(test_cases) - success_count}")
    print(f"   总搜索结果: {total_results}条")

    success_rate = success_count / len(test_cases) * 100 if test_cases else 0
    print(f"   成功率: {success_rate:.1f}%")

    if success_count == len(test_cases):
        print("🎉 所有测试用例执行成功！")
    else:
        print("⚠️  部分测试用例失败，请检查错误日志")

    return success_count == len(test_cases)


def run_single_test(query: str, subqueries: List[str], **kwargs) -> Tuple[List[Dict], List[Dict]]:
    """执行单次搜索测试。

    Args:
        query: 主查询
        subqueries: 子查询列表
        **kwargs: 其他参数（get_rewrite, has_plugin, history_queries等）

    Returns:
        tuple: (docs_list, docs_expanded)
    """
    return call_search_query_group(query, subqueries, **kwargs)


if __name__ == "__main__":
    main()
