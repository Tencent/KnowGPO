#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
（匿名模型）API意图分类工具

该模块提供了调用（匿名模型）API进行意图分类的完整功能，支持批量查询和并发处理，
能够将用户查询自动分类到政策、法律、医疗健康、教育、其他等类别。

主要功能：
- ✅ 意图分类API调用
- ✅ 批量查询并发处理
- ✅ 结果解析和格式化
- ✅ 错误处理和重试机制
- ✅ 测试用例和示例
- ✅ 多模型支持

核心特性：
- **并发处理**: 支持多线程并发处理批量查询
- **智能解析**: 自动解析API返回结果，支持JSON和文本格式
- **重试机制**: 内置指数退避重试逻辑，提高调用成功率
- **类型安全**: 完整的类型提示和参数验证
- **易于集成**: 提供简洁的API接口，易于集成到其他系统
- **多分类支持**: 支持政策、法律、医疗健康、教育等多种意图分类

使用示例：
    from call_vendor_search_get_qintent import get_intention_info

    # 单个查询
    query = "劳动合同试用期解除条件"
    intentions = get_intention_info(query)
    print(f"意图分类: {intentions}")

    # 批量查询
    queries = ["劳动合同试用期解除条件", "生育津贴申领依据"]
    all_intentions = get_intention_info(queries, max_workers=5)
    for query, intentions in zip(queries, all_intentions):
        print(f"查询: {query} -> 意图: {intentions}")

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""

from typing import List, Union, Dict
import uuid
import time
import json
import re
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



def get_intention_info(
    sub_queries: Union[str, List[str]],
    max_workers: int = 10,
    model: str = "intention_0.6b_911"
) -> List[List[str]]:
    """
    获取查询对应的意图分类

    该函数通过调用（匿名模型）API对用户查询进行意图分类，支持单个查询和批量查询。
    使用多线程并发处理提高效率，自动处理API调用错误和结果解析。

    Args:
        sub_queries (Union[str, List[str]]): 查询列表或单个查询字符串
            - 单个查询: "劳动合同试用期解除条件"
            - 批量查询: ["劳动合同试用期解除条件", "生育津贴申领依据"]
        max_workers (int, optional): 并发线程数，默认10
            - 建议根据API限制和网络状况调整
            - 过高的并发可能导致API限流
        model (str, optional): 使用的模型名称，默认 "intention_0.6b_911"
            - 支持不同的意图分类模型版本

    Returns:
        List[List[str]]: 每个查询对应的意图分类列表
            - 单个查询返回: [["法律"]]
            - 批量查询返回: [["法律"], ["政策", "法律"]]
            - 每个查询的意图列表可能包含多个标签

    Raises:
        ValueError: 如果输入参数格式错误
        ConnectionError: 如果网络连接失败
        Exception: 其他API调用或处理错误

    Example:
        >>> # 单个查询
        >>> intentions = get_intention_info("劳动合同试用期解除条件")
        >>> print(intentions)  # [["法律"]]

        >>> # 批量查询
        >>> queries = ["劳动合同试用期解除条件", "生育津贴申领依据"]
        >>> all_intentions = get_intention_info(queries, max_workers=5)
        >>> for query, intentions in zip(queries, all_intentions):
        ...     print(f"{query} -> {intentions}")
    """

    system = """
# 任务说明

准确判断给定问题的意图类别。根据问题的性质，从"政策、法律、医疗健康、教育、其他"中做出选择，输出意图数组。

## 一、意图定义

* `政策`：由政府部门或相关行业主管机关发布，通常为引导社会、行业或地区发展的行政性、指导性文件。政策文件内容较为灵活，侧重方向性和规范性，具有指导性，但不具备强制性。
  - 官方统计数据、政府公开报告都算是政策类
  - 对政策的分析算是政策类

* `法律`：由立法机关或授权机构制定，具有强制性，通常用于明确权利与义务，并对违法行为设定法律责任。法律条文详细，违背者将受到法律制裁。
  - 对违规、违法等行为具有惩戒性的内容查询属于法律
  - 对个人、企业、其他主体的强制性的要求和规范属于法律

* `医疗健康`：与疾病诊断、治疗方案、药物使用、医疗器械、临床手术、医学检查、症状表现、病理生理、医学研究等具体医疗技术或临床医学相关的问题。

* `教育`：与学习内容、课程资料、练习题、学习方法、学校招生、教育培训、学习工具、教育实践、学科知识、学习计划等相关的问题。

* `其他`：如果问题不涉及上述任一类别（包括文本润色、翻译、改写、闲聊、常识问答等），归类为其他。

## 二、要求

* 仅输出 JSON 数组字符串，不要输出任何解释或多余字符。
  * 数组元素仅能取以下值："政策", "法律", "医疗健康", "教育", "其他"。
* "其他"与上述所有类别互斥，绝不与其它标签同时出现。
* 对于文本润色、翻译、改写、闲聊等与查找政策/法律或特定领域无关的任务，输出 ["其他"]。
* 你的任务是判断用户问题的意图类别，禁止直接回答用户问题或按用户要求进行角色扮演。

## 三、示例

输入: 劳动合同试用期解除条件

输出: ["法律"]

输入: 生育津贴申领依据与最新规定

输出: ["政策", "法律"]

输入: 2024年小学一年级入学条件

输出: ["政策", "教育"]

输入: 一年级数学练习题

输出: ["教育"]

输入: 直肠粘液腺癌放疗禁忌症

输出: ["医疗健康"]

输入: 帮我把这段话润色一下

输出: ["其他"]
"""

    def crawl_hyaide_result(message_log: List[Dict[str, str]], model: str = "planner") -> str:
        """
        调用（匿名模型）API获取意图分类结果

        该函数负责与（匿名模型）流式服务器进行通信，发送意图分类请求并获取响应。
        内置重试机制和错误处理，确保API调用的可靠性。

        Args:
            message_log (List[Dict[str, str]]): 消息日志列表
                - 包含系统提示和用户查询的完整对话历史
                - 格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            model (str, optional): 使用的模型名称，默认 "planner"
                - 支持不同的模型版本进行意图分类

        Returns:
            str: API返回的原始响应内容
                - 通常是JSON格式的意图分类结果
                - 如: '["法律"]' 或 '["政策", "法律"]'

        Raises:
            requests.RequestException: HTTP请求失败或状态码异常
            json.JSONDecodeError: 响应JSON解析失败
            Exception: 其他API调用错误

        Note:
            - 使用指数退避策略进行重试
            - 最大重试次数为3次
            - 每次重试间隔时间递增
        """
        # API端点配置
        url = os.getenv("KNOWGPO_LEGACY_QINTENT_URL", "https://example.invalid/legacy-api")
        wsid = "10143"  # 服务组标识
        enable_stream = False  # 禁用流式响应，适合意图分类场景

        # 请求头配置
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("KNOWGPO_SEARCH_API_KEY", ""),
            "Wsid": wsid,
        }

        # 请求数据配置
        req_data = {
            "model": model,
            "query_id": "test_query_id_" + str(uuid.uuid4()),  # 唯一请求标识
            "messages": message_log,
            "temperature": 0.1,        # 低温度确保输出稳定性
            "top_p": 0.8,              # 核采样参数
            "top_k": 20,               # Top-K采样参数
            "repetition_penalty": 1,    # 重复惩罚系数
            "output_seq_len": 64,       # 输出序列长度限制
            "max_input_seq_len": 10240,  # 最大输入序列长度
            "openai_infer": True,       # 启用OpenAI兼容模式
            "stream": enable_stream,    # 流式响应控制
        }

        # 重试配置
        max_retries = 3     # 最大重试次数
        backoff = 0.5       # 初始退避时间（秒）
        last_exc = None     # 记录最后一次异常

        # 重试循环
        for attempt in range(max_retries):
            try:
                # 发送HTTP POST请求
                response = requests.post(url, json=req_data, headers=headers, stream=False)

                # 检查HTTP状态码
                if response.status_code != 200:
                    raise requests.RequestException(f"HTTP状态码异常: {response.status_code}")

                # 解析JSON响应
                data = json.loads(response.text)

                # 验证响应结构
                if "choices" not in data:
                    raise ValueError("API响应缺少choices字段")

                choices = data["choices"]
                if not choices:
                    raise ValueError("API响应choices字段为空")

                # 提取内容
                choice = choices[0]
                if "message" not in choice or "content" not in choice["message"]:
                    raise ValueError("API响应缺少message.content字段")

                content = choice["message"]["content"]

                # logger.debug(f"[DEBUG]API调用成功，尝试次数: {attempt + 1}")
                return content

            except Exception as e:
                last_exc = e

                # 如果不是最后一次尝试，进行重试
                if attempt < max_retries - 1:
                    wait_time = backoff * (2 ** attempt)  # 指数退避
                    logger.warning(f"[WARNING]API调用失败，{wait_time}秒后重试... (错误: {str(e)})")
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败，抛出异常
                    logger.error(f"[ERROR]API调用失败，已达到最大重试次数: {str(last_exc)}")
                    raise last_exc

    def remove_think_tag(content: str) -> str:
        """
        去除思考标签和内部思考过程

        该函数用于清理API返回的响应内容，移除模型内部思考过程标记。
        支持正则表达式匹配和多种格式的思考标签清理。

        Args:
            content (str): 原始响应内容，可能包含思考标签
                - 如: "思考过程...["法律"]"

        Returns:
            str: 清理后的纯净内容
                - 如: "["法律"]"

        Example:
            >>> content = "这是一个思考过程["法律"]"
            >>> clean_content = remove_think_tag(content)
            >>> print(clean_content)  # "["法律"]"
        """
        if not content:
            return ""

        # 使用正则表达式移除思考标签
        # 匹配格式: <think>...</think>
        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        # 移除多余空白字符
        cleaned_content = cleaned_content.strip()

        # logger.debug(f"[DEBUG]思考标签清理: {len(content)} -> {len(cleaned_content)} 字符")
        return cleaned_content

    def parse_intention_result(predict_text: str) -> List[str]:
        """
        解析意图分类结果

        该函数支持多种格式的意图分类结果解析，包括:
        - JSON数组格式: ["法律"]
        - Python列表格式: ['政策', '法律']
        - 文本关键词匹配: 包含"法律"等关键词

        Args:
            predict_text (str): 预测文本内容
                - 可能格式: ["法律"], ['政策', '法律'], "法律"等

        Returns:
            List[str]: 解析出的意图标签列表
                - 如: ["法律"], ["政策", "法律"], ["其他"]

        Raises:
            ValueError: 如果解析过程中发生严重错误

        Note:
            - 优先使用JSON解析，安全性最高
            - 支持eval解析作为备选方案
            - 最后使用关键词匹配作为兜底策略
            - 所有解析失败时返回["其他"]
        """
        # 1. 输入验证和预处理
        if not predict_text or predict_text.strip() == "":
            logger.warning("[WARNING]预测文本为空，返回默认分类[其他]")
            return ["其他"]

        predict_text = str(predict_text).strip()
        # logger.debug(f"[DEBUG]开始解析意图结果: {predict_text[:100]}...")
        # 2. 优先尝试JSON解析（最安全）
        try:
            intentions = json.loads(predict_text)
            if isinstance(intentions, list):
                # 验证列表元素都是字符串
                if all(isinstance(item, str) for item in intentions):
                    # logger.debug(f"[DEBUG]JSON解析成功: {intentions}")
                    return intentions
                else:
                    logger.warning("[WARNING]JSON解析结果包含非字符串元素")
        except json.JSONDecodeError as e:
            logger.exception(f"[ERROR]JSON解析失败: {e}")
        except Exception as e:
            logger.exception(f"[ERROR]JSON解析异常: {e}")

        # 3. 备选方案：eval解析（注意安全性）
        try:
            # 安全检查：只允许包含字母、数字、引号、逗号、方括号和空格
            safe_pattern = r'^[\[\]"\'\w\s,]+$'
            if re.match(safe_pattern, predict_text):
                intentions = eval(predict_text)
                if isinstance(intentions, list):
                    # 验证列表元素都是字符串
                    if all(isinstance(item, str) for item in intentions):
                        # logger.debug(f"[DEBUG]eval解析成功: {intentions}")
                        return intentions
                    else:
                        logger.warning("[WARNING]eval解析结果包含非字符串元素")
        except Exception as e:
            logger.exception(f"[ERROR]eval解析失败: {e}")

        # 4. 兜底方案：关键词匹配
        intention_keywords = ["政策", "法律", "医疗健康", "教育", "其他"]
        found_intentions = []

        for keyword in intention_keywords:
            if keyword in predict_text:
                found_intentions.append(keyword)
                # logger.debug(f"[DEBUG]关键词匹配: {keyword}")

        if found_intentions:
            # 检查"其他"标签的互斥性
            if "其他" in found_intentions and len(found_intentions) > 1:
                # "其他"与其他标签互斥，只保留其他标签
                found_intentions = ["其他"]
                # logger.debug('[DEBUG]检测到"其他"标签与其他标签共存，应用互斥规则')

            # logger.debug(f"[DEBUG]关键词匹配成功: {found_intentions}")
            return found_intentions

        # 5. 所有解析方法都失败
        logger.error(f"[ERROR]无法解析的意图结果: {predict_text}")
        # logger.debug("[DEBUG]返回默认分类[其他]")
        return ["其他"]

    def process_single_query(query: str) -> List[str]:
        """
        处理单个查询的意图分类

        该函数负责单个查询的完整处理流程，包括：
        - 查询格式标准化
        - API调用和响应处理
        - 结果解析和清理
        - 错误处理和降级

        Args:
            query (str): 用户查询字符串
                - 如: "劳动合同试用期解除条件"

        Returns:
            List[str]: 意图分类结果列表
                - 如: ["法律"], ["政策", "法律"], ["其他"]
                - 错误时返回默认分类["其他"]

        Example:
            >>> intentions = process_single_query("劳动合同试用期解除条件")
            >>> print(intentions)  # ["法律"]
        """
        try:
            # 1. 查询格式标准化
            if not query.endswith(" /no_think"):
                query = query + " /no_think"
            # logger.debug(f"[DEBUG]处理查询: {query}")

            # 2. 构建消息日志
            message_log = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]

            # 3. 调用API获取响应
            # logger.debug("[DEBUG]开始调用（匿名模型）API...")
            response = crawl_hyaide_result(message_log, model)
            # logger.debug(f"[DEBUG]API响应长度: {len(response)} 字符")

            # 4. 清理响应内容
            cleaned_response = remove_think_tag(response)
            # logger.debug(f"[DEBUG]清理后响应: {cleaned_response[:100]}...")

            # 5. 解析意图分类结果
            intentions = parse_intention_result(cleaned_response)
            # logger.debug(f"[DEBUG]查询处理完成: {query} -> {intentions}")

            return intentions

        except Exception as e:
            # 6. 错误处理和降级
            logger.exception(f"[ERROR]处理查询 '{query}' 时发生错误: {e}")
            logger.error("[ERROR]返回默认分类[其他]作为降级方案")
            return ["其他"]

    # =============================================================================
    # 主处理逻辑
    # =============================================================================

    # 1. 输入参数验证和标准化
    if isinstance(sub_queries, str):
        # 单个查询转换为列表
        sub_queries = [sub_queries]
        # logger.debug("[DEBUG]检测到单个查询，已转换为列表格式")

    if not isinstance(sub_queries, list) or len(sub_queries) == 0:
        raise ValueError("sub_queries参数必须是非空列表或字符串")

    # logger.debug(f"[DEBUG]开始处理 {len(sub_queries)} 个查询，并发数: {max_workers}")

    # 2. 初始化结果容器
    results = [[] for _ in range(len(sub_queries))]

    # 3. 使用线程池进行并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # logger.debug("[DEBUG]启动线程池执行器...")

        # 3.1 提交所有任务到线程池
        future_to_index = {}
        for i, query in enumerate(sub_queries):
            future = executor.submit(process_single_query, query)
            future_to_index[future] = i
            # logger.debug(f"[DEBUG]已提交查询 {i+1}/{len(sub_queries)}: {query[:50]}...")

        # 3.2 收集和处理结果
        completed_count = 0
        total_count = len(future_to_index)

        # logger.debug(f"[DEBUG]开始收集 {total_count} 个任务的结果...")

        for future in as_completed(future_to_index):
            try:
                # 获取任务结果
                intentions = future.result()
                original_index = future_to_index[future]
                results[original_index] = intentions

                completed_count += 1
                # logger.debug(f"[DEBUG]任务完成进度: {completed_count}/{total_count}")
                
            except Exception as e:
                # 处理任务执行异常
                original_index = future_to_index[future]
                error_query = sub_queries[original_index]
                logger.exception(f"[ERROR]任务 {original_index} 执行失败: {e}")
                logger.error(f"[ERROR]失败查询: {error_query}")

                # 设置默认结果
                results[original_index] = ["其他"]
                completed_count += 1

        # logger.debug(f"[DEBUG]所有任务处理完成，成功: {completed_count}/{total_count}")

    return results

# =============================================================================
# 测试和演示代码
# =============================================================================


def test_single_query() -> None:
    """
    测试单个查询的意图分类

    演示如何对单个查询进行意图分类，展示基本使用方式。
    """
    print("=" * 60)
    print("单个查询测试")
    print("=" * 60)

    # 测试查询
    query = "劳动合同试用期解除条件"
    print(f"[测试]查询内容: {query}")

    try:
        # 调用意图分类函数
        print("[测试]开始意图分类...")
        intentions = get_intention_info(query)

        # 显示结果
        print(f"[结果]意图分类: {intentions[0]}")
        print("✓ 单个查询测试成功")

    except Exception as e:
        print(f"✗ 单个查询测试失败: {e}")


def test_batch_queries() -> None:
    """
    测试批量查询的意图分类

    演示如何对多个查询进行批量意图分类，展示并发处理能力。
    """
    print("\n" + "=" * 60)
    print("批量查询测试")
    print("=" * 60)

    # 测试查询列表（覆盖所有意图类别）
    queries = [
        "劳动合同试用期解除条件",                    # 法律
        "生育津贴申领依据与最新规定",                # 政策、法律
        "2024年小学一年级入学条件",                 # 政策、教育
        "一年级数学练习题",                         # 教育
        "直肠粘液腺癌放疗禁忌症",                   # 医疗健康
        "帮我把这段话润色一下",                     # 其他
        "公司注册资本变更流程",                     # 法律、政策
        "个人所得税专项附加扣除标准",                # 政策
        "高血压患者饮食注意事项",                   # 医疗健康
        "初中物理实验操作规范"                      # 教育
    ]

    print(f"[测试]批量查询数量: {len(queries)}")
    print("[测试]查询列表:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")

    try:
        # 调用批量意图分类函数
        print("\n[测试]开始批量意图分类...")
        all_intentions = get_intention_info(queries, max_workers=5)

        # 显示结果
        print("\n[结果]批量查询分类结果:")
        print("-" * 60)
        for i, (query, intentions) in enumerate(zip(queries, all_intentions), 1):
            print(f"{i:2d}. {query:<30} -> {intentions}")

        # 统计结果
        category_count = {}
        for intentions in all_intentions:
            for category in intentions:
                category_count[category] = category_count.get(category, 0) + 1

        print("-" * 60)
        print("[统计]意图分类分布:")
        for category, count in sorted(category_count.items()):
            print(f"  {category}: {count} 次")

        print("✓ 批量查询测试成功")

    except Exception as e:
        print(f"✗ 批量查询测试失败: {e}")


def test_performance() -> None:
    """
    测试性能表现

    演示不同并发数下的性能表现，帮助用户选择合适的并发参数。
    """
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)

    # 测试数据
    test_queries = [
        "劳动合同试用期解除条件",
        "生育津贴申领依据",
        "小学入学条件",
        "数学练习题",
        "医疗禁忌症"
    ]

    # 测试不同并发数
    concurrency_levels = [1, 3, 5, 10]

    print("[测试]性能测试配置:")
    print(f"  查询数量: {len(test_queries)}")
    print(f"  并发级别: {concurrency_levels}")

    for max_workers in concurrency_levels:
        print(f"\n[测试]测试并发数: {max_workers}")

        try:
            start_time = time.time()

            # 执行测试
            results = get_intention_info(test_queries, max_workers=max_workers)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # 计算性能指标
            avg_time_per_query = elapsed_time / len(test_queries)
            queries_per_second = len(test_queries) / elapsed_time

            print("[结果]性能指标:")
            print(f"  总耗时: {elapsed_time:.2f} 秒")
            print(f"  平均每个查询: {avg_time_per_query:.2f} 秒")
            print(f"  查询处理速度: {queries_per_second:.2f} 个/秒")

            # 验证结果
            success_count = sum(1 for r in results if r != ["其他"])
            print(f"  成功分类: {success_count}/{len(test_queries)}")

        except Exception as e:
            print(f"✗ 并发数 {max_workers} 测试失败: {e}")


if __name__ == "__main__":
    """
    主程序入口

    执行意图分类工具的测试和演示功能。
    """
    print("🎯 （匿名模型）API意图分类工具测试套件")
    print("=" * 60)

    # 执行测试用例
    try:
        # 1. 单个查询测试
        test_single_query()

        # 2. 批量查询测试
        test_batch_queries()

        # 3. 性能测试（可选，注释掉以加快测试）
        # test_performance()

        print("\n" + "=" * 60)
        print("🎉 所有测试执行完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[INFO]测试被用户中断")
    except Exception as e:
        print(f"\n[ERROR]测试执行失败: {e}")
        import traceback
        traceback.print_exc()
