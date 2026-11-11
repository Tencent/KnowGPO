#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
（匿名模型）API生成式奖励评分工具

该模块提供了基于（匿名模型）API的生成式奖励评分功能，支持流式和非流式响应处理，
能够对模型生成的序列进行质量评估和评分。

主要功能：
- ✅ 文本输出提取和格式化
- ✅ 流式响应处理
- ✅ 非流式响应处理
- ✅ API异步查询和重试机制
- ✅ 错误处理和降级策略

核心特性：
- **多格式支持**: 支持JSON、文本等多种响应格式
- **流式处理**: 实时处理流式API响应
- **异步查询**: 支持并发API调用
- **重试机制**: 内置指数退避重试策略
- **错误恢复**: 完善的错误处理和降级机制

使用示例：
    from api_generative import prime_query_openai_async, extract_output

    # 配置API参数
    config = {
        'url': 'http://api.example.com',
        'model': 'your-model',
        'wsid': 'your-wsid',
        'api_key': 'your-token',
        'stream': False,
        'temperature': 0.1,
        'max_retries': 3
    }

    # 查询序列
    sequence = "模型生成的文本序列"
    response = prime_query_openai_async(config, sequence)

    # 提取评分结果
    score = extract_output(response.text)
    print(f"评分结果: {score}")

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""
import re
import asyncio
import json
import logging
import traceback
from typing import List, Any, Optional, Dict, Generator

import torch
import requests
from transformers import AutoTokenizer

from .vendor_train_api_chat_completion import VendorTrainOpenAPI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_output(solution_text: str) -> Optional[str]:
    """
    从解决方案文本中提取格式化输出

    该函数使用正则表达式匹配文本中的格式化标记（如\\bold{}），
    提取最后一个匹配项的内容作为输出结果。

    Args:
        solution_text (str): 包含格式化标记的解决方案文本
            - 如: "答案是\\bold{42}" 或 "最终结果: \\bold{正确}"

    Returns:
        Optional[str]: 提取的输出内容，如果没有匹配则返回None
            - 如: "42", "正确"

    Raises:
        TypeError: 如果输入参数不是字符串

    Example:
        >>> text = "根据计算，答案是\\bold{42}"
        >>> result = extract_output(text)
        >>> print(result)  # "42"

        >>> text = "没有格式化标记的文本"
        >>> result = extract_output(text)
        >>> print(result)  # None
    """
    # 参数验证
    if not isinstance(solution_text, str):
        raise TypeError(f"solution_text必须是字符串，实际类型: {type(solution_text)}")

    if not solution_text.strip():
        print("[WARNING]输入文本为空，无法提取输出")
        return None

    # 正则表达式模式：匹配\\bold{}格式的内容
    boxed_pattern = r"\\bold{(.*)}"

    try:
        # 查找所有匹配项
        matches = re.findall(boxed_pattern, solution_text)

        if matches:
            # 提取最后一个匹配项的内容
            last_match = matches[-1].strip()
            print(f"[DEBUG]成功提取输出: '{last_match}'")
            return last_match
        else:
            # 没有找到匹配项
            print(f"[DEBUG]未找到格式化标记，文本长度: {len(solution_text)}")
            return None

    except re.error as e:
        print(f"[ERROR]正则表达式匹配失败: {e}")
        return None
    except Exception as e:
        print(f"[ERROR]提取输出时发生未知错误: {e}")
        return None


def process_stream_response(response: requests.Response) -> Generator[str, None, None]:
    """
    处理流式API响应，逐行提取内容

    该函数处理服务器推送的流式响应数据，实时提取并返回文本内容。
    支持SSE（Server-Sent Events）格式的数据流处理。

    Args:
        response (requests.Response): HTTP响应对象
            - 必须是流式响应，包含iter_lines()方法
            - 响应格式应为SSE格式（data:开头）

    Yields:
        str: 从流式响应中提取的文本内容片段
            - 每个片段都是连续的文本块
            - 可能包含部分单词或完整句子

    Raises:
        ValueError: 如果响应对象无效或不是流式响应
        UnicodeDecodeError: 如果响应编码不是UTF-8

    Example:
        >>> response = requests.post(url, stream=True)
        >>> for content_chunk in process_stream_response(response):
        ...     print(f"收到内容: {content_chunk}")
    """
    # 参数验证
    if not response or not hasattr(response, "iter_lines"):
        raise ValueError("无效的响应对象，必须是支持流式处理的响应")

    if not response.ok:
        raise ValueError(f"HTTP响应状态码异常: {response.status_code}")

    line_count = 0
    content_count = 0

    print("[INFO]开始处理流式响应...")

    for line in response.iter_lines():
        line_count += 1

        # 跳过空行
        if not line:
            continue

        try:
            # 解码UTF-8编码的字节流
            line_str = line.decode("utf-8")

            # 检查是否为SSE格式的数据行
            if line_str.startswith("data:"):
                try:
                    # 解析JSON数据（去除'data:'前缀）
                    json_data = json.loads(line_str[5:])

                    # 安全地提取内容
                    choices = json_data.get("choices", [{}])
                    if choices:
                        first_choice = choices[0]
                        delta = first_choice.get("delta", {})
                        content = delta.get("content", "").strip()

                        # 只返回非空内容
                        if content:
                            content_count += 1
                            print(
                                f"[DEBUG]提取到内容片段 #{content_count}: {content[:50]}..."
                            )
                            yield content

                except json.JSONDecodeError as e:
                    print(f"[WARNING]JSON解析失败 (行{line_count}): {e}")
                    print(f"[DEBUG]原始数据: {line_str[:100]}")
                    continue

        except UnicodeDecodeError as e:
            print(f"[ERROR]UTF-8解码失败 (行{line_count}): {e}")
            continue
        except Exception as e:
            print(f"[ERROR]处理流数据时发生错误 (行{line_count}): {e}")
            continue

    print(
        f"[INFO]流式响应处理完成，处理行数: {line_count}, 提取内容片段: {content_count}"
    )


def process_non_stream_response(response: requests.Response) -> str:
    """
    处理非流式API响应，提取完整内容

    该函数处理标准的JSON格式响应，提取模型生成的完整文本内容。
    支持OpenAI兼容的API响应格式。

    Args:
        response (requests.Response): HTTP响应对象
            - 必须是成功的HTTP响应（状态码200）
            - 响应体应为有效的JSON格式

    Returns:
        str: 从响应中提取的文本内容
            - 已去除首尾空白字符
            - 如果内容为空则返回空字符串

    Raises:
        ValueError: 如果响应格式不符合预期或缺少必要字段
        json.JSONDecodeError: 如果响应体不是有效的JSON格式
        requests.RequestException: 如果HTTP请求失败

    Example:
        >>> response = requests.post(url, json=data)
        >>> content = process_non_stream_response(response)
        >>> print(f"模型生成内容: {content}")
    """
    # 参数验证
    if not response or not response.ok:
        raise ValueError(
            f"HTTP响应状态码异常: {response.status_code if response else '无响应'}"
        )

    try:
        # 解析JSON响应
        response_data = response.json()

        # 验证响应结构
        if "choices" not in response_data:
            raise ValueError("响应缺少'choices'字段")

        choices = response_data["choices"]
        if not choices or not isinstance(choices, list):
            raise ValueError("'choices'字段为空或不是列表")

        first_choice = choices[0]
        if "message" not in first_choice:
            raise ValueError("choice缺少'message'字段")

        message = first_choice["message"]
        if "content" not in message:
            raise ValueError("message缺少'content'字段")

        # 提取并清理内容
        content = message["content"].strip()

        print(f"[INFO]成功提取内容，长度: {len(content)} 字符")
        if len(content) > 100:
            print(f"[DEBUG]内容预览: {content[:100]}...")
        else:
            print(f"[DEBUG]完整内容: {content}")

        return content

    except json.JSONDecodeError as e:
        error_msg = f"无效的JSON响应: {e}"
        print(f"[ERROR]{error_msg}")
        print(f"[DEBUG]响应文本: {response.text[:200]}")
        raise ValueError(error_msg)

    except KeyError as e:
        error_msg = f"响应格式异常，缺少必要字段: {e}"
        print(f"[ERROR]{error_msg}")
        print(
            f"[DEBUG]响应结构: {list(response_data.keys()) if 'response_data' in locals() else '未解析'}"
        )
        raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"处理非流式响应时发生未知错误: {e}"
        print(f"[ERROR]{error_msg}")
        raise ValueError(error_msg)


def prime_query_openai_async(
    config: Dict[str, Any], sequence_str: str
) -> Optional[requests.Response]:
    """
    异步查询（匿名模型）API进行评分

    该函数是核心API调用接口，支持重试机制和错误处理。
    根据配置参数构建完整的对话消息，调用（匿名模型）API获取评分结果。

    Args:
        config (Dict[str, Any]): API配置字典
            - url (str): API服务地址
            - model (str): 模型名称
            - wsid (str): 服务组标识
            - api_key (str): 认证令牌
            - stream (bool): 是否使用流式响应
            - temperature (float): 温度参数
            - top_p (float): 核采样参数
            - top_k (int): Top-K采样参数
            - repetition_penalty (float): 重复惩罚系数
            - output_seq_len (int): 输出序列长度
            - max_input_seq_len (int): 最大输入序列长度
            - max_retries (int): 最大重试次数
            - system_prompt (str): 系统提示词
            - scoring_prompt (str): 评分提示词
            - tokenizer (str, optional): 分词器路径
            - apply_chat_template (bool, optional): 是否应用聊天模板

        sequence_str (str): 待评分的序列文本
            - 如: "模型生成的文本内容"

    Returns:
        Optional[requests.Response]: API响应对象
            - 成功时返回Response对象
            - 失败时返回None

    Raises:
        ValueError: 配置参数错误或缺失
        ConnectionError: 网络连接失败
        TimeoutError: 请求超时

    Example:
        >>> config = {
        ...     'url': 'http://api.example.com',
        ...     'model': 'your-model',
        ...     'max_retries': 3,
        ...     'system_prompt': '你是一个评分助手',
        ...     'scoring_prompt': '请对以下文本评分:'
        ... }
        >>> sequence = "这是一段需要评分的文本"
        >>> response = prime_query_openai_async(config, sequence)
        >>> if response:
        ...     score = extract_output(response.text)
        ...     print(f"评分结果: {score}")
    """
    # 1. 参数验证
    required_config_keys = [
        "url",
        "model",
        "wsid",
        "api_key",
        "max_retries",
        "system_prompt",
        "scoring_prompt",
    ]

    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"配置参数缺失: {key}")

    if not sequence_str or not sequence_str.strip():
        raise ValueError("sequence_str不能为空")

    print(f"[INFO]开始API查询，序列长度: {len(sequence_str)} 字符")

    # 2. 初始化API客户端
    try:
        api_client = VendorTrainOpenAPI(
            base_url=config["url"],
            model=config["model"],
            wsid=config["wsid"],
            auth_token=config["api_key"],
            stream=config.get("stream", False),
            temperature=config.get("temperature", 0.1),
            top_p=config.get("top_p", 0.8),
            top_k=config.get("top_k", 20),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            output_seq_len=config.get("output_seq_len", 64),
            max_input_seq_len=config.get("max_input_seq_len", 10240),
        )
    except Exception as e:
        print(f"[ERROR]API客户端初始化失败: {e}")
        return None

    # 3. 重试机制
    max_retries = config["max_retries"]

    for attempt in range(max_retries):
        print(f"[INFO]尝试第 {attempt + 1}/{max_retries} 次API调用...")

        try:
            # 4. 构建消息结构
            messages = [
                {"role": "system", "content": config["system_prompt"]},
                {
                    "role": "user",
                    "content": f"{config['scoring_prompt']}\n{sequence_str}",
                },
            ]

            print(f"[DEBUG]系统提示词长度: {len(config['system_prompt'])}")
            print(f"[DEBUG]用户提示词长度: {len(config['scoring_prompt'])}")
            print(f"[DEBUG]序列文本长度: {len(sequence_str)}")

            # 5. 应用聊天模板（如果配置）
            if config.get("tokenizer") and config.get("apply_chat_template"):
                print("[INFO]应用聊天模板...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
                    messages = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    print("[INFO]聊天模板应用成功")
                except Exception as e:
                    print(f"[WARNING]聊天模板应用失败: {e}")
                    print("[INFO]使用原始消息格式")

            # 6. 执行API调用
            print("[INFO]开始API调用...")
            response = api_client.chat_completion(messages=messages)

            # 7. 验证响应
            if not response:
                raise ValueError("API响应为空")

            if not response.ok:
                raise ValueError(f"HTTP状态码异常: {response.status_code}")

            print(f"[SUCCESS]API调用成功，状态码: {response.status_code}")
            return response

        except Exception as e:
            print(f"[ERROR]第 {attempt + 1} 次尝试失败: {str(e)}")
            traceback.print_exc()

            # 最后一次尝试失败
            if attempt == max_retries - 1:
                print(f"[ERROR]所有 {max_retries} 次尝试均失败")
                return None

            # 指数退避等待
            wait_time = 2**attempt
            print(f"[INFO]等待 {wait_time} 秒后重试...")
            import time

            time.sleep(wait_time)

    return None


async def query_openai_async(
    client: VendorTrainOpenAPI,
    sequence_str: str,
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    index: int,
) -> tuple[int, float]:
    """
    异步查询（匿名模型）API获取评分结果

    该函数是异步版本的核心API调用接口，使用信号量控制并发数，
    支持重试机制和错误降级处理。

    Args:
        client (VendorTrainOpenAPI): 已初始化的API客户端
        sequence_str (str): 待评分的序列文本
        config (Dict[str, Any]): API配置参数
        semaphore (asyncio.Semaphore): 并发控制信号量
        index (int): 任务索引，用于结果排序

    Returns:
        tuple[int, float]: 包含索引和评分的元组
            - index: 原始任务索引
            - score: 评分结果（0.0-1.0）

    Raises:
        asyncio.CancelledError: 任务被取消
        ValueError: 配置参数错误

    Note:
        - 使用信号量控制并发数，避免API限流
        - 内置重试机制，失败时返回默认分数
        - 支持聊天模板应用和结果提取
    """
    max_retries = config["max_retries"]
    scoring_prompt = config["scoring_prompt"]

    async with semaphore:
        for attempt in range(max_retries):
            print(
                f"[ASYNC]开始异步查询，索引: {index}, 尝试: {attempt + 1}/{max_retries}"
            )

            try:
                # 构建消息结构
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的评估助手，请根据给定的评估标准对内容进行评估，并严格按照要求的JSON格式输出评估结果。",
                    },
                    {"role": "user", "content": f"{scoring_prompt}\n{sequence_str}"},
                ]

                # 应用聊天模板（如果配置）
                if config.get("tokenizer") and config.get("apply_chat_template"):
                    print(f"[ASYNC]应用聊天模板，索引: {index}")
                    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
                    messages = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                # 异步API调用
                print(f"[ASYNC]开始API调用，索引: {index}")
                response = await asyncio.to_thread(
                    client.chat_completion, messages=messages
                )

                # 验证响应
                if not response or not response.ok:
                    raise ValueError(
                        f"API响应异常，状态码: {response.status_code if response else '无响应'}"
                    )

                # 提取内容
                content = response.json()["choices"][0]["message"]["content"]
                print(f"[ASYNC]API调用成功，索引: {index}, 内容长度: {len(content)}")

                # 提取评分结果
                extracted = extract_output(content)
                print(f"[ASYNC]评分提取结果，索引: {index}, 结果: {extracted}")

                if extracted is not None:
                    score = float(extracted)
                    # 验证分数范围
                    if 0.0 <= score <= 1.0:
                        print(f"[SUCCESS]异步查询完成，索引: {index}, 分数: {score}")
                        return index, score
                    else:
                        print(f"[WARNING]分数超出范围，索引: {index}, 分数: {score}")
                        raise ValueError("分数超出有效范围(0.0-1.0)")
                else:
                    raise ValueError("无法从响应中提取有效评分")

            except Exception as e:
                print(
                    f"[ERROR]异步查询失败，索引: {index}, 尝试: {attempt + 1}, 错误: {str(e)}"
                )

                # 最后一次尝试失败，返回默认分数
                if attempt == max_retries - 1:
                    default_score = config.get("default_score", 0.5)
                    print(
                        f"[WARNING]所有尝试失败，索引: {index}, 使用默认分数: {default_score}"
                    )
                    return index, default_score

                # 指数退避等待
                wait_time = 2**attempt
                print(f"[INFO]等待 {wait_time} 秒后重试，索引: {index}")
                await asyncio.sleep(wait_time)

        # 理论上不会执行到这里
        default_score = config.get("default_score", 0.5)
        return index, default_score


async def process_data_async(
    data_source: List[str],
    solution_str: List[str],
    ground_truth: List[str],
    extra_info: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    异步批量处理数据并获取评分

    该函数支持批量异步处理多个数据样本，使用并发控制优化性能。
    每个样本都会调用（匿名模型）API获取评分，最终返回评分张量。

    Args:
        data_source (List[str]): 数据源标识列表
        solution_str (List[str]): 待评分的解决方案列表
        ground_truth (List[str]): 参考答案列表
        extra_info (List[Dict[str, Any]]): 额外信息列表
        config (Dict[str, Any]): API配置参数

    Returns:
        torch.Tensor: 评分结果张量
            - 形状: (len(solution_str),)
            - 数据类型: torch.float32
            - 值范围: 0.0-1.0

    Raises:
        ValueError: 输入参数长度不匹配
        RuntimeError: 异步任务执行失败

    Example:
        >>> solutions = ["答案1", "答案2", "答案3"]
        >>> references = ["参考1", "参考2", "参考3"]
        >>> scores = await process_data_async(["source1", "source2", "source3"], solutions, references, [{}]*3, config)
        >>> print(f"评分结果: {scores}")
    """
    # 参数验证
    if len(solution_str) != len(ground_truth):
        raise ValueError("solution_str和ground_truth长度必须相同")

    if extra_info and len(extra_info) != len(solution_str):
        raise ValueError("extra_info长度必须与solution_str相同")

    print(f"[INFO]开始异步批量处理，样本数量: {len(solution_str)}")

    # 初始化评分张量
    reward_tensor = torch.zeros(len(solution_str), dtype=torch.float32)

    # 初始化API客户端
    try:
        api_client = VendorTrainOpenAPI(
            base_url=config["url"],
            model=config["model"],
            wsid=config["wsid"],
            auth_token=config["api_key"],
            stream=config.get("stream", False),
            temperature=config.get("temperature", 0.1),
            top_p=config.get("top_p", 0.8),
            top_k=config.get("top_k", 20),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            output_seq_len=config.get("output_seq_len", 64),
            max_input_seq_len=config.get("max_input_seq_len", 10240),
        )
    except Exception as e:
        raise RuntimeError(f"API客户端初始化失败: {e}")

    # 创建信号量控制并发数
    max_concurrency = config.get("max_concurrency", 5)
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"[INFO]并发控制: 最大并发数 {max_concurrency}")

    # 创建异步任务列表
    tasks = []
    for i in range(len(solution_str)):
        prompt = solution_str[i]
        response = ground_truth[i]

        # 构建查询序列
        if response is None:
            sequence_str = prompt
        else:
            sequence_str = f"{prompt}\nReference:\n{response}"

        # 创建异步任务
        task = asyncio.create_task(
            query_openai_async(api_client, sequence_str, config, semaphore, i)
        )
        tasks.append(task)

        print(f"[DEBUG]创建异步任务，索引: {i}, 序列长度: {len(sequence_str)}")

    # 等待所有任务完成
    print("[INFO]等待所有异步任务完成...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    success_count = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"[ERROR]异步任务执行失败: {result}")
            continue

        if isinstance(result, tuple) and len(result) == 2:
            index, score = result
            reward_tensor[index] = score
            success_count += 1
            print(f"[DEBUG]处理结果，索引: {index}, 分数: {score}")

    print(f"[INFO]异步批量处理完成，成功: {success_count}/{len(solution_str)}")
    return reward_tensor
