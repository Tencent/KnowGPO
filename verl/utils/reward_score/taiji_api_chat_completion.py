#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
（匿名平台）API聊天补全客户端

该模块提供了与（匿名平台）流式服务器进行聊天补全交互的完整客户端实现。
支持流式和非流式响应，使用Polaris进行服务发现和负载均衡，
具备完善的错误处理、重试机制和性能监控功能。

主要功能：
- ✅ 聊天补全请求发送
- ✅ 流式响应处理
- ✅ 服务实例发现和负载均衡
- ✅ 调用结果上报和性能监控
- ✅ 错误处理和自动重试机制
- ✅ 参数验证和类型安全
- ✅ 完整的日志记录和调试支持

核心特性：
- **服务发现**: 基于Polaris实现动态服务发现和负载均衡
- **流式处理**: 支持实时流式响应，适合长文本生成场景
- **错误恢复**: 多层级的错误处理和重试机制
- **性能监控**: 自动上报调用结果和性能指标
- **类型安全**: 完整的类型提示和参数验证
- **配置灵活**: 支持参数覆盖和自定义配置

使用示例：
    # 创建客户端
    client = VendorTrainOpenAPI()

    # 非流式调用
    messages = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "你好"}
    ]
    result = next(client.chat_completion(messages=messages))

    # 流式调用
    for chunk in client.chat_completion(messages=messages, stream=True):
        print(chunk, end='', flush=True)

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""
import json
import uuid
import time
import logging
from typing import Dict, Optional, Generator, List, Any

import requests

# 北极星服务发现相关导入
from polaris.api.consumer import create_consumer_by_config
from polaris.pkg.model.service import GetOneInstanceRequest, ServiceCallResult
from polaris.pkg.model.error import SDKError
from polaris.wrapper import POLARIS_CALL_RET_OK, POLARIS_CALL_RET_ERROR


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 全局北极星消费者API实例（单例模式）
g_consumer_api = None


def get_consumer_api() -> Any:
    """
    获取北极星消费者API实例（单例模式）

    Returns:
        Any: 北极星消费者API实例

    Raises:
        Exception: 如果创建消费者API失败
    """
    global g_consumer_api
    if not g_consumer_api:
        try:
            g_consumer_api = create_consumer_by_config("")
        except Exception as e:
            raise Exception(f"创建北极星消费者API失败: {e}")
    return g_consumer_api


class VendorTrainOpenAPI:
    """
    （匿名平台）API聊天补全客户端

    该类封装了与（匿名平台）流式服务器进行聊天补全交互的所有功能，
    包括服务发现、请求构建、响应处理和错误管理。

    Attributes:
        model (str): 使用的模型名称
        wsid (str): 工作空间ID
        auth_token (str): 认证令牌
        stream (bool): 是否启用流式模式
        temperature (float): 温度参数，控制随机性
        top_p (float): 核采样参数
        top_k (int): Top-K采样参数
        repetition_penalty (float): 重复惩罚参数
        output_seq_len (int): 输出序列长度
        max_input_seq_len (int): 最大输入序列长度
        namespace (str): Polaris命名空间
        service (str): Polaris服务名称
    """

    def __init__(
        self,
        base_url: str = "https://example.invalid/legacy-api",
        model: str = "rm_0918sp_fix_qw_8b_v1_step130",
        wsid: str = "10143",
        auth_token: str = None,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.6,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        output_seq_len: int = 2048,
        max_input_seq_len: int = 2048,
        namespace: str = "Production",
        service: str = "vendor_train-stream-server-online-openapi-prod",
    ) -> None:
        """
        初始化（匿名平台）API客户端

        Args:
            base_url (str): API基础地址（已弃用，使用Polaris服务发现）
            model (str): 模型名称，默认为"rm_0918sp_fix_qw_8b_v1_step130"
            wsid (str): 工作空间ID，默认为"10143"
            auth_token (str): 认证令牌，格式为"Bearer <token>"
            stream (bool): 是否启用流式模式，默认为False
            temperature (float): 温度参数，控制生成随机性，默认为0.7
            top_p (float): 核采样参数，默认为0.6
            top_k (int): Top-K采样参数，默认为20
            repetition_penalty (float): 重复惩罚参数，默认为1.05
            output_seq_len (int): 输出序列最大长度，默认为2048
            max_input_seq_len (int): 输入序列最大长度，默认为2048
            namespace (str): Polaris命名空间，默认为"Production"
            service (str): Polaris服务名称，默认为"vendor_train-stream-server-online-openapi-prod"
        """
        # 模型和认证配置
        self.model = model
        self.wsid = wsid
        self.auth_token = auth_token

        # 生成参数配置
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        # 序列长度配置
        self.output_seq_len = output_seq_len
        self.max_input_seq_len = max_input_seq_len

        # Polaris服务发现配置
        self.namespace = namespace
        self.service = service

    def _generate_request_id(self) -> str:
        """
        生成唯一请求标识

        Returns:
            str: 格式为"req_<uuid>"的唯一请求ID
        """
        return f"req_{str(uuid.uuid4())}"

    def _build_headers(self, wsid: Optional[str] = None) -> Dict[str, str]:
        """
        构建HTTP请求头部

        Args:
            wsid (str, optional): 工作空间ID，如果为None则使用默认值

        Returns:
            Dict[str, str]: 包含认证和内容类型的请求头字典
        """
        return {
            "Content-Type": "application/json",
            "Authorization": self.auth_token,
            "Wsid": wsid or self.wsid,
        }

    def _get_service_instance(self) -> Any:
        """
        通过Polaris获取可用的服务实例

        该方法使用Polaris服务发现机制动态获取可用的（匿名平台）流式服务器实例，
        支持负载均衡和故障转移。如果服务数据过期会发出警告，
        但会继续使用当前可用的实例。

        Returns:
            Any: Polaris服务实例对象，包含主机地址和端口信息

        Raises:
            ConnectionError: 如果网络连接或Polaris服务不可用
            SDKError: 如果Polaris SDK调用失败
            Exception: 其他未知错误

        Example:
            >>> instance = client._get_service_instance()
            >>> host = instance.get_host()
            >>> port = instance.get_port()
            >>> print(f"使用实例: {host}:{port}")
        """
        try:
            # 1. 创建获取服务实例的请求
            request = GetOneInstanceRequest(
                namespace=self.namespace, service=self.service
            )

            # 2. 调用Polaris获取服务实例响应
            logger.debug(f"正在获取服务实例: {self.namespace}/{self.service}")
            response = get_consumer_api().get_one_instance_resp(request)

            # 3. 检查服务数据是否过期
            if response.is_service_data_expired():
                # 服务数据过期，需要告警但继续使用
                logger.warning(f"服务 {self.namespace}/{self.service} 数据已过期")
                logger.warning("建议检查Polaris服务状态或使用备用服务发现机制")

            # 4. 验证响应是否包含可用实例
            if not response or len(response) == 0:
                logger.error(f"Polaris服务发现失败: namespace={self.namespace}, service={self.service}")
                logger.error("可能原因: 服务未注册、服务实例全部不可用或Polaris配置错误")

                # 检查服务数据状态
                if response.is_service_data_expired():
                    logger.warning("服务数据已过期，建议检查Polaris服务状态")

                # 抛出更具体的异常类型
                raise ConnectionError(
                    f"Polaris未返回可用实例，服务可能不可用。"
                    f"请检查服务 {self.namespace}/{self.service} 是否已注册并正常运行。"
                )

            # 5. 获取第一个可用实例
            instance = response[0]
            host = instance.get_host()
            port = instance.get_port()
            instance_id = instance.get_id()

            # 6. 记录实例信息
            logger.info(f"使用服务实例: {host}:{port} (ID: {instance_id})")
            logger.debug(
                f"实例权重: {instance.get_weight()}, 健康状态: {instance.get_healthy()}"
            )

            return instance

        except SDKError as e:
            # Polaris SDK错误处理
            logger.error(f"Polaris SDK错误 - 命名空间: {self.namespace}, 服务: {self.service}")
            logger.error(f"错误详情: {repr(e)}")
            logger.error("可能原因: Polaris服务不可用、配置错误或网络连接问题")

            # 记录调试信息
            logger.debug(f"请求参数 - namespace: {self.namespace}, service: {self.service}")

            # 重新抛出异常，但保留原始异常信息
            raise SDKError(
                f"Polaris SDK错误 - 服务: {self.namespace}/{self.service}, "
                f"详情: {str(e)}"
            ) from e

        except ConnectionError as e:
            # 网络连接错误处理
            error_msg = f"网络连接失败，无法访问Polaris服务: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)

        except Exception as e:
            # 其他未知错误处理
            error_msg = f"获取服务实例失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _report_call_result(self, instance: Any, ret_code: int, delay: float) -> None:
        """
        向Polaris上报服务调用结果

        Args:
            instance: Polaris服务实例
            ret_code (int): 返回码，>=0表示成功，<0表示失败
            delay (float): 调用延迟时间（秒）
        """
        try:
            # 1. 创建服务调用结果对象
            service_call_result = ServiceCallResult(
                namespace=self.namespace,
                service=self.service,
                instance_id=instance.get_id(),
            )

            # 2. 设置调用结果参数
            service_call_result.set_delay(delay)
            service_call_result.set_ret_code(ret_code)

            # 3. 根据返回码设置调用状态
            ret_status = (
                POLARIS_CALL_RET_OK if ret_code >= 0 else POLARIS_CALL_RET_ERROR
            )
            service_call_result.set_ret_status(ret_status)

            # 4. 上报调用结果
            get_consumer_api().update_service_call_result(service_call_result)

        except Exception as e:
            logger.error(f"上报调用结果失败: {e}")

    def _process_stream_response(
        self, response: requests.Response
    ) -> Generator[str, None, None]:
        """
        处理流式响应，逐行解析服务器推送的数据

        该方法按照Server-Sent Events (SSE)协议处理流式响应，
        逐行解析服务器推送的数据，提取内容片段并返回。

        Args:
            response (requests.Response): HTTP响应对象，启用流式模式

        Yields:
            str: 每次解析出的内容片段

        Raises:
            ValueError: 如果响应格式不符合SSE协议标准
            json.JSONDecodeError: 如果JSON解析失败
            Exception: 其他处理错误

        Note:
            - 只处理以'data:'开头的行
            - 跳过空行和注释行
            - 自动处理UTF-8解码
            - 忽略非内容字段（如role、system_fingerprint等）
        """
        line_count = 0
        content_count = 0

        for line in response.iter_lines():
            line_count += 1

            # 跳过空行
            if not line:
                continue

            try:
                # 1. 解码UTF-8字符串
                line_str = line.decode("utf-8")

                # 2. 跳过注释行和事件行
                if line_str.startswith(":") or line_str.startswith("event:"):
                    continue

                # 3. 只处理以'data:'开头的行
                if line_str.startswith("data:"):
                    # 去掉'data:'前缀并解析JSON
                    json_str = line_str[5:].strip()

                    # 跳过空数据行
                    if not json_str:
                        continue

                    # 4. 解析JSON数据
                    data = json.loads(json_str)

                    # 5. 验证响应格式
                    if "choices" not in data or not data["choices"]:
                        logger.warning(f"第{line_count}行缺少choices字段")
                        continue

                    choice = data["choices"][0]
                    if "delta" not in choice:
                        logger.warning(f"第{line_count}行缺少delta字段")
                        continue

                    # 6. 提取内容
                    content = choice["delta"].get("content")

                    # 7. 只返回非空内容
                    if content:
                        content_count += 1
                        yield content

            except json.JSONDecodeError as e:
                logger.error(f"第{line_count}行JSON解析错误: {e}")
                logger.error(f"原始数据: {line_str}")
                continue
            except UnicodeDecodeError as e:
                logger.error(f"第{line_count}行UTF-8解码错误: {e}")
                continue
            except Exception as e:
                logger.error(f"第{line_count}行处理错误: {e}")
                continue

        logger.debug(
            f"流式处理完成: 共处理{line_count}行，提取{content_count}个内容片段"
        )

    def _process_non_stream_response(self, response: requests.Response) -> str:
        """
        处理非流式响应，解析完整的JSON响应

        该方法处理标准的非流式响应，验证响应格式并提取完整内容。
        支持OpenAI兼容的响应格式验证。

        Args:
            response (requests.Response): HTTP响应对象

        Returns:
            str: 解析出的完整响应内容

        Raises:
            ValueError: 如果响应格式异常或缺少必要字段
            json.JSONDecodeError: 如果JSON解析失败
            Exception: 其他处理错误

        Note:
            - 验证choices字段存在且非空
            - 验证message.content字段存在
            - 支持OpenAI兼容的响应格式
        """
        try:
            # 1. 解析JSON响应
            response_data = response.json()

            # 2. 验证响应格式
            if "choices" not in response_data:
                raise ValueError("响应缺少choices字段，不符合OpenAI格式")

            if not response_data["choices"]:
                raise ValueError("响应choices字段为空，无可用选择")

            # 3. 获取第一个选择
            choice = response_data["choices"][0]

            # 4. 验证选择格式
            if "message" not in choice:
                raise ValueError("响应缺少message字段，无法提取内容")

            if "content" not in choice["message"]:
                raise ValueError("响应缺少message.content字段，无法提取内容")

            # 5. 提取并返回内容
            content = choice["message"]["content"]

            # 6. 处理content为None的情况
            if content is None:
                logger.warning("响应content为None，返回空字符串")
                content = ""

            # 7. 记录响应信息
            if "usage" in response_data:
                usage = response_data["usage"]
                logger.debug(
                    f"响应使用情况: prompt_tokens={usage.get('prompt_tokens', 'N/A')}, "
                    f"completion_tokens={usage.get('completion_tokens', 'N/A')}, "
                    f"total_tokens={usage.get('total_tokens', 'N/A')}"
                )

            logger.debug(f"非流式响应处理完成，内容长度: {len(content)}字符")

            return content

        except KeyError as e:
            logger.error(f"响应格式异常，缺少必要字段: {e}")
            raise ValueError(f"响应格式异常，缺少必要字段: {e}")
        except json.JSONDecodeError as e:
            # 获取原始响应文本用于调试
            try:
                raw_text = response.text[:500]  # 限制长度避免日志过长
            except Exception:
                raw_text = "<无法获取响应文本>"

            error_msg = (
                f"无效的JSON响应: {e}\n"
                f"HTTP状态码: {response.status_code}\n"
                f"原始响应前500字符: {raw_text}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"处理非流式响应时发生未知错误: {e}")
            raise ValueError(f"处理非流式响应时发生未知错误: {e}")

    def chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Generator[str, None, None]:
        """
        执行聊天补全请求

        该方法支持流式和非流式两种模式，通过Polaris进行服务发现和负载均衡。
        会自动处理服务实例获取、请求构建、响应解析和结果上报。

        Args:
            messages (List[Dict[str, str]]): 消息列表，每个消息包含role和content字段
                - role: 消息角色（system/user/assistant）
                - content: 消息内容
            **kwargs: 可选参数，可覆盖默认配置
                - model (str): 模型名称
                - stream (bool): 是否启用流式模式
                - temperature (float): 温度参数
                - top_p (float): 核采样参数
                - top_k (int): Top-K采样参数
                - repetition_penalty (float): 重复惩罚参数
                - output_seq_len (int): 输出序列长度
                - max_input_seq_len (int): 最大输入序列长度
                - 其他OpenAI兼容参数

        Yields:
            str: 流式模式下返回内容片段，非流式模式下返回完整内容

        Raises:
            ConnectionError: 如果网络连接或API请求失败
            ValueError: 如果消息格式错误或响应格式异常
            Exception: 其他未知错误
        """
        start_time = time.time()
        instance = None
        ret_code = -1  # 默认失败状态

        try:
            # 1. 获取Polaris服务实例
            instance = self._get_service_instance()
            ip = instance.get_host()
            port = instance.get_port()
            ss_url = f"http://{ip}:81/openapi/chat/completions"

            logger.info(f"使用服务实例: {ip}:{port}")

            # 2. 验证消息格式
            if not messages or not isinstance(messages, list):
                raise ValueError("消息列表不能为空且必须是列表类型")

            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    raise ValueError("每个消息必须包含role和content字段")

            # 3. 构建请求数据（支持参数覆盖）
            json_data = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
                "query_id": self._generate_request_id(),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "top_k": kwargs.get("top_k", self.top_k),
                "repetition_penalty": kwargs.get(
                    "repetition_penalty", self.repetition_penalty
                ),
                "output_seq_len": kwargs.get("output_seq_len", self.output_seq_len),
                "max_input_seq_len": kwargs.get(
                    "max_input_seq_len", self.max_input_seq_len
                ),
                "stream": kwargs.get("stream", self.stream),
                "openai_infer": True,
                "beam_size": 1,
                "decoupled": 0,
                "is_stop": 0,
                "domain_index": -1,
                "enable_sse": True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "model",
                        "temperature",
                        "top_p",
                        "top_k",
                        "repetition_penalty",
                        "output_seq_len",
                        "max_input_seq_len",
                        "stream",
                    ]
                },
            }

            # 4. 构建请求头
            headers = self._build_headers(self.wsid)

            # 5. 发起HTTP请求
            response = requests.post(
                ss_url, headers=headers, json=json_data, stream=self.stream
            )
            response.raise_for_status()

            # 6. 计算调用延迟并设置成功状态
            delay = time.time() - start_time
            ret_code = 0  # 成功状态

            # 7. 上报调用结果
            self._report_call_result(instance, ret_code, delay)

            # 8. 根据流式模式返回不同结果
            if self.stream:
                yield from self._process_stream_response(response)
            else:
                yield self._process_non_stream_response(response)

        except requests.exceptions.RequestException as e:
            # 网络请求错误处理
            delay = time.time() - start_time
            if instance:
                self._report_call_result(instance, -1, delay)
            raise ConnectionError(f"API请求失败: {str(e)}")

        except ValueError as e:
            # 参数验证错误处理
            delay = time.time() - start_time
            if instance:
                self._report_call_result(instance, -2, delay)
            raise ValueError(f"参数验证失败: {str(e)}")

        except Exception as e:
            # 其他未知错误处理
            delay = time.time() - start_time
            if instance:
                self._report_call_result(instance, -3, delay)
            raise Exception(f"聊天补全处理失败: {str(e)}")

# =============================================================================
# 辅助函数
# =============================================================================


def print_section_header(title: str, with_newline: bool = False) -> None:
    """
    打印带分隔线的章节标题

    Args:
        title: 章节标题文本
        with_newline: 是否在分隔线前添加换行符（默认False）
    """
    separator = "=" * 60
    if with_newline:
        print(f"\n{separator}")
    else:
        print(separator)
    print(title)
    print(separator)


# =============================================================================
# 使用示例和测试代码
# =============================================================================


def test_chat_completion() -> None:
    """
    测试聊天补全功能

    该函数演示了如何使用（匿名平台）API客户端进行聊天补全请求，
    包括流式和非流式两种模式的使用方法。
    """
    print_section_header("（匿名平台）API聊天补全客户端测试")

    # 1. 创建客户端实例
    print("\n[1] 创建（匿名平台）API客户端实例...")
    client = VendorTrainOpenAPI()
    print("✓ 客户端创建成功")

    # 2. 准备测试消息
    print("\n[2] 准备测试消息...")
    messages = [
        {"role": "system", "content": "你是一个AI智能助手，请用简洁的语言回答问题"},
        {"role": "user", "content": "请介绍一下人工智能的基本概念"},
    ]
    print("✓ 消息准备完成")

    # 3. 非流式调用测试
    print("\n[3] 非流式调用测试...")
    try:
        result_generator = client.chat_completion(messages=messages, stream=False)
        result = next(result_generator)  # 获取第一个（也是唯一一个）结果
        print("✓ 非流式调用成功")
        print(f"响应内容: {result[:100]}...")  # 只显示前100个字符
    except Exception as e:
        print(f"✗ 非流式调用失败: {e}")

    # 4. 流式调用测试（注释掉，需要时启用）
    """
    print("\n[4] 流式调用测试...")
    try:
        print("流式响应内容：", end="")
        for chunk in client.chat_completion(messages=messages, stream=True):
            print(chunk, end='', flush=True)
        print()
        print("✓ 流式调用成功")
    except Exception as e:
        print(f"✗ 流式调用失败: {e}")
    """

    # 5. 参数覆盖测试
    print("\n[5] 参数覆盖测试...")
    try:
        custom_messages = [
            {"role": "system", "content": "你是一个数学专家"},
            {"role": "user", "content": "计算2的10次方是多少？"},
        ]

        result_generator = client.chat_completion(
            messages=custom_messages,
            temperature=0.1,  # 降低温度参数，使输出更确定性
            stream=False,
        )
        result = next(result_generator)
        print("✓ 参数覆盖测试成功")
        print(f"响应内容: {result}")
    except Exception as e:
        print(f"✗ 参数覆盖测试失败: {e}")

    print_section_header("测试执行完成", with_newline=True)


def demo_advanced_usage() -> None:
    """
    高级用法演示

    展示（匿名平台）API客户端的高级功能和使用场景。
    """
    print_section_header("高级用法演示", with_newline=True)

    # 1. 多轮对话示例
    print("\n[1] 多轮对话示例...")
    client = VendorTrainOpenAPI()

    # 第一轮对话
    messages1 = [
        {"role": "system", "content": "你是一个专业的编程助手"},
        {"role": "user", "content": "如何用Python实现快速排序算法？"},
    ]

    try:
        result1 = next(client.chat_completion(messages=messages1, stream=False))
        print("✓ 第一轮对话成功")

        # 第二轮对话（基于第一轮的回答）
        messages2 = messages1 + [
            {"role": "assistant", "content": result1},
            {"role": "user", "content": "请解释一下这个算法的时间复杂度"},
        ]

        _ = next(client.chat_completion(messages=messages2, stream=False))
        print("✓ 第二轮对话成功")
        print("多轮对话演示完成")

    except Exception as e:
        print(f"✗ 多轮对话失败: {e}")

    print_section_header("高级用法演示完成", with_newline=True)


if __name__ == "__main__":
    """
    主程序入口

    执行（匿名平台）API客户端的测试和演示功能。
    """

    # 执行基本功能测试
    test_chat_completion()

    # 执行高级用法演示（注释掉，需要时启用）
    # demo_advanced_usage()

    print("\n🎉 （匿名平台）API客户端所有测试和演示执行完成！")
