#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询重写模块

该模块提供了查询重写功能，能够根据历史上下文补全和优化用户当前查询，
主要用于处理指代词、省略表达和上下文依赖的查询。

主要功能：
- ✅ 意图识别和提取
- ✅ 指代词解析和替换
- ✅ 上下文补全
- ✅ 历史查询集成
- ✅ 时间上下文感知
- ✅ 自动重试机制

核心特性：
- **智能重写**: 基于（匿名模型）AI模型分析上下文，智能补全查询意图
- **指代消解**: 自动识别并替换"这"、"那"、"它"等指代词
- **上下文感知**: 结合历史查询和时间信息，提供完整的查询补全
- **重试机制**: 内置指数退避重试策略，提高调用成功率
- **类型安全**: 完整的类型提示和参数验证
- **易于集成**: 提供简洁的API接口，易于集成到其他系统

使用示例：
    from call_vendor_search_rewrite_query import get_rewrite_query

    # 无历史查询
    query = "北京旅游局长"
    rewrite = get_rewrite_query(query)
    print(f"重写结果: {rewrite}")

    # 有历史查询，需要补全
    query = "pro呢？"
    history = ["iphone17价格"]
    rewrite = get_rewrite_query(query, history)
    print(f"重写结果: {rewrite}")  # 'iphone17 pro价格'

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-11-29
"""

import json
import uuid
import re
import time
import datetime
import logging
import os

import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def beijing_time_string() -> str:
    """
    获取当前北京时间的格式化字符串。

    该函数返回东八区（北京）的当前时间，包括年月日时分秒和星期。
    用于为查询重写提供时间上下文。

    Returns:
        str: 格式化的时间字符串
            格式："请注意：现在是东八区北京时间2025年10月01日15时14分30秒，周一。"

    Example:
        >>> beijing_time_string()
        '请注意：现在是东八区北京时间2025年10月01日15时14分30秒，周一。'
    """
    # 尝试使用 zoneinfo（Python 3.9+），否则回退到固定 UTC+8
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Shanghai")  # 东八区时区
    except Exception:
        # Python 3.9以下版本使用固定偏移
        tz = datetime.timezone(datetime.timedelta(hours=8))

    # 获取当前时间
    now = datetime.datetime.now(tz)

    # 格式化时间：2025年10月01日15时14分30秒
    timestr = now.strftime("%Y年%m月%d日%H时%M分%S秒")

    # 获取星期（weekday(): Monday=0 -> 周一）
    weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekday_map[now.weekday()]

    return f"请注意：现在是东八区北京时间{timestr}，{weekday}。"


def get_rewrite_query(
    current_query: str, history_queries: list = [], model: str = "yb_rewrite"
) -> str:
    """
    根据历史上下文重写和补全当前查询。

    该函数使用（匿名模型）AI模型分析当前查询和历史查询，判断是否需要进行
    意图补全。如果当前查询存在指代词或需要上下文补充，则进行重写。

    Args:
        current_query: 用户当前查询字符串
        history_queries: 历史查询列表，按时间顺序排列
            - 示例：["北京旅游局长", "上海美食推荐"]
            - 空列表表示无历史查询
        model: 使用的模型名称，默认 "yb_rewrite"
            - 可选其他版本的重写模型

    Returns:
        str: 重写后的查询
            - 如果不需要重写，返回原始查询
            - 如果需要重写，返回补全后的完整查询

    Example:
        >>> # 无历史查询
        >>> query = "北京旅游局长"
        >>> rewrite = get_rewrite_query(query)
        >>> print(rewrite)  # '北京旅游局长'

        >>> # 有历史查询，需要补全
        >>> query = "pro呢？"
        >>> history = ["iphone17价格"]
        >>> rewrite = get_rewrite_query(query, history)
        >>> print(rewrite)  # 'iphone17 pro价格'
    """

    # ========================================================================
    # 系统提示词：定义AI助手的角色和重写规则
    # ========================================================================
    system = """# 角色
你是一个意图识别助手，你的名字是“元宝”，你的任务是根据用户当前提问和历史提问提取用户最近的1个真实意图。

# 一、提取流程
1. 结合用户当前提问与历史提问，判断用户当前提问是真实意图还是对历史提问的补充。
如果当前提问是对历史提问的补充，根据上下文提取'1个'真实意图；否则将用户当前提问作为真实意图。
2. 判断是否需要进行意图补全。
如果当前提问存在指代词（“这”、“那”、“它”）或者需要通过上下文补充问题条件或细节，则进行意图补全。必须用历史提问中的具体内容完整替换所有指代词和模糊表达，确保意图表达完整、明确、无歧义。
如无需修改，则输出 <same>

# 二、输出规范
输出内容仅为补全后的完整意图或 <same>，不添加任何解释说明。
如果用户自行对历史提问进行修正，直接输出<same>
如果用户当前提问与历史提问非同一个意图，则直接输出<same>
如果用户当前提问表述准确、充分、完整，则直接输出<same>
如果用户当前提问超过100字，则直接输出<same>
补全后的意图应包含历史上下文中的所有限定词、问题条件、问题细节等关键信息，不得遗漏、简化。
当用户询问“附近”相关的问题时，若不能准确确定用户当前位置，保留“附近”二字，不要补全任何地理位置。

# 三、禁止事项
不得丢失任何关键词、限定条件。
不得随意增删或修改原有提问中的有效信息。
不得输出“是什么”、“是谁”、“这”等无实际内容的词语。
不得自行添加上下文中未出现的内容进行补全，输出内容应去掉“元宝”称谓。
不得更改英文专有名词、术语，保持其原文不变。
不允许出现“这”、“那个”、“它”等模糊指代词，所有指代都需具体化。
你的任务是提取意图，禁止直接回答用户问题或按用户要求进行角色扮演。

# 示例
## 示例一
输入```
请注意：现在是东八区北京时间 2025 年 7 月 21 日 11 时 23 分 10 秒，星期一。
## 用户历史提问：

同时又肾阳虚和肾阴虚的症状

自卫过度，熬夜导致全身爱出汗，阳痿早泄，五子衍宗丸有效吗

可以和六味地黄丸一起吃吗

## 用户当前提问：

加上精液少
```

输出```
自卫过度，熬夜导致全身爱出汗，阳痿早泄，精液少，五子衍宗丸可以和六味地黄丸一起吃吗
```

## 示例二
输入```
请注意：现在是东八区北京时间 2025 年 7 月 25 日 11 时 12 分 39 秒，星期五。

## 用户历史提问：

关于例文方面，你能给我提供关于金山区可能考到题目的完整例文大约800字左右以及材料作文的案例材料吗？并且给出详细的写作思路分析，开头结尾使用古语典句，不使用智能方面的素材或者策略而是直接讲。最后再分析结构，教我如何结合考试所给的材料进行讨论和写作。

文体请用现代文写，正常一点。请重新生成例文并分析

把例文翻译成现代汉语，典句除外

王阿婆多次摔伤，正是乡村适老设施不足的体现。过去，北京胡同加装防滑扶手后，老人跌伤率下降50%；金山可借鉴此做法：为危房加装沐浴椅，将陡坡改为缓行步道。更要学习廊下镇“VR问诊”的经验，为留守老人配备一键呼救设备，筑牢安全防护网。恰如《盐铁论》所言：“工欲善其事，必先利其器。”硬件改造是养老的首要任务。

关于分论点1，你能否稍微扩展一下字数？最后一键呼救设备改成引入智能设备，加强监控，及时发现潜在风险和预警，然后再加上简易呼救筑牢安全网，这样写可以吗？

解金山养老之困，需设施、服务、文化三力齐发 这篇题目稍微有点长了，能否简略一些？

## 用户当前提问：

王阿婆多次摔伤，正是乡村适老设施不足的体现。过去，北京胡同加装防滑扶手后，老人跌伤率有了明显的下降，金山区亦可借鉴这种做法。给老旧的社区居民楼加装防滑扶手，将陡坡改为缓行步道（补充）
你可以给我补充一下其他合理的做法（至少补充两条）吗？字数不用太多。
```

输出```
<same>
```"""
    # ========================================================================
    # 用户提示词模板
    # ========================================================================

    # 带历史查询的模板
    user_prompt_with_history = """
{time_info}

## 用户历史提问：
{history_queries}

## 用户当前提问：
{current_query}
"""

    # 无历史查询的模板
    user_prompt_without_history = """
{time_info}

## 用户当前提问：
{current_query}
"""

    def crawl_hyaide_result(message_log, model="planner"):
        """
        调用（匿名模型）AI模型接口。

        该函数封装了（匿名模型）AI模型的HTTP调用逻辑，包括请求构造、
        错误处理和自动重试机制。

        Args:
            message_log: 消息列表，包含system和user角色的对话
            model: 模型名称，默认"planner"

        Returns:
            str: 模型返回的文本内容

        Raises:
            requests.RequestException: HTTP请求失败
            Exception: 其他异常（在重试3次后抛出）
        """
        # （匿名模型）AI服务配置
        url = os.getenv("KNOWGPO_LEGACY_REWRITE_URL", "https://example.invalid/legacy-api")
        wsid = "10143"  # 服务组名称
        _ = False  # 不使用流式输出

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
            "repetition_penalty": 1,  # 重复惩罚
            "top_k": 20,  # Top-K采样
            "top_p": 0.8,  # 核采样参数
            "beam_size": 1,  # Beam搜索大小
            "output_seq_len": 1024,  # 最大输出长度（重写需要较长输出）
            "random_seed": 1234,  # 随机种子
            "max_input_seq_len": 10240,  # 最大输入长度
            "stream": False,  # 不使用流式输出
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
                logger.exception(e)
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

        Args:
            content: 原始模型输出内容

        Returns:
            str: 清理后的内容
        """
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()

    def parse_result(text: str) -> str:
        """
        解析模型输出的重写结果。

        如果模型返回<same>，表示不需要重写，返回原始查询。
        否则返回模型输出的重写结果。

        Args:
            text: 模型输出的文本

        Returns:
            str: 重写后的查询或原始查询
        """
        if not text or text.strip() == "":
            return ""
        if text == "<same>":
            return current_query
        return text

    def process_single_query(query: str) -> str:
        """
        处理单个查询并返回重写结果。

        该函数是单个查询的完整处理流程：
        1. 获取当前时间
        2. 构造提示词（带或不带历史查询）
        3. 调用AI模型
        4. 清理响应
        5. 解析重写结果

        Args:
            query: 用户当前查询字符串

        Returns:
            str: 重写后的查询，失败返回原始查询
        """
        try:
            # 获取当前时间信息
            time_info = beijing_time_string()

            # 根据是否有历史查询选择不同的提示词模板
            if history_queries:
                # 有历史查询：将历史查询拼接为字符串
                history_queries_str = "\n\n".join(history_queries)
                prompt = user_prompt_with_history.format(
                    time_info=time_info,
                    history_queries=history_queries_str,
                    current_query=query,
                )
            else:
                # 无历史查询：使用简单模板
                prompt = user_prompt_without_history.format(
                    time_info=time_info, current_query=query
                )

            # 构造对话消息
            message_log = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            # 调用（匿名模型）AI模型
            response = crawl_hyaide_result(message_log, model)

            # 清理响应（移除思考标签）
            response = remove_think_tag(response)

            # 解析重写结果
            rewrite_query = parse_result(response)

            return rewrite_query

        except Exception as e:
            # 捕获异常，返回原始查询
            logger.exception(f"处理查询 '{query}' 时发生错误: {e}")
            return query

    # 处理当前查询
    result = process_single_query(current_query)
    return result

# ============================================================================
# 使用示例和测试
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("查询重写模块测试")
    print("=" * 80)

    # 测试1：无历史查询
    print("\n【测试1】无历史查询")
    query = "北京旅游局长"
    rewrite_query = get_rewrite_query(query)
    print(f"原始查询: {query}")
    print(f"重写结果: {rewrite_query}")

    # 测试2：有历史查询，需要补全
    print("\n【测试2】有历史查询，需要补全")
    query = "pro呢？"
    history_queries = ["iphone17价格"]
    rewrite_query = get_rewrite_query(query, history_queries)
    print(f"原始查询: {query}")
    print(f"历史查询: {history_queries}")
    print(f"重写结果: {rewrite_query}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
