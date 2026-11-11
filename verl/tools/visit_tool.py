#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网页访问工具 - verl框架集成

该模块提供了网页访问和内容提取功能，与verl框架的工具接口对齐。
支持批量URL访问、内容提取和错误处理。

主要功能：
- ✅ 网页内容访问和爬取
- ✅ 批量URL并行访问
- ✅ 内容提取和摘要
- ✅ 错误处理和异常恢复
- ✅ Ray分布式执行框架集成

作者: anonymous@example.com
版本: 1.0.0
日期: 2025-01-20
"""

import json
import logging
import os
import importlib
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import ray

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .search_tool_vendor_search import (
    PoolMode,
    SearchExecutionWorker,
    init_search_execution_pool,
    extract_prompt_str,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_func(func_path: str):
    """动态导入函数
    
    Args:
        func_path: 函数路径，格式为 "module.path.function_name"
        
    Returns:
        导入的函数对象
    """
    module_path, func_name = func_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


class VisitTool(BaseTool):
    """网页访问工具
    
    该工具提供网页内容访问和提取功能，支持：
    - 单个/批量URL访问
    - 内容提取和摘要
    - 错误处理和异常恢复
    - Ray分布式执行
    
    注意：visit工具是独立的HTTP访问功能，不依赖于搜索后端，所有后端都支持visit
    
    方法：
        get_openai_tool_schema: 返回OpenAI格式的工具schema
        create: 为轨迹创建工具实例
        execute: 执行网页访问
        calc_reward: 计算工具状态的奖励
        release: 释放工具实例
    """
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """初始化网页访问工具
        
        Args:
            config: 配置字典
                - num_workers: 工作线程数（默认10）
                - rate_limit: 速率限制（默认10）
                - timeout: 请求超时时间（默认60秒）
                - max_tool_response_length: 工具响应最大长度（可选）
            tool_schema: OpenAI函数工具schema定义
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # 配置参数
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.timeout = config.get("timeout", 60)
        self.max_tool_response_length = config.get("max_tool_response_length", None)
        
        # 初始化执行池
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        
        print(f"初始化VisitTool，配置: {config}")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """返回OpenAI工具schema"""
        return self.tool_schema
    
    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """创建工具实例"""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()
    
    def _clean_urls(self, urls: List[str]) -> List[str]:
        """清理和验证URL列表
        
        Args:
            urls: 原始URL列表
            
        Returns:
            清理后的有效URL列表
        """
        cleaned_urls = []
        for url in urls:
            if isinstance(url, str):
                # 去除首尾空白和引号
                cleaned = url.strip().strip('"').strip("'")
                # 检查URL是否有效
                if cleaned and (cleaned.startswith('http://') or cleaned.startswith('https://')):
                    cleaned_urls.append(cleaned)
                    print(f"[Visit] URL有效: {cleaned}")
                else:
                    print(f"[Visit] URL无效或格式错误: {repr(url)} -> {repr(cleaned)}")
            else:
                print(f"[Visit] URL类型错误: {type(url)} - {url}")
        return cleaned_urls
    
    def execute_visit(
        self,
        instance_id: str,
        urls: List[str],
        goal: str,
    ) -> Tuple[str, bool]:
        """执行网页访问
        
        Args:
            instance_id: 工具实例ID
            urls: URL列表
            goal: 访问目标/目的
            
        Returns:
            tuple: (result_str, success)
        """
        # 打印原始参数帮助调试
        print(f"[Visit] 原始 urls: {urls}")
        print(f"[Visit] urls 类型: {type(urls)}")
        print(f"[Visit] goal: {goal[:100] if goal else '(空)'}...")
        
        if not urls:
            print(f"[Visit] 错误: 未提供URL")
            return "No URLs provided", False
        
        # URL格式清理和验证
        cleaned_urls = self._clean_urls(urls)
        
        if not cleaned_urls:
            print(f"[Visit] 错误: 没有有效的URL")
            return "No valid URLs provided. URLs must start with http:// or https://", False
        
        # 动态导入访问函数
        try:
            from verl.tools.utils.call_visit import call_visit
        except ImportError as e:
            print(f"[Visit] 导入错误: {e}")
            try:
                call_visit = _get_func("verl.tools.utils.call_visit.call_visit")
            except Exception as e2:
                print(f"[Visit] 备用导入也失败: {e2}")
                return f"Import error: {e}", False
        
        # 执行访问 - call_visit期望一个字典参数 {"url": [...], "goal": "..."}
        params = {"url": cleaned_urls, "goal": goal}
        print(f"[Visit] 最终请求参数: {params}")
        print(f"[Visit] 开始访问...")
        
        try:
            result = call_visit(params)
        except Exception as e:
            print(f"[Visit] 调用异常: {e}")
            return f"Visit call failed: {str(e)}", False
        
        # 打印结果长度和预览
        print(f"[Visit] 返回长度: {len(result) if result else 0} chars")
        if result:
            print(f"[Visit] 结果预览: {result[:500]}...")
        else:
            print(f"[Visit] 警告: 返回结果为空!")
        
        # 检查是否有错误或空结果
        if not result or len(result.strip()) == 0:
            print(f"[Visit] 执行失败: 返回结果为空")
            return "Visit returned empty content. The webpage may be inaccessible or require authentication.", False
        
        if result.startswith("[ERROR]") or (result.startswith("[Visit]") and "Failed" in result):
            print(f"[Visit] 执行失败: {result}")
            return result, False
        
        return result, True
    
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行网页访问工具
        
        Args:
            instance_id: 工具实例ID
            parameters: 工具参数，包含url和goal
                - url: URL列表或单个URL字符串
                - goal: 访问目的/目标
            **kwargs: 其他参数
            
        Returns:
            tuple: (tool_response, tool_reward_score, tool_metrics)
        """
        # 打印原始参数帮助调试
        print(f"[Visit] 原始 tool_arguments: {parameters}")
        print(f"[Visit] tool_arguments 类型: {type(parameters)}")
        
        urls = parameters.get('url', [])
        goal = parameters.get('goal', '')
        
        # 兼容处理：如果url是单个字符串，转换为列表
        if isinstance(urls, str):
            print(f"[Visit] url是字符串，转换为列表: {urls}")
            urls = [urls]
        
        # 详细打印URL信息
        print(f"[Visit] urls 类型: {type(urls)}")
        print(f"[Visit] urls 内容: {urls}")
        if isinstance(urls, list):
            for i, url in enumerate(urls):
                print(f"[Visit] url[{i}] 类型={type(url)}, 值={repr(url)}")
        
        if not urls:
            error_msg = "错误：参数中缺少'url'或为空。"
            return (
                ToolResponse(text=json.dumps({"result": error_msg}, ensure_ascii=False)),
                0.0,
                {},
            )
        
        try:
            # 使用Ray执行池执行访问
            result_str, success = await self.execution_pool.execute.remote(
                self.execute_visit,
                instance_id,
                urls,
                goal,
            )
            
            # 存储结果用于奖励计算
            self._instance_dict[instance_id]["reward"].append(result_str.strip())
            
            # 构建指标
            metrics = {
                "url_count": len(urls) if isinstance(urls, list) else 1,
                "status": "success" if success else "failed",
                "result_length": len(result_str) if result_str else 0,
            }
            
            # 响应长度限制
            if self.max_tool_response_length and len(result_str) > self.max_tool_response_length:
                result_str = result_str[:self.max_tool_response_length] + "...(已截断)"
            
            return (
                ToolResponse(text=json.dumps({"result": result_str}, ensure_ascii=False)),
                0.0,
                metrics,
            )
            
        except Exception as e:
            error_result = json.dumps(
                {"result": f"网页访问执行失败: {e}"}, ensure_ascii=False
            )
            print(f"[VisitTool] 执行失败: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """计算工具实例的奖励"""
        return self._instance_dict[instance_id]["reward"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """释放工具实例"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
