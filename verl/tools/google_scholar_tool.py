#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Scholar学术搜索工具 - verl框架集成

该模块提供了Google Scholar学术搜索功能，与verl框架的工具接口对齐。
支持批量查询、结果格式化和错误处理。

主要功能：
- ✅ Google Scholar学术论文搜索
- ✅ 批量查询并行处理
- ✅ 搜索结果格式化
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


class GoogleScholarTool(BaseTool):
    """Google Scholar学术搜索工具
    
    该工具提供Google Scholar学术论文搜索功能，支持：
    - 多查询并行搜索
    - 搜索结果格式化
    - 错误处理和异常恢复
    - Ray分布式执行
    
    方法：
        get_openai_tool_schema: 返回OpenAI格式的工具schema
        create: 为轨迹创建工具实例
        execute: 执行学术搜索
        calc_reward: 计算工具状态的奖励
        release: 释放工具实例
    """
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """初始化Google Scholar搜索工具
        
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
        
        print(f"初始化GoogleScholarTool，配置: {config}")
    
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
    
    def _extract_queries(self, tool_arguments: Dict) -> List[str]:
        """从工具参数中提取查询列表
        
        Args:
            tool_arguments: 工具参数字典
            
        Returns:
            查询列表
        """
        # 尝试多种可能的参数名
        queries = tool_arguments.get('query_list', [])
        if not queries:
            queries = tool_arguments.get('queries', [])
        if not queries:
            queries = tool_arguments.get('query', [])
        
        # 如果是单个字符串，转换为列表
        if isinstance(queries, str):
            queries = [queries]
        
        return queries
    
    def execute_google_scholar(
        self,
        instance_id: str,
        queries: List[str],
    ) -> Tuple[str, bool, Dict]:
        """执行Google Scholar搜索
        
        Args:
            instance_id: 工具实例ID
            queries: 查询列表
            
        Returns:
            tuple: (result_str, success, search_stats)
        """
        if not queries:
            print(f"[Google Scholar] 错误: 未提供查询参数")
            return "No queries provided", False, {}
        
        # 动态导入搜索函数
        try:
            from verl.tools.utils.call_google_scholar import (
                call_google_scholar,
                format_search_results_scholar_for_tongyi,
            )
        except ImportError as e:
            print(f"[Google Scholar] 导入错误: {e}")
            try:
                call_scholar = _get_func("verl.tools.utils.call_google_scholar.call_google_scholar")
                format_func = _get_func("verl.tools.utils.call_google_scholar.format_search_results_scholar_for_tongyi")
            except Exception as e2:
                print(f"[Google Scholar] 备用导入也失败: {e2}")
                return f"Import error: {e}", False, {}
        else:
            call_scholar = call_google_scholar
            format_func = format_search_results_scholar_for_tongyi
        
        # 执行搜索
        print(f"[Google Scholar] 开始搜索: {queries}")
        search_error_num, docs = call_scholar(queries)
        
        # 打印原始返回数据
        print(f"[Google Scholar] 原始返回: search_error_num={search_error_num}")
        for q, result in docs.items():
            if isinstance(result, list):
                print(f"  query='{q}': 返回{len(result)}条结果")
                if result:
                    print(f"    第1条: {str(result[0])[:200]}...")
            else:
                print(f"  query='{q}': 错误/无结果 - {result}")
        
        # 统计信息
        total_docs = sum(len(d) for d in docs.values() if isinstance(d, list)) if docs else 0
        search_stats = {
            'search_error_num': search_error_num,
            'total_docs': total_docs,
            'queries': queries
        }
        
        # 格式化结果
        result_str = format_func(docs)
        
        # 如果所有查询都没有返回结果，建议使用通用搜索
        if total_docs == 0:
            suggestion = "\n\n[Suggestion] No relevant academic papers found on Google Scholar. This query may not be suitable for academic search. Consider using the general search tool (search) to find relevant information."
            result_str = result_str + suggestion
            print(f"[Google Scholar] No results for all queries, added suggestion to use search tool")
        
        # 打印结果长度和预览
        print(f"[Google Scholar] 格式化结果: 总文档数={total_docs}, 返回长度={len(result_str)} chars")
        print(f"[Google Scholar] 结果预览: {result_str[:500]}...")
        
        return result_str, True, search_stats
    
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行学术搜索工具
        
        Args:
            instance_id: 工具实例ID
            parameters: 工具参数，包含query_list
            **kwargs: 其他参数
            
        Returns:
            tuple: (tool_response, tool_reward_score, tool_metrics)
        """
        queries = self._extract_queries(parameters)
        
        if not queries:
            error_msg = "错误：参数中缺少'query_list'、为空或不是列表类型。"
            return (
                ToolResponse(text=json.dumps({"result": error_msg}, ensure_ascii=False)),
                0.0,
                {},
            )
        
        try:
            # 使用Ray执行池执行搜索
            result_str, success, search_stats = await self.execution_pool.execute.remote(
                self.execute_google_scholar,
                instance_id,
                queries,
            )
            
            # 存储结果用于奖励计算
            self._instance_dict[instance_id]["reward"].append(result_str.strip())
            
            # 构建指标
            metrics = {
                "query_count": len(queries),
                "status": "success" if success else "failed",
                "total_results": search_stats.get("total_docs", 0),
                "search_error_num": search_stats.get("search_error_num", 0),
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
                {"result": f"Google Scholar搜索执行失败: {e}"}, ensure_ascii=False
            )
            print(f"[GoogleScholarTool] 执行失败: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        """计算工具实例的奖励"""
        return self._instance_dict[instance_id]["reward"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """释放工具实例"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
