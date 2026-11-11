#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置HF_ENDPOINT环境变量，指向HF镜像
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507", cache_dir="${WORKSPACE}")


import sys
import json
import json_repair
import random
import re
import json
import requests
from openai import OpenAI
import yaml
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
import uuid
import os
import logging

import sseclient
import urllib.request
import requests
from typing import Tuple
from typing import List,Any,Dict
from urllib.parse import urlparse
from datetime import datetime
now = datetime.now()
TIME_STR = now.strftime("%Y-%m-%d")

sys.path.append('./')
sys.path.append('../')

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


TOKENS = [
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    # "REPLACE_WITH_YOUR_API_KEY",
    "REPLACE_WITH_YOUR_API_KEY"
]

DATE2WEEK = {
    0: "周一",
    1: "周二",
    2: "周三",
    3: "周四",
    4: "周五",
    5: "周六",
    6: "周日",
}

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSONL文件加载数据

    Args:
        file_path: JSONL文件路径

    Returns:
        List[Dict]: 包含所有数据项的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json_repair.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """
    将数据保存为JSONL格式

    Args:
        data: 要保存的数据列表
        file_path: 保存的文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_prompt(prompt_file_path):
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()
    return prompt

def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def call_model_api_with_messages(messages, model = 'claude-4-sonnet-20250514', token_index = 0, tokens = TOKENS, max_retries = 20):
    """通用模型调用函数"""
    if model == "deepseek-v3-0324" or model == "deepseek-r1":
        url = "https://api.lkeap.cloud.anonymous_org.com/v1/chat/completions"
        auth_token = "REPLACE_WITH_YOUR_API_KEY"
    else:
        if tokens is None:
            raise Exception("没有提供token")
        url = "https://api.example.com/v1/chat/completions"
        token = tokens[token_index % len(tokens)]
        auth_token = token
    payload = {
        'model': model,
        'messages': messages,
        'thinking_enabled': True,
        'thinking_tokens': 4096,
        'max_tokens': 8192
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)

            if response.status_code != 200:
                raise Exception(f"请求失败，状态码: {response.status_code}。响应: {response.text}")

            response_data = response.json()
            # print(response_data)
            result_text = response_data['choices'][0]['message']['content']
            reason_text = response_data['choices'][0]['message'].get('reasoning_content')

            return result_text,reason_text

        except Exception as e:
            logger.error(f"第{retry_count + 1}次尝试失败。错误: {str(e)}")
            retry_count += 1
            sleep_time = 2 * retry_count  # 指数退避策略
            logger.debug(f"{sleep_time}秒后重试...")
            time.sleep(sleep_time)

    return None, None

def call_model_api(prompt, history = [], model = 'claude-4-sonnet-20250514', token_index = 0, tokens = TOKENS, max_retries = 20, system_prompt = None):
    """通用模型调用函数"""
    if model == "deepseek-v3-0324" or model == "deepseek-r1":
        url = "https://api.lkeap.cloud.anonymous_org.com/v1/chat/completions"
        auth_token = "REPLACE_WITH_YOUR_API_KEY"
    else:
        if tokens is None:
            raise Exception("没有提供token")
        url = "https://api.example.com/v1/chat/completions"
        token = tokens[token_index % len(tokens)]
        auth_token = token

    messages = [
            {
                'role': 'system',
                'content': 'you are a helpful assistant' if not system_prompt else system_prompt
            },
    ] + history + [
            {
                'role': 'user',
                'content': prompt
            }
    ]
    # print(messages)

    payload = {
        'model': model,
        'messages': messages,
        'thinking_enabled': True,
        'thinking_tokens': 4096,
        'max_tokens': 8192
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)

            if response.status_code != 200:
                raise Exception(f"请求失败，状态码: {response.status_code}。响应: {response.text}")

            response_data = response.json()
            # print(response_data)
            result_text = response_data['choices'][0]['message']['content']
            reason_text = response_data['choices'][0]['message']['reasoning_content']

            return result_text,reason_text

        except Exception as e:
            logger.error(f"第{retry_count + 1}次尝试失败。错误: {str(e)}")
            retry_count += 1
            sleep_time = 2 ** retry_count  # 指数退避策略
            logger.error(f"{sleep_time}秒后重试...")
            time.sleep(sleep_time)

    return None, None

def date_to_week(date_string: str) -> str:
    try:
        date_object = datetime.datetime.strptime(date_string, '%Y年%m月%d日')
    except Exception:
        date_object = datetime.datetime.strptime(date_string, '%Y-%m-%d')

    week_day = date_object.weekday()
    return DATE2WEEK[week_day]

def call_api(content, history = [], model_name = "deepseek-r1", system_prompt = None) :
    url = "https://api.lkeap.cloud.anonymous_org.com/v1/chat/completions"
    headers = {
        'Authorization': "Bearer REPLACE_WITH_YOUR_API_KEY",
        'Content-Type': 'application/json'
    }
    xinan_label = """{"RedOne":{"Level":"100","Meaning":"红一","Label":20001},
            "Illegality":{"Level":"100","Meaning":"违法","Label":20006},
            "Attack":{"Level":"100","Meaning":"指令攻击","Label":30001},
            "Political":{"Level":"100","Meaning":"政治or红一","Label":20001},
            "Society":{"Level":"100","Meaning":"社会","Label":20004},
            "Privacy":{"Level":"100","Meaning":"隐私","Label":20755},
            "Porn":{"Level":"100","Meaning":"色情","Label":20002},
            "Others":{"Level":"100","Meaning":"其他","Label":21000}}"""
    query_id = f"anonymous_test_{str(uuid.uuid4())}"
    system_prompt = system_prompt if system_prompt else "you are a helpful assistant"
    messages = [
         {"role":"system","content": system_prompt}
    ] + history +[
         {"role": "user", "content": content}
    ]

    data = {
        "model": model_name,
        "stream": True,
        "max_tokens": 4096,
        "query_id": query_id,
        "session_id": str(uuid.uuid4()),
        "uid": "jack_test",
        "messages": messages,
        "repetition_penalty":1,"temperature":0.6,"top_k":150,"top_p":0.7,
        "ext": {
            "xinan_label": xinan_label,
            "xinan_prompt_code": 100,
        },
    }
    response = requests.post(url, json=data, headers=headers, timeout=3000,stream=True)
    ret = ""
    reason = ""
    if response.status_code == 200:
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                continue
            json_data_ = json.loads(event.data)
            if "choices" in json_data_:
                choices = json_data_["choices"]
                content = choices[0]["delta"]["content"]
                reasoning_content = choices[0]["delta"].get("reasoning_content","")
                ret += content
                reason += reasoning_content
        return ret,reason
    else:
        return "",""

def call_api_with_retry(prompt, history = [], model_name = "deepseek-r1", system_prompt = None, retry_cnt = 3) :
    while retry_cnt > 0:
        ret, reason = call_api(prompt, history, model_name, system_prompt = system_prompt)
        if ret:
            return ret, reason
        else:
            print(f"API call failed, retrying... {retry_cnt} attempts left.")
            time.sleep(1)
        retry_cnt -= 1
    return "", ""

def call_llm(*args,**kargs):
    model_name = kargs.get('model_name',"deepseek-r1")
    if model_name == "deepseek-r1":
        return call_api_with_retry(*args,**kargs)
    elif model_name == "claude-4-sonnet-20250514":
        prompt = kargs.get('prompt',"")
        system_prompt = kargs.get('system_prompt',"")
        return call_model_api(prompt, system_prompt=system_prompt)
    elif model_name == "websailer":
        prompt = kargs.get("prompt","")
        system_prompt = kargs.get("system_prompt", "")
        return call_local_model(prompt, system_prompt = system_prompt)


def call_gpt(messages, token_index = 0, tokens = TOKENS, max_retries = 20):
    """通用模型调用函数"""
    if tokens is None:
        raise Exception("没有提供token")
    url = "https://api.example.com/v1/chat/completions"
    token = tokens[token_index % len(tokens)]
    auth_token = token
    model = "gpt-5"
    payload = {
            'model': model,
            'messages': messages,
            'thinking_enabled': True,
            'thinking_tokens': 4096,
            'max_completion_tokens': 8192,
            # "temperature": 1,
        }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)

            if response.status_code != 200:
                raise Exception(f"请求失败，状态码: {response.status_code}。响应: {response.text}")

            response_data = response.json()
            # print(response_data)
            result_text = response_data['choices'][0]['message']['content']
            return result_text

        except Exception as e:
            print(f"第{retry_count + 1}次尝试失败。错误: {str(e)}")
            retry_count += 1
            sleep_time = 2 ** retry_count  # 指数退避策略
            print(f"{sleep_time}秒后重试...")
            time.sleep(sleep_time)

    return None



def get_doc_confidence_new(qd_auth_level):
    try:
        auth = int(qd_auth_level)
    except Exception as e:
        print('auth error ', e)

    if auth == 0 :
        return "极低权威性"
    if auth == 1:
        return "低权威性"
    if auth == 2:
        return "中等权威性"
    if auth == 3:
        return "高权威性"
    if auth == 4:
        return "极高权威性"
    if auth == -1:
        return "暂未找到权威性"
    else:
        return "未知"

def extract_domain(url: str) -> str:
    # 在原始url中解析域名
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain

# 搜索相关
def parse_data(origin):
    try:
        res = json.loads(origin)
    except Exception as e:
        print(f"parse_data执行错误: {e}")
        return [],[]

    if not res:
        return [],[]
    try:
        item = res["passage_group"]
    except Exception as e:
        print(f"parse_data执行错误:  passage_group 解析错误{e}")
        return [],[]
    if not item:
        return [],[]
    docs_info = item["doc_infos"]

    docs = []
    docs_expanded = []
    for d_index, passage_info in enumerate(item["passage_infos"]):
        passage = passage_info["passage_long"] if len(passage_info.get('passage_long','')) != 0 else passage_info["passage"]
        try:
            date_str = passage_info["features"]["date"]
            res_type = passage_info["features"]["res_type"]

            url = passage_info["features"]["url"]
            is_expand = passage_info["features"].get("expand","0")
            if is_expand == "0":
                qd_auth_level = passage_info["features"]["qd_auth_level"]
            else:
                qd_auth_level = passage_info["features"].get("qd_auth_level",-1)

            url = docs_info[d_index].get("url","")
            if res_type == 'wechat':
                url = url + '/' + passage_info["features"]["new_sn"]
            if res_type == 'medical':
                passage = f"[medical] 该网页内容来自医院权威医生\n{passage}"
                qd_auth_level = 4
            title = docs_info[d_index].get("title","")
            domain = extract_domain(url)
            auth_string = get_doc_confidence_new(qd_auth_level)
            item = {"title": title, "content": passage, "url":url, "date": date_str,  "auth": auth_string}

        except Exception as e:
            continue

        if is_expand == "1":
            docs_expanded.append(item)
        else:
            docs.append(item)
    return docs, docs_expanded

def get_docs(all_queries, query):
    url = "https://search.example.com/api"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "request_header": {
            "request_id": "expa",
            "guid": "wuye",
            "session_id": "wuye",
            "enable_full_content": False,
            "enable_debug": True,
            "business_type": "",
            "scene" : {"id":"10086","name":"offline_test"},
        },
        "request_type": 5, # 3表示商业流量 2表示AI研究助手 1表示通用流量 0表示元宝
        "query_group": {
            "origin_query": query,
            "all_queries": all_queries
        },
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
    if response.status_code !=200:
        print(f"response.status_code !=200, status_code:{response.status_code}")
        return [],[]

    docs, docs_expanded = parse_data(response.text)
    response.close()
    return docs,docs_expanded

def call_search(query: str, subqueries: List[str]) -> Tuple[Dict[str, Any]]:
    try:
        # 第一个query的query type是0，后面的是1
        subqueries_dict_list = [{"query": subq, "query_type": min(1,i)} for i,subq in enumerate(subqueries)]
        docs, docs_expanded = get_docs(subqueries_dict_list,query)
        return docs, docs_expanded
    except Exception as e:
        print(f"call_search执行错误: {e}")
        return [],[]

def parse_data_tongyi(origin):
    try:
        res = json.loads(origin)
    except Exception as e:
        print(f"parse_data执行错误: {e}")
        return [],[]

    if not res:
        return [],[]
    try:
        item = res["passage_group"]
    except Exception as e:
        print(f"parse_data执行错误:  passage_group 解析错误{e}")
        return [],[]
    if not item:
        return [],[]
    docs_info = item["doc_infos"]

    docs = []
    docs_expanded = []
    for d_index, passage_info in enumerate(item["passage_infos"]):
        passage = passage_info["passage"]
        passage_long = passage_info["passage_long"]
        try:
            date_str = passage_info["features"]["date"]
            res_type = passage_info["features"]["res_type"]

            url = passage_info["features"]["url"]
            is_expand = passage_info["features"].get("expand","0")
            if is_expand == "0":
                qd_auth_level = passage_info["features"]["qd_auth_level"]
            else:
                qd_auth_level = passage_info["features"].get("qd_auth_level",-1)

            url = docs_info[d_index].get("url","")
            if res_type == 'wechat':
                url = url + '/' + passage_info["features"]["new_sn"]
            if res_type == 'medical':
                passage = f"[medical] 该网页内容来自医院权威医生\n{passage}"
                qd_auth_level = 4
            title = docs_info[d_index].get("title","")
            domain = extract_domain(url)
            auth_string = get_doc_confidence_new(qd_auth_level)
            item = {"title": title, "content": passage, "url":url, "date": date_str,  "auth": auth_string, "content_long": passage_long}

        except Exception as e:
            continue

        if is_expand == "1":
            docs_expanded.append(item)
        else:
            docs.append(item)
    return docs, docs_expanded

def get_docs_tongyi(query: str, subqueries: List[str]) -> Tuple[Dict[str,Any]]:
    url = "https://search.example.com/api"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "request_header": {
            "request_id": "expa",
            "guid": "wuye",
            "session_id": "wuye",
            "enable_full_content": False,
            "enable_debug": True,
            "business_type": "",
            "scene" : {"id":"10086","name":"offline_test"},
        },
        "request_type": 5, # 3表示商业流量 2表示AI研究助手 1表示通用流量 0表示元宝
        "query_group": {
            "origin_query": query,
            "all_queries": all_queries
        },
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
    if response.status_code !=200:
        print(f"response.status_code !=200, status_code:{response.status_code}")
        return [],[]

    docs, docs_expanded = parse_data_tongyi(response.text)
    response.close()
    return docs,docs_expanded

def call_search_tongyi(query: str, subqueries: List[str]) -> Tuple[Dict[str, Any]]:
    try:
        # 第一个query的query type是0，后面的是1
        subqueries_dict_list = [{"query": subq, "query_type": min(1,i)} for i,subq in enumerate(subqueries)]
        docs, docs_expanded = get_docs_tongyi(subqueries_dict_list,query)
        return docs, docs_expanded
    except Exception as e:
        print(f"call_search执行错误: {e}")
        return [],[]

# api调度：
class PriorityGreedyDispatcher:
    def __init__(self, tokens, qpm_mapping=None):
        self.lock = threading.RLock()


        self.window_size = 10


        self.api_groups = self._parse_tokens(tokens)


        self.qpm_mapping = qpm_mapping or {
            '2125': 60,
            '3892': 10,
            '4257': 10
        }


        self.initial_wait_mapping = {
            '2125': 1.0,
            '3892': 6.0,
            '4257': 6.0
        }


        self.response_time_windows = {}


        self.api_profiles = {}
        self._init_api_profiles()


        self.token_buckets = {}
        self._init_token_buckets()


        self.priority_list = []
        self.priority_list_last_update = datetime.now()
        self.priority_update_interval = 30


        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0


        self.api_error_counts = defaultdict(int)
        self.api_lockout_until = defaultdict(lambda: datetime.now())

        print(f"初始化优先级贪心调度器:")
        for api_id, profile in self.api_profiles.items():
            print(f"  API {api_id}: QPM={profile['qpm']}, 初始等待时间={self.initial_wait_mapping[api_id]}秒")

    def _parse_tokens(self, tokens):
        """解析tokens并按API分组"""

        groups = defaultdict(list)
        for token in tokens:
            if '@' in token:
                api_id = token.split('@')[1]
                groups[api_id].append(token)
            else:
                groups['default'].append(token)
        return dict(groups)

    def _init_api_profiles(self):
        """初始化API档案"""
        for api_id, tokens in self.api_groups.items():
            if api_id in self.qpm_mapping:
                self.api_profiles[api_id] = {
                    'api_id': api_id,
                    'tokens': tokens,
                    'qpm': self.qpm_mapping[api_id],
                    'qps': self.qpm_mapping[api_id] / 60.0,
                    'avg_response_time': 5.0,
                    'current_token_index': 0,
                    # 'last_request_time': datetime.now() - timedelta(minutes=1)
                }

                self.response_time_windows[api_id] = deque(maxlen=self.window_size)

    def _init_token_buckets(self):
        """初始化令牌桶限流器 - 标准实现"""
        for api_id, profile in self.api_profiles.items():
            qpm = profile['qpm']


            if qpm <= 10:
                burst_limit = 2.0
            else:
                burst_limit = 5.0


            initial_tokens = burst_limit

            self.token_buckets[api_id] = {
                'tokens': initial_tokens,
                'max_tokens': burst_limit,
                'last_refill': datetime.now(),
                'refill_rate': qpm / 60.0
            }
            print(f"初始化API {api_id}: 容量={burst_limit}, QPM={qpm}, 补充速率={qpm/60.0:.3f}/秒")

    def _refill_token_bucket(self, api_id):
        """补充令牌桶 - 标准实现"""
        bucket = self.token_buckets[api_id]
        now = datetime.now()
        time_passed = (now - bucket['last_refill']).total_seconds()


        if time_passed > 0:
            tokens_to_add = time_passed * bucket['refill_rate']


            bucket['tokens'] = min(bucket['max_tokens'], bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now

    def _can_api_make_request(self, api_id):
        """检查API是否可以发起请求 - 标准实现"""
        now = datetime.now()


        if now < self.api_lockout_until[api_id]:
            return False


        self._refill_token_bucket(api_id)


        return self.token_buckets[api_id]['tokens'] >= 1

    def _consume_token(self, api_id):
        """消费令牌 - 标准实现"""
        if self._can_api_make_request(api_id):

            self.token_buckets[api_id]['tokens'] -= 1
            return True
        return False

    def _update_response_time(self, api_id, response_time):
        """更新API的平均响应时间（滑动窗口）"""
        with self.lock:
            window = self.response_time_windows[api_id]
            window.append(response_time)


            if window:
                avg_time = sum(window) / len(window)
                self.api_profiles[api_id]['avg_response_time'] = avg_time

    def _generate_priority_list(self):
        """生成优先级列表 - 核心算法"""
        with self.lock:
            available_apis = []

            for api_id, profile in self.api_profiles.items():
                if self._can_api_make_request(api_id):
                    available_apis.append({
                        'api_id': api_id,
                        'response_time': profile['avg_response_time'],
                        'qpm': profile['qpm']
                    })




            available_apis.sort(key=lambda x: (x['response_time'], -x['qpm']))

            self.priority_list = [api['api_id'] for api in available_apis]
            self.priority_list_last_update = datetime.now()

            return self.priority_list.copy()

    def _get_priority_list(self):
        """获取当前优先级列表（带缓存） - 修复版本"""
        now = datetime.now()
        # 核心修复：如果优先级列表为空，或缓存已过期，则强制重新生成
        if not self.priority_list or (now - self.priority_list_last_update).total_seconds() > self.priority_update_interval:
            return self._generate_priority_list()

        return self.priority_list.copy()

    def get_best_available_api(self, thread_id=None):
        """获取最佳可用API（工作线程调度算法）"""
        if thread_id is None:
            thread_id = threading.current_thread().name

        with self.lock:
            priority_list = self._get_priority_list()

            # 按优先级顺序尝试获取API
            for api_id in priority_list:
                if self._consume_token(api_id):
                    # 获取当前token
                    profile = self.api_profiles[api_id]
                    tokens = profile['tokens']
                    current_index = profile['current_token_index']
                    token = tokens[current_index]

                    # 更新token索引
                    profile['current_token_index'] = (current_index + 1) % len(tokens)
                    profile['last_request_time'] = datetime.now()

                    return api_id, token

            return None, None

    def wait_for_available_api(self, max_wait_time=1800, thread_id=None):
        """等待可用API - 修复版本"""
        if thread_id is None:
            thread_id = threading.current_thread().name

        start_time = datetime.now()
        wait_intervals = [1, 2, 5, 10, 30, 60]  # 渐进式等待
        interval_index = 0

        while True:
            if (datetime.now() - start_time).total_seconds() > max_wait_time:
                raise Exception(f"[{thread_id}] 等待可用API超时（{max_wait_time}秒）")

            api_id, token = self.get_best_available_api(thread_id)
            if api_id and token:
                return api_id, token

            # 改进：为等待时间添加抖动，避免“惊群效应”
            base_wait_time = wait_intervals[min(interval_index, len(wait_intervals) - 1)]
            wait_time = base_wait_time * random.uniform(0.8, 1.2)  # 添加±20%的抖动
            interval_index += 1

            # print(f"[{thread_id}] 所有API都不可用，等待 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)

    def record_request(self, api_id, thread_id=None):
        """记录请求"""
        if thread_id is None:
            thread_id = threading.current_thread().name

        with self.lock:
            self.total_requests += 1

    def record_success(self, api_id, response_time, thread_id=None):
        """记录成功请求"""
        if thread_id is None:
            thread_id = threading.current_thread().name

        with self.lock:
            self.successful_requests += 1
            self._update_response_time(api_id, response_time)

            # 重置错误计数和锁死状态
            if api_id in self.api_error_counts:
                del self.api_error_counts[api_id]
            if api_id in self.api_lockout_until:
                del self.api_lockout_until[api_id]

    def record_error(self, api_id, error_type="general", thread_id=None):
        """记录错误并实施惩罚"""
        if thread_id is None:
            thread_id = threading.current_thread().name

        with self.lock:
            self.failed_requests += 1
            self.api_error_counts[api_id] += 1
            error_count = self.api_error_counts[api_id]

            # 根据QPM设置基础等待时间
            base_wait = self.initial_wait_mapping.get(api_id, 5.0)

            if error_type == "rate_limit":
                # 限流错误：指数退避 + 抖动
                backoff_multiplier = 2 ** (error_count - 1)
                cooldown_seconds = min(base_wait * backoff_multiplier, 300)  # 最多5分钟

                # 添加抖动 (±20%)
                jitter = random.uniform(0.8, 1.2)
                cooldown_seconds *= jitter
            else:
                # 其他错误：线性增长
                cooldown_seconds = min(base_wait * error_count, 120)  # 最多2分钟

            self.api_lockout_until[api_id] = datetime.now() + timedelta(seconds=cooldown_seconds)
            print(f"[{thread_id}] API {api_id} 发生{error_type}错误(第{error_count}次)，锁死 {cooldown_seconds:.1f} 秒")

    def get_status_summary(self):
        """获取状态摘要"""
        with self.lock:
            now = datetime.now()
            status = {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'current_priority_list': self._get_priority_list(),
                'apis': {}
            }

            for api_id, profile in self.api_profiles.items():
                self._refill_token_bucket(api_id)
                bucket = self.token_buckets[api_id]
                is_locked = now < self.api_lockout_until[api_id]

                status['apis'][api_id] = {
                    'qpm': profile['qpm'],
                    'avg_response_time': profile['avg_response_time'],
                    'available_tokens': int(bucket['tokens']),
                    'max_tokens': bucket['max_tokens'],
                    'is_locked': is_locked,
                    'error_count': self.api_error_counts[api_id],
                    'lockout_remaining': max(0, (self.api_lockout_until[api_id] - now).total_seconds()) if is_locked else 0
                }

            return status

priority_dispatcher = PriorityGreedyDispatcher(TOKENS)


def call_dsv31_terminus(messages, enable_thinking=False):
    """使用优先级贪心调度策略的LLM调用函数"""
    model = "deepseek-v3.1-terminus"
    url = "https://api.example.com/v1/chat/completions"

    thread_id = threading.current_thread().name
    max_retries = 5
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        api_id = None
        token = None
        request_start_time = datetime.now()

        try:
            # 获取最佳可用API
            api_id, token = priority_dispatcher.wait_for_available_api(
                max_wait_time=1800, thread_id=thread_id
            )

            # 记录请求
            priority_dispatcher.record_request(api_id, thread_id)

            payload = {
                'model': model,
                'messages': messages,
                'thinking_enabled': enable_thinking,
                'thinking_tokens': 8192,
                'max_tokens': 8192
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}'
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            response_time = (datetime.now() - request_start_time).total_seconds()

            if response.status_code == 429:  # 限流错误
                priority_dispatcher.record_error(api_id, "rate_limit", thread_id)
                retry_count += 1

                # 智能重试：带抖动的指数退避
                base_wait = priority_dispatcher.initial_wait_mapping.get(api_id, 5.0)
                backoff_time = min(base_wait * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter

                print(f"[{thread_id}] API {api_id} 限流，退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                error_msg = f"请求失败，状态码: {response.status_code}。响应: {response.text}"
                priority_dispatcher.record_error(api_id, "http_error", thread_id)
                raise Exception(error_msg)

            # 请求成功
            priority_dispatcher.record_success(api_id, response_time, thread_id)

            response_data = response.json()
            answer = response_data['choices'][0]['message']['content']
            reason = ''
            if enable_thinking:
                reason = response_data['choices'][0]['message']['reasoning_content']
            return answer, reason

        except requests.exceptions.Timeout:
            if api_id:
                priority_dispatcher.record_error(api_id, "timeout", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 超时的指数退避
                backoff_time = min(10 * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 超时退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = "请求超时"

        except Exception as e:
            if api_id:
                priority_dispatcher.record_error(api_id, "general", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 一般错误的指数退避
                backoff_time = min(8 * (2 ** (retry_count - 1)), 120)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 错误退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = str(e)

    raise Exception(f"[{thread_id}] 所有重试都失败了，最后的错误: {last_error}")

def call_dsv31_terminus_thinking(messages, enable_thinking=True):
    """使用优先级贪心调度策略的LLM调用函数"""
    model = "deepseek-v3.1-terminus"
    url = "https://api.example.com/v1/chat/completions"

    thread_id = threading.current_thread().name
    max_retries = 5
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        api_id = None
        token = None
        request_start_time = datetime.now()

        try:
            # 获取最佳可用API
            api_id, token = priority_dispatcher.wait_for_available_api(
                max_wait_time=1800, thread_id=thread_id
            )

            # 记录请求
            priority_dispatcher.record_request(api_id, thread_id)

            payload = {
                'model': model,
                'messages': messages,
                'thinking_enabled': enable_thinking,
                'thinking_tokens': 8192,
                'max_tokens': 8192
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}'
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            response_time = (datetime.now() - request_start_time).total_seconds()

            if response.status_code == 429:  # 限流错误
                priority_dispatcher.record_error(api_id, "rate_limit", thread_id)
                retry_count += 1

                # 智能重试：带抖动的指数退避
                base_wait = priority_dispatcher.initial_wait_mapping.get(api_id, 5.0)
                backoff_time = min(base_wait * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter

                print(f"[{thread_id}] API {api_id} 限流，退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                error_msg = f"请求失败，状态码: {response.status_code}。响应: {response.text}"
                priority_dispatcher.record_error(api_id, "http_error", thread_id)
                raise Exception(error_msg)

            # 请求成功
            priority_dispatcher.record_success(api_id, response_time, thread_id)

            response_data = response.json()
            result_text = response_data['choices'][0]['message']['content']
            result_reason = response_data['choices'][0]['message']['reasoning_content']
            return result_text, result_reason

        except requests.exceptions.Timeout:
            if api_id:
                priority_dispatcher.record_error(api_id, "timeout", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 超时的指数退避
                backoff_time = min(10 * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 超时退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = "请求超时"

        except Exception as e:
            if api_id:
                priority_dispatcher.record_error(api_id, "general", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 一般错误的指数退避
                backoff_time = min(8 * (2 ** (retry_count - 1)), 120)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 错误退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = str(e)

    raise Exception(f"[{thread_id}] 所有重试都失败了，最后的错误: {last_error}")

def call_dsv31(messages, enable_thinking=False):
    """使用优先级贪心调度策略的LLM调用函数"""
    model = "deepseek-v3.1"
    url = "https://api.example.com/v1/chat/completions"

    thread_id = threading.current_thread().name
    max_retries = 5
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        api_id = None
        token = None
        request_start_time = datetime.now()

        try:
            # 获取最佳可用API
            api_id, token = priority_dispatcher.wait_for_available_api(
                max_wait_time=1800, thread_id=thread_id
            )

            # 记录请求
            priority_dispatcher.record_request(api_id, thread_id)

            payload = {
                'model': model,
                'messages': messages,
                'thinking_enabled': enable_thinking,
                'thinking_tokens': 8192,
                'max_tokens': 8192
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}'
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            response_time = (datetime.now() - request_start_time).total_seconds()

            if response.status_code == 429:  # 限流错误
                priority_dispatcher.record_error(api_id, "rate_limit", thread_id)
                retry_count += 1

                # 智能重试：带抖动的指数退避
                base_wait = priority_dispatcher.initial_wait_mapping.get(api_id, 5.0)
                backoff_time = min(base_wait * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter

                print(f"[{thread_id}] API {api_id} 限流，退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                error_msg = f"请求失败，状态码: {response.status_code}。响应: {response.text}"
                priority_dispatcher.record_error(api_id, "http_error", thread_id)
                raise Exception(error_msg)

            # 请求成功
            priority_dispatcher.record_success(api_id, response_time, thread_id)

            response_data = response.json()
            answer = response_data['choices'][0]['message']['content']

            return answer

        except requests.exceptions.Timeout:
            if api_id:
                priority_dispatcher.record_error(api_id, "timeout", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 超时的指数退避
                backoff_time = min(10 * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 超时退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = "请求超时"

        except Exception as e:
            if api_id:
                priority_dispatcher.record_error(api_id, "general", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 一般错误的指数退避
                backoff_time = min(8 * (2 ** (retry_count - 1)), 120)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 错误退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = str(e)

    raise Exception(f"[{thread_id}] 所有重试都失败了，最后的错误: {last_error}")

def call_dsv31_thinking(messages):
    """使用优先级贪心调度策略的LLM调用函数"""
    model = "deepseek-v3.1"
    url = "https://api.example.com/v1/chat/completions"

    thread_id = threading.current_thread().name
    max_retries = 5
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        api_id = None
        token = None
        request_start_time = datetime.now()

        try:
            # 获取最佳可用API
            api_id, token = priority_dispatcher.wait_for_available_api(
                max_wait_time=1800, thread_id=thread_id
            )

            # 记录请求
            priority_dispatcher.record_request(api_id, thread_id)

            payload = {
                'model': model,
                'messages': messages,
                'thinking_enabled': True,
                'thinking_tokens': 8192,
                'max_tokens': 8192
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}'
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            response_time = (datetime.now() - request_start_time).total_seconds()

            if response.status_code == 429:  # 限流错误
                priority_dispatcher.record_error(api_id, "rate_limit", thread_id)
                retry_count += 1

                # 智能重试：带抖动的指数退避
                base_wait = priority_dispatcher.initial_wait_mapping.get(api_id, 5.0)
                backoff_time = min(base_wait * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter

                print(f"[{thread_id}] API {api_id} 限流，退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                error_msg = f"请求失败，状态码: {response.status_code}。响应: {response.text}"
                priority_dispatcher.record_error(api_id, "http_error", thread_id)
                raise Exception(error_msg)

            # 请求成功
            priority_dispatcher.record_success(api_id, response_time, thread_id)

            response_data = response.json()
            answer = response_data['choices'][0]['message']['content']
            reason = response_data['choices'][0]['message']['reasoning_content']
            return answer, reason

        except requests.exceptions.Timeout:
            if api_id:
                priority_dispatcher.record_error(api_id, "timeout", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 超时的指数退避
                backoff_time = min(10 * (2 ** (retry_count - 1)), 240)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 超时退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = "请求超时"

        except Exception as e:
            if api_id:
                priority_dispatcher.record_error(api_id, "general", thread_id)

            retry_count += 1
            if retry_count < max_retries:
                # 一般错误的指数退避
                backoff_time = min(8 * (2 ** (retry_count - 1)), 120)
                jitter = random.uniform(0.8, 1.2)
                wait_time = backoff_time * jitter
                print(f"[{thread_id}] 错误退避等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            last_error = str(e)

    raise Exception(f"[{thread_id}] 所有重试都失败了，最后的错误: {last_error}")

def build_history_1(prev_query_list):
    if not prev_query_list:
        return "暂无历史"
    separator = "\n\n" + "="*20+"\n\n"
    return separator.join(prev_query_list)

def call_r1(content, history = [], model_name = "deepseek-r1", system_prompt = None) :
    url = "https://api.lkeap.cloud.anonymous_org.com/v1/chat/completions"
    headers = {
        'Authorization': "Bearer REPLACE_WITH_YOUR_API_KEY",
        'Content-Type': 'application/json'
    }
    xinan_label = """{"RedOne":{"Level":"100","Meaning":"红一","Label":20001},
            "Illegality":{"Level":"100","Meaning":"违法","Label":20006},
            "Attack":{"Level":"100","Meaning":"指令攻击","Label":30001},
            "Political":{"Level":"100","Meaning":"政治or红一","Label":20001},
            "Society":{"Level":"100","Meaning":"社会","Label":20004},
            "Privacy":{"Level":"100","Meaning":"隐私","Label":20755},
            "Porn":{"Level":"100","Meaning":"色情","Label":20002},
            "Others":{"Level":"100","Meaning":"其他","Label":21000}}"""
    query_id = f"anonymous_test_{str(uuid.uuid4())}"
    system_prompt = system_prompt if system_prompt else "you are a helpful assistant"
    messages = [
         {"role":"system","content": system_prompt}
    ] + history +[
         {"role": "user", "content": content}
    ]

    data = {
        "model": model_name,
        "stream": True,
        "max_tokens": 4096,
        "query_id": query_id,
        "session_id": str(uuid.uuid4()),
        "uid": "jack_test",
        "messages": messages,
        "repetition_penalty":1,"temperature":0.6,"top_k":150,"top_p":0.7,
        "ext": {
            "xinan_label": xinan_label,
            "xinan_prompt_code": 100,
        },
    }
    response = requests.post(url, json=data, headers=headers, timeout=3000,stream=True)
    ret = ""
    reason = ""
    if response.status_code == 200:
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                continue
            json_data_ = json.loads(event.data)
            if "choices" in json_data_:
                choices = json_data_["choices"]
                content = choices[0]["delta"]["content"]
                reasoning_content = choices[0]["delta"].get("reasoning_content","")
                ret += content
                reason += reasoning_content
        return ret,reason
    else:
        return "",""

def parse_reason_dsv31(response_text, is_first_round=True):
    """
    解析返回结果中的思考过程
    Args:
        response_text: 响应文本
        is_first_round: 是否为首轮，默认为True
    """
    if is_first_round:
        # 首轮：提取 <｜search begin｜> 和 <｜tool calls begin｜> 之间的内容
        pattern = r'<｜search▁begin｜>\s*(.*?)\s*<｜tool▁calls▁begin｜>'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    else:
        # 非首轮：提取开头到 <｜tool calls begin｜> 之间的内容
        pattern = r'^(.*?)\s*<｜tool▁calls▁begin｜>'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            raw_reason = match.group(1).strip()
            reason = raw_reason.replace('<｜search▁begin｜>', '')
            return reason.strip()

    return ""

def parse_queries_dsv31(response_text):
    """
    解析返回结果中的查询列表
    从 questions 字段中提取用 || 分隔的查询
    """
    # 提取 tool_call 中的 questions 参数
    pattern = r'<｜tool▁call▁begin｜>search<｜tool▁sep｜>({.*?})<｜tool▁call▁end｜>'
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        try:
            tool_args = json.loads(match.group(1))
            questions = tool_args.get('questions', '')
            # 按 || 分割查询
            queries = [q.strip() for q in questions.split('||') if q.strip()]
            return queries
        except json.JSONDecodeError:
            print("解析tool参数失败")
            print(response_text)
            return []
    return []

def parse_call_llm_dsv31(response_text):
    """
    解析返回结果中的call_llm调用
    从 doc_list 字段中提取文档列表
    """
    # 提取 tool_call 中的 doc_list 参数
    pattern = r'<｜tool▁call▁begin｜>call_llm<｜tool▁sep｜>({.*?})<｜tool▁call▁end｜>'
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        try:
            tool_args = json.loads(match.group(1))
            doc_list = tool_args.get('doc_list', [])

            # 验证doc_list格式并提取文档信息
            documents = []
            for doc in doc_list:
                if isinstance(doc, dict) and 'index' in doc and 'title' in doc:
                    documents.append({
                        'index': doc['index'],
                        'title': doc['title']
                    })
                else:
                    print(f"无效的文档格式: {doc}")

            return documents
        except json.JSONDecodeError:
            print("解析call_llm参数失败")
            return []
    return []

def parse_final_answer_dsv31(response_text):
    """
    解析最终答案
    提取 <｜search▁end｜> 前后的内容
    返回: {'reason': str, 'answer': str}
    """
    # 查找 <｜search▁end｜> 的位置
    end_marker = '<｜search▁end｜>'
    end_pos = response_text.find(end_marker)

    if end_pos == -1:
        return {'reason': '', 'answer': ''}

    reason_text = response_text[:end_pos].strip()

    # 提取后面的内容作为answer
    answer_start = end_pos + len(end_marker)
    answer_text = response_text[answer_start:].strip()

    return {
        'reason': reason_text,
        'answer': answer_text
    }

def parse_reason_qwen3(response_text):
    """
    解析qwen3的思考过程
    """
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def parse_queries_dsv31(response_text):
    """
    解析返回结果中的查询列表
    从 questions 字段中提取用 || 分隔的查询
    """
    # 提取 tool_call 中的 questions 参数
    pattern = r'<｜tool▁call▁begin｜>search<｜tool▁sep｜>({.*?})<｜tool▁call▁end｜>'
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        try:
            tool_args = json.loads(match.group(1))
            questions = tool_args.get('questions', '')
            # 按 || 分割查询
            queries = [q.strip() for q in questions.split('||') if q.strip()]
            return queries
        except json.JSONDecodeError:
            print("解析tool参数失败")
            print(response_text)
            return []
    return []


def parse_urls_dsv31(response_text):
    """
    解析返回结果中的URL列表和目标
    从visit工具调用中提取url和goal字段
    """
    # 提取 tool_call 中的 url 和 goal 参数
    pattern = r'<｜tool▁call▁begin｜>visit<｜tool▁sep｜>({.*?})<｜tool▁call▁end｜>'
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        try:
            tool_args = json.loads(match.group(1))
            url = tool_args.get('url', [])
            goal = tool_args.get('goal', '')

            # 确保url是列表格式
            if isinstance(url, str):
                urls = [url]
            elif isinstance(url, list):
                urls = url
            else:
                urls = []

            return {
                'urls': urls,
                'goal': goal
            }
        except json.JSONDecodeError:
            print("解析visit tool参数失败")
            print(response_text)
            return {'urls': [], 'goal': ''}

    return {'urls': [], 'goal': ''}

def parse_tools_dsv31(response_text):
    """
    解析返回结果中的工具调用(search和visit)
    返回: {
        'tool_type': 'search' | 'visit' | None,
        'queries': List[str],  # 当tool_type为search时
        'urls': List[str],     # 当tool_type为visit时
        'goal': str            # 当tool_type为visit时
    }
    """
    # 首先尝试匹配search工具
    search_pattern = r'<｜tool▁call▁begin｜>search<｜tool▁sep｜>({.*?})<｜tool▁call▁end｜>'
    search_match = re.search(search_pattern, response_text, re.DOTALL)

    if search_match:
        try:
            tool_args = json.loads(search_match.group(1))
            questions = tool_args.get('questions', '')
            # 按 || 分割查询
            queries = [q.strip() for q in questions.split('||') if q.strip()]
            return {
                'tool_type': 'search',
                'queries': queries,
                'urls': [],
                'goal': ''
            }
        except json.JSONDecodeError:
            print("解析search tool参数失败")
            print(response_text)

    # 尝试匹配visit工具
    visit_pattern = r'<｜tool▁call▁begin｜>visit<｜tool▁sep｜>({.*?})<｜tool▁call▁end｜>'
    visit_match = re.search(visit_pattern, response_text, re.DOTALL)

    if visit_match:
        try:
            tool_args = json.loads(visit_match.group(1))
            url = tool_args.get('url', [])
            goal = tool_args.get('goal', '')

            # 确保url是列表格式
            if isinstance(url, str):
                urls = [url]
            elif isinstance(url, list):
                urls = url
            else:
                urls = []

            return {
                'tool_type': 'visit',
                'queries': [],
                'urls': urls,
                'goal': goal
            }
        except json.JSONDecodeError:
            print("解析visit tool参数失败")
            print(response_text)

    # 没有匹配到任何工具
    return {
        'tool_type': None,
        'queries': [],
        'urls': [],
        'goal': ''
    }

def parse_tools_tongyi(response_text):
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, response_text, re.DOTALL)
    if not match:
        return None, None, True

    try:
        raw_match = match.group(1).strip()
        json_data = json_repair.loads(raw_match)
        function_type = json_data.get("name")
        arguments = json_data.get("arguments")

        if not function_type or not arguments:
            return None, None, True

        if function_type == "search":
            if "query" not in arguments:
                print("query 字段不存在")
                return function_type, None, True
            return function_type, arguments, False

        elif function_type == "visit":
            if "url" not in arguments or "goal" not in arguments:
                print("url 或 goal 字段不存在")
                return function_type, None, True
            return function_type, arguments, False

        elif function_type in ["PythonInterpreter", "google_scholar", "parse_file"]:
            return function_type, None, True

        else:
            return function_type, None, True

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return None, None, True

def parse_queries_qwen3(response_text):
    """
    解析qwen3的查询列表
    """
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            raw_match = match.group(1).strip()
            json_data = json_repair.loads(raw_match)
            if "arguments" in json_data:
                arguments = json_data["arguments"]
                if "query_list" in arguments:
                    query_list = arguments["query_list"]
                    return query_list
                else:
                    print("query_list 字段不存在")
                    return []
            else:
                print("arguments 字段不存在")
                return []

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            return []
    return []

def parse_tool_call_qwen3(response_text):
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            raw_match = match.group(1).strip()
            json_data = json_repair.loads(raw_match)
            assert 'name' in json_data, 'tool name not in response'
            assert 'arguments' in json_data, 'arguments not in response'
            return json_data

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            return {}
    return {}


def extract_history_queries_from_history(turn_history: List[dict]) -> List[str]:
    """
    从对话历史中提取所有查询
    """
    history_queries = []
    for item in turn_history:
        if item.get('role') == 'user':
            query = item.get('content', '')
            history_queries.append(query)
    return history_queries

def extract_history_queries_answers_from_history(turn_history: List[dict]) -> List[dict]:
    """
    从对话历史中提取所有查询和答案
    """
    history_queries_answers = []
    for item in turn_history:
        if item.get('role') == 'user':
            query = item.get('content', '')
            history_queries_answers.append({'query': query, 'answer': ''})
        elif item.get('role') == 'assistant':
            answer = item.get('content', '')
            if history_queries_answers:
                history_queries_answers[-1]['answer'] = answer
    return history_queries_answers

def deduplicate_and_fill_docs(docs_candidate: List[dict], docs_hashmap: dict, total_budget: int = 24576) -> Tuple[List[dict], dict, dict]:
    """
    对候选文档进行去重，使用total_budget控制总长度
    每次计划放进来就做一个判断，如果超过total_budget就不放进来，如果不超过就放进来
    """
    if len(docs_candidate) == 0:
        return [], docs_hashmap, {
            'original_count': 0,
            'added_count': 0,
            'removed_duplicates': 0,
            'final_count': 0,
            'total_length': 0
        }

    final_docs = []
    current_total_length = 0
    removed_duplicates = 0
    added_count = 0

    for doc in docs_candidate:
        # 检查是否重复
        doc_key = f"{doc.get('title', '')}__{doc.get('domain', '')}__{doc.get('content', '')[:100]}"
        if doc_key in docs_hashmap:
            removed_duplicates += 1
            continue

        # 计算当前文档的长度
        doc_length = len(doc.get('content', ''))

        # 判断是否超过total_budget
        if current_total_length + doc_length <= total_budget:
            # 不超过budget，可以放进来
            docs_hashmap[doc_key] = True
            final_docs.append(doc)
            current_total_length += doc_length
            added_count += 1
        else:
            # 超过budget，不放进来
            break

    stats = {
        'original_count': len(docs_candidate),
        'added_count': added_count,
        'removed_duplicates': removed_duplicates,
        'final_count': len(final_docs),
        'total_length': current_total_length
    }

    return final_docs, docs_hashmap, stats


def deduplicate_and_fill_docs_as(
    results: Dict[str, Tuple[List[Any], List[Any]]],
    docs_hashmap: dict,
    total_budget: int = 24576
) -> Tuple[Dict[str, Tuple[List[Any], List[Any]]], dict, dict]:
    """
    对每个subquery的docs进行去重，返回去重后的results

    Args:
        results: {subquery: (docs, docs_expanded)} 格式的搜索结果
        docs_hashmap: 用于去重的hashmap
        total_budget: 每个subquery的总长度预算

    Returns:
        Tuple[去重后的results, 更新后的docs_hashmap, 统计信息]
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
        # 合并 docs 和 docs_expanded 进行去重
        all_docs = docs + docs_expanded

        if len(all_docs) == 0:
            deduplicated_results[subquery] = ([], [])
            continue

        final_docs = []
        final_docs_expanded = []
        current_total_length = 0
        removed_duplicates = 0
        added_count = 0

        # 先处理 docs
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

        # 再处理 docs_expanded
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

        # 累计统计
        total_stats['original_count'] += len(all_docs)
        total_stats['added_count'] += added_count
        total_stats['removed_duplicates'] += removed_duplicates
        total_stats['final_count'] += len(final_docs) + len(final_docs_expanded)
        total_stats['total_length'] += current_total_length

    return deduplicated_results, docs_hashmap, total_stats

def calculate_search_budget(step: int, current_trajectory_length: int, max_total_length: int = 60000, reserved_length: int = 5000) -> int:
    """
    根据当前轮次和已有轨迹长度计算搜索预算
    """
    available_budget = max_total_length - current_trajectory_length - reserved_length

    # 如果可用预算不足，返回最小预算
    if available_budget < 1000:
        return 1000

    # 根据搜索轮次分配预算（递减策略）
    budget_ratios = {
        1: 0.35,  # 第1轮：35%
        2: 0.25,  # 第2轮：25%
        3: 0.20,  # 第3轮：20%
        4: 0.15,  # 第4轮：15%
        5: 0.05,  # 第5轮：5%
    }

    ratio = budget_ratios.get(step, 0.05)
    budget = int(available_budget * ratio)

    # 确保最小预算
    return max(budget, 1000)

def has_tool_call_dsv31(response_text):
    if '<｜tool▁calls▁begin｜>' in response_text and '<｜tool▁calls▁end｜>' in response_text and '<｜search▁end｜>' not in response_text:
        return True
    return False

def has_tool_call_qwen3(response_text):
    if '<tool_call>' in response_text and '</tool_call>' in response_text:
        return True
    return False



def extract_citation_indexes(answer_text, unique=True, sorted_result=True):
    """
    从最终回答中解析引用文档索引

    Args:
        answer_text (str): 包含引用标记的回答文本
        unique (bool): 是否去重，默认True
        sorted_result (bool): 是否对结果排序，默认True

    Returns:
        list: 引用索引列表，例如 [0, 1, 2, 3, 6, 8, 13, 18]
    """
    # 匹配 [citation:数字] 的模式
    citation_pattern = r'\[citation:(\d+)\]'

    # 找到所有匹配项
    matches = re.findall(citation_pattern, answer_text)

    # 转换为整数
    citation_indexes = [int(idx) for idx in matches]

    # 根据参数处理结果
    if unique:
        citation_indexes = list(set(citation_indexes))

    if sorted_result:
        citation_indexes.sort()

    return citation_indexes

def call_dsv31_vendor_b(messages, max_retries=10, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
    """
    调用（匿名机构）云DeepSeek V3.1模型，带有指数退让重试机制

    Args:
        messages: 消息列表
        max_retries: 最大重试次数，默认10次（429错误除外，会无限重试）
        base_delay: 基础延迟时间（秒），默认1秒
        max_delay: 最大延迟时间（秒），默认60秒
        backoff_factor: 退让因子，默认2.0

    Returns:
        str: 模型返回的内容
    """
    import time
    import random

    attempt = 0
    while True:
        try:
            client = OpenAI(
                # 请用知识引擎原子能力API Key将下行替换为：api_key="sk-xxx",
                api_key="REPLACE_WITH_YOUR_API_KEY", # 如何获取API Key：https://cloud.anonymous_org.com/document/product/1772/115970
                base_url="https://api.lkeap.cloud.anonymous_org.com/v1",
            )
            completion = client.chat.completions.create(
                model="deepseek-v3.1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=messages,
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                },
                stream=False
            )

            content = completion.choices[0].message.content
            return content

        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")

            # 检查是否是429错误（速率限制）
            is_rate_limit_error = False
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit_error = True
            elif "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                is_rate_limit_error = True

            # 如果不是429错误且达到最大重试次数，返回空字符串
            if not is_rate_limit_error and attempt > max_retries:
                print(f"Non-rate-limit error occurred {max_retries} times. Returning empty content.")
                return ""

            # 计算退让延迟时间：基础延迟 * (退让因子 ^ 尝试次数) + 随机抖动
            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            # 添加随机抖动，避免多个请求同时重试
            jitter = random.uniform(0, delay * 0.1)  # 10%的随机抖动
            total_delay = delay + jitter

            if is_rate_limit_error:
                print(f"Rate limit error (429) detected. Waiting {total_delay:.2f} seconds before unlimited retry...")
            else:
                print(f"Waiting {total_delay:.2f} seconds before retry ({attempt}/{max_retries})...")

            time.sleep(total_delay)


def call_dsv32_vendor_b(messages, max_retries=10, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
    """
    调用（匿名机构）云DeepSeek V3.1模型，带有指数退让重试机制

    Args:
        messages: 消息列表
        max_retries: 最大重试次数，默认10次（429错误除外，会无限重试）
        base_delay: 基础延迟时间（秒），默认1秒
        max_delay: 最大延迟时间（秒），默认60秒
        backoff_factor: 退让因子，默认2.0

    Returns:
        str: 模型返回的内容
    """
    import time
    import random

    attempt = 0
    while True:
        try:
            client = OpenAI(
                # 请用知识引擎原子能力API Key将下行替换为：api_key="sk-xxx",
                api_key="REPLACE_WITH_YOUR_API_KEY", # 如何获取API Key：https://cloud.anonymous_org.com/document/product/1772/115970
                base_url="https://api.lkeap.cloud.anonymous_org.com/v1",
            )
            completion = client.chat.completions.create(
                model="deepseek-v3.2",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=messages,
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                },
                stream=False
            )

            content = completion.choices[0].message.content
            return content

        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")

            # 检查是否是429错误（速率限制）
            is_rate_limit_error = False
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit_error = True
            elif "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                is_rate_limit_error = True

            # 如果不是429错误且达到最大重试次数，返回空字符串
            if not is_rate_limit_error and attempt > max_retries:
                print(f"Non-rate-limit error occurred {max_retries} times. Returning empty content.")
                return ""

            # 计算退让延迟时间：基础延迟 * (退让因子 ^ 尝试次数) + 随机抖动
            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            # 添加随机抖动，避免多个请求同时重试
            jitter = random.uniform(0, delay * 0.1)  # 10%的随机抖动
            total_delay = delay + jitter

            if is_rate_limit_error:
                print(f"Rate limit error (429) detected. Waiting {total_delay:.2f} seconds before unlimited retry...")
            else:
                print(f"Waiting {total_delay:.2f} seconds before retry ({attempt}/{max_retries})...")

            time.sleep(total_delay)


def call_dsv31_terminus_vendor_b(messages, enable_thinking=False, max_retries=10, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
    """
    调用（匿名机构）云DeepSeek V3.1模型，带有指数退让重试机制

    Args:
        messages: 消息列表
        max_retries: 最大重试次数，默认10次（429错误除外，会无限重试）
        base_delay: 基础延迟时间（秒），默认1秒
        max_delay: 最大延迟时间（秒），默认60秒
        backoff_factor: 退让因子，默认2.0

    Returns:
        str: 模型返回的内容
    """
    import time
    import random

    thinking = "disabled"
    if enable_thinking:
        thinking = "enabled"

    attempt = 0
    while True:
        try:
            client = OpenAI(
                # 请用知识引擎原子能力API Key将下行替换为：api_key="sk-xxx",
                api_key="REPLACE_WITH_YOUR_API_KEY", # 如何获取API Key：https://cloud.anonymous_org.com/document/product/1772/115970
                base_url="https://api.lkeap.cloud.anonymous_org.com/v1",
            )
            completion = client.chat.completions.create(
                model="deepseek-v3.1-terminus",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=messages,
                extra_body={
                    "thinking": {
                        "type": thinking
                    }
                },
                stream=False
            )
            answer = completion.choices[0].message.content
            reason = ''
            if enable_thinking:
                reason = completion.choices[0].message.reasoning_content
            return answer, reason

        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")

            # 检查是否是429错误（速率限制）
            is_rate_limit_error = False
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit_error = True
            elif "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                is_rate_limit_error = True

            # 如果不是429错误且达到最大重试次数，返回空字符串
            if not is_rate_limit_error and attempt > max_retries:
                print(f"Non-rate-limit error occurred {max_retries} times. Returning empty content.")
                return ""

            # 计算退让延迟时间：基础延迟 * (退让因子 ^ 尝试次数) + 随机抖动
            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            # 添加随机抖动，避免多个请求同时重试
            jitter = random.uniform(0, delay * 0.1)  # 10%的随机抖动
            total_delay = delay + jitter

            if is_rate_limit_error:
                print(f"Rate limit error (429) detected. Waiting {total_delay:.2f} seconds before unlimited retry...")
            else:
                print(f"Waiting {total_delay:.2f} seconds before retry ({attempt}/{max_retries})...")

            time.sleep(total_delay)
    return None,None

def call_dsv31_vendor_b_thinking(messages, max_retries=10, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
    """
    调用（匿名机构）云DeepSeek V3.1模型，带有指数退让重试机制

    Args:
        messages: 消息列表
        max_retries: 最大重试次数，默认10次（429错误除外，会无限重试）
        base_delay: 基础延迟时间（秒），默认1秒
        max_delay: 最大延迟时间（秒），默认60秒
        backoff_factor: 退让因子，默认2.0

    Returns:
        str: 模型返回的内容
    """
    import time
    import random

    attempt = 0
    while True:
        try:
            client = OpenAI(
                # 请用知识引擎原子能力API Key将下行替换为：api_key="sk-xxx",
                api_key="REPLACE_WITH_YOUR_API_KEY", # 如何获取API Key：https://cloud.anonymous_org.com/document/product/1772/115970
                base_url="https://api.lkeap.cloud.anonymous_org.com/v1",
            )
            completion = client.chat.completions.create(
                model="deepseek-v3.1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=messages,
                extra_body={
                    "thinking": {
                        "type": "enabled"
                    }
                },
                stream=False
            )

            thinking_content = completion.choices[0].message.reasoning_content
            content = completion.choices[0].message.content
            assert content!="", f"输出结果为空, {completion.choices[0].message}"
            return content, thinking_content

        except Exception as e:
            attempt += 1
            import traceback
            print(traceback.format_exc())
            print(f"Attempt {attempt} failed: {e}")

            # 检查是否是429错误（速率限制）
            is_rate_limit_error = False
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit_error = True
            elif "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                is_rate_limit_error = True

            # 如果不是429错误且达到最大重试次数，返回空字符串
            if not is_rate_limit_error and attempt > max_retries:
                print(f"Non-rate-limit error occurred {max_retries} times. Returning empty content.")
                return ""

            # 计算退让延迟时间：基础延迟 * (退让因子 ^ 尝试次数) + 随机抖动
            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            # 添加随机抖动，避免多个请求同时重试
            jitter = random.uniform(0, delay * 0.1)  # 10%的随机抖动
            total_delay = delay + jitter

            if is_rate_limit_error:
                print(f"Rate limit error (429) detected. Waiting {total_delay:.2f} seconds before unlimited retry...")
            else:
                print(f"Waiting {total_delay:.2f} seconds before retry ({attempt}/{max_retries})...")

            time.sleep(total_delay)

def extract_think_content(text):
    """提取<think></think>标签之间的内容"""
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0]  # 返回第一个匹配的内容
    return ''

def extract_answer_content(text):
    """
    提取</think>之后的内容，包括<tool_call>或<answer>标签的完整内容
    """
    if not text:
        return ""

    # 方法1: 使用正则表达式找到</think>之后的所有内容
    answer_pattern = r'</think>(.*)'
    matches = re.findall(answer_pattern, text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # 方法2: 使用split但更安全的方式
    splits = text.split('</think>')
    if len(splits) > 1:
        return splits[1].strip()

    return ""

webpage_tpl="""[webpage {idx} begin]
[webpage title]
{title}
[webpage url]
{domain}
[webpage date published]
{date}
[webpage authoritativeness]
{auth}
[webpage content begin]
{passage}
[webpage content end]
[webpage {idx} end]"""

def format_search_results(docs_list, global_index):
    """
    拼装搜索结果
    """
    search_result = ""

    for doc in docs_list:
        search_result += webpage_tpl.format(
            idx=global_index+1,
            title=doc['title'],
            domain=doc['url'],
            date=doc.get('date', ''),
            auth=doc.get('auth',''),
            passage=doc['content'],
        ) + '\n\n'
        global_index += 1

    return search_result.strip(), global_index


def format_search_results_as(
    query_list: List[str],
    results: Dict[str, Tuple[List[Any], List[Any]]],
    top_k: int = 10,
    global_index: int = 0
) -> Tuple[str, int]:
    """
    遍历query_list，取results里对应的docs的top_k，用template拼接

    Args:
        query_list: 查询列表
        results: {subquery: (docs, docs_expanded)} 格式的搜索结果
        top_k: 每个query取的文档数量，默认10
        global_index: 全局索引起始值

    Returns:
        Tuple[拼接后的搜索结果字符串, 更新后的global_index]
    """
    all_search_results = []

    for query in query_list:
        if query not in results:
            all_search_results.append(f"No results found for query: '{query}'. Use a less specific query.")
            continue

        docs, docs_expanded = results[query]
        # 合并docs，优先取docs，再取docs_expanded
        all_docs = docs # + docs_expanded
        top_docs = all_docs[:top_k]

        if not top_docs:
            all_search_results.append(f"No results found for query: '{query}'. Use a less specific query.")
            continue

        # 拼接该query的搜索结果
        query_result = f"A search for {query} found the following results:\n\n"

        for doc in top_docs:
            query_result += webpage_tpl.format(
                idx=global_index + 1,
                title=doc.get('title', ''),
                domain=doc.get('url', ''),
                date=doc.get('date', ''),
                auth=doc.get('auth', ''),
                passage=doc.get('content', '')
            ) + '\n\n'
            global_index += 1

        all_search_results.append(query_result.strip())

    # 用 \n\n 拼接所有query的结果
    final_result = "\n\n".join(all_search_results).strip()

    return final_result, global_index

def format_search_results_tongyi_visit(docs_list, global_index):
    """
    拼装搜索结果
    """
    search_result = ""

    for doc in docs_list:
        search_result += webpage_tpl.format(
            idx=global_index,
            title=doc['title'],
            domain=doc['url'],
            date=doc.get('date', ''),
            auth=doc.get('auth',''),
            passage=doc['content_long'],
        )
        global_index += 1

    return search_result, global_index

def parse_answer(text):
    """提取<answer></answer>标签之间的内容"""
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()  # 返回第一个匹配的内容
    return ''

def count_tokens(raw_trajectory):
    return len(tokenizer.encode(raw_trajectory))


def majority_voting(func, query, turn_history, prev_trajectory="", prev_conversation=[], n=5):
    """
    多数投票函数：调用process_query n次，收集答案，使用verifier选择最佳答案

    Args:
        query: 用户查询
        turn_history: 轮次历史
        prev_trajectory: 之前的轨迹
        prev_conversation: 之前的对话
        n: 投票次数，默认3次

    Returns:
        选中的最佳答案结果
    """
    majority_voting_tpl = load_prompt("prompts/majority_voting_tpl.txt") # none

    print(f"开始majority voting，将调用process_query {n}次")
    answers = []
    all_results = []

    for i in range(n):
        print(f"执行第{i+1}次process_query...")
        try:
            result = func(query, turn_history, prev_trajectory, prev_conversation)
            all_results.append(result)
            answers.append(result['answer'])
        except Exception as e:
            print(f"第{i+1}次process_query执行失败: {e}")
            answers.append("执行失败")
            all_results.append(None)

    print(f"收集到{len(answers)}个答案，开始verifier选择...")
    verifier_input = majority_voting_tpl + f"\n[query begin]{query}[query end]\n"
    for i, answer in enumerate(answers):
        verifier_input += f"[answer {i} begin] {answer} [answer {i} end]\n"
    try:
        verifier_response, verifier_reason = call_model_api(verifier_input, model = 'gpt5-chat', system_prompt = "You are a fair judge")

        # 解析verifier响应
        try:
            verifier_result = json.loads(verifier_response)
            selected_index = verifier_result.get('index', 0)
            reason = verifier_result.get('reason', '未提供理由')

            print(f"Verifier选择了索引{selected_index}的答案，理由: {reason}")

            # 确保索引有效
            if 0 <= selected_index < len(all_results) and all_results[selected_index] is not None:
                return all_results[selected_index]
            else:
                print(f"选择的索引{selected_index}无效，返回第一个有效结果")
                for result in all_results:
                    if result is not None:
                        return result

        except json.JSONDecodeError as e:
            print(f"解析verifier响应失败: {e}")
            print(f"原始响应: {verifier_response}")

    except Exception as e:
        print(f"调用verifier失败: {e}")

    # 如果verifier失败，返回第一个有效结果
    print("Verifier失败，返回第一个有效结果")
    for result in all_results:
        if result is not None:
            return result

    # 如果所有结果都失败，返回默认结果
    print("所有process_query都失败，返回默认结果")
    return {
        'query': query,
        'answer': '抱歉，无法获取有效答案',
        'conversations': [],
        'search_result_all': '',
        'raw_tracjectory': '',
        'readable_trajectory': '',
        'r1_ori_answer': '',
        'r1_citation_answer': '',
        'status': 'error',
        'final_trajectory_length': 0,
        'sft_data': {
            'conversations': prev_conversation.copy(),
            'system': "You are a helpful assistant.",
            'loss_mask': [0 for _ in range(len(prev_conversation))]
        }
    }

def process_sft_session(func, session_dict, system_prompt, is_voting=False):
    try:
        turn_history = [{
            'role': 'system',
            'content': system_prompt,
        }]
        # print(session_dict)
        session_dict['session_results'] = {
            'rounds': [],
        }
        # 前一轮的问答长度 用于控制后面的长度预算
        prev_trajectory = '<｜im_start｜> system\n' + system_prompt

        if 'fix_history' in session_dict:
            for item in session_dict['fix_history']:
                if item['role'] == 'user':
                    turn_history.append({
                        'role': 'user',
                        'content': item['content']
                    })
                elif item['role'] == 'assistant':
                    turn_history.append({
                        'role': 'assistant',
                        'content': item['content']
                    })

        for item in session_dict['rounds']:

            query = item['query']
            if is_voting:
                response = majority_voting(func, query, turn_history, prev_trajectory,  n=3)
            else:
                response = func(query, turn_history, prev_trajectory, max_steps=100)
            turn_history.append({
                'role': 'user',
                'content': query
            })
            # 把all_docs thinking作为历史输出
            # 先判断是否有all_docs thinking with trajectory answer, 然后判断是否有all_docs_thinking_answer
            if 'v31_all_docs_thinking_with_trajectory_answer' in response:
                if response['v31_all_docs_thinking_with_trajectory_answer']:
                    print('用了两阶段轨迹作为历史了！')

                    answer = response['v31_all_docs_thinking_with_trajectory_answer']
                elif 'v31_all_docs_thinking_answer' in response:
                    print('用了二阶段作为历史了！')
                    answer = response['v31_all_docs_thinking_answer'] if response['v31_all_docs_thinking_answer'] else response['answer']
            else:
                if 'v31_all_docs_thinking_answer' in response:
                    print('用了二阶段作为历史了！')
                    answer = response['v31_all_docs_thinking_answer'] if response['v31_all_docs_thinking_answer'] else response['answer']
                else:
                    print("thinking结果为空？")
                    answer = response['answer']

            turn_history.append({
                'role': 'assistant',
                'content': '<think>\n\n</think>\n' + answer
            })
            prev_trajectory += '<|im_start|>user\n' + query + '<|im_end><|im_start|>assistant\n' + '<think>\n\n</think>\n' + answer + '<|im_end|>'

            session_dict['session_results']['rounds'].append(response)

        return session_dict
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"session_dict 有问题 :{session_dict}")
        return None

def process_sft_data(func, system_prompt, input_path, output_path, num_threads=4, start_index=0,is_voting=False):
    import os
    from datetime import datetime
    from concurrent.futures import as_completed
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    # 读取数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    round_output_path = output_path.replace('.jsonl', '_round.jsonl')
    data = data[start_index:]

    output_filename = os.path.basename(output_path).replace('.jsonl', '')
    error_log_path = os.path.join('log/error_log', f'{output_filename}_multi_turn_errors.log')
    failed_indices_path = os.path.join('log/error_log', f'{output_filename}_multi_turn_failed_indices.txt')

    print(f"总共需要处理 {len(data)} 条多轮对话数据")
    print(f"输出文件：")
    print(f"  Round data: {round_output_path}")
    print(f"错误日志将保存到: {error_log_path}")
    print(f"失败索引将保存到: {failed_indices_path}")
    progress_bar = tqdm(total=len(data), desc="Processing Multi-turn", position=0, leave=True)


    completed_results = {}
    next_write_index = 0
    failed_indices = set()

    # 确保log目录存在
    os.makedirs('log', exist_ok=True)
    os.makedirs('log/error_log', exist_ok=True)
    # 初始化错误日志文件
    with open(error_log_path, 'a', encoding='utf-8') as error_log:
        error_log.write(f"多轮对话错误日志 - 开始时间: {datetime.now()}\n")
        error_log.write(f"输入文件: {input_path}\n")
        error_log.write(f"输出文件: {round_output_path}\n")
        error_log.write(f"总数据量: {len(data)}\n")
        error_log.write("=" * 80 + "\n\n")

    with open(round_output_path, 'a', encoding='utf-8') as round_f:

        # 文件锁，确保多线程写入安全
        round_lock = threading.Lock()

        def write_result(original_index, session_data):
            """写入结果到文件"""
            if session_data is None:
                return

            # 添加original_index到结果中
            session_data['original_index'] = original_index

            # 写入round_data
            with round_lock:
                round_f.write(json.dumps(session_data, ensure_ascii=False) + '\n')
                round_f.flush()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务，同时记录索引
            future_to_index = {executor.submit(process_sft_session, func, line, system_prompt,is_voting): i for i, line in enumerate(data)}

            # 使用 as_completed 来获取完成的任务
            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    # 获取返回值
                    session_data = future.result()

                    # 检查session_data是否为None
                    if session_data is None:
                        failed_indices.add(original_index)
                        error_msg = f"索引 {original_index} 处理失败: process_sft_session\n"
                        print(error_msg.strip())

                        # 写入错误日志
                        with open(error_log_path, 'a', encoding='utf-8') as error_log:
                            error_log.write(f"时间: {datetime.now()}\n")
                            error_log.write(f"索引: {original_index}\n")
                            error_log.write(f"错误: process_sft_session\n")
                            if original_index < len(data):
                                error_log.write(f"数据: {str(data[original_index])[:500]}...\n")
                            error_log.write("-" * 40 + "\n\n")
                    else:
                        # 立即写入结果
                        write_result(original_index, session_data)

                except Exception as e:
                    failed_indices.add(original_index)
                    error_msg = f"索引 {original_index} 处理失败: {str(e)}\n"
                    print(error_msg.strip())

                    # 写入错误日志
                    with open(error_log_path, 'a', encoding='utf-8') as error_log:
                        error_log.write(f"时间: {datetime.now()}\n")
                        error_log.write(f"索引: {original_index}\n")
                        error_log.write(f"错误: {str(e)}\n")
                        if original_index < len(data):
                            error_log.write(f"数据: {str(data[original_index])[:500]}...\n")
                        error_log.write("-" * 40 + "\n\n")

                # 无论成功失败都更新进度条
                progress_bar.update(1)

    progress_bar.close()

    # 现在读取文件并按original_index重新排序
    print("正在按索引顺序重新排序结果...")

    # 读取round数据并排序
    round_data = []
    with open(round_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_item = json.loads(line)
                round_data.append(data_item)

    # 按original_index排序
    round_data.sort(key=lambda x: x.get('original_index',0))

    # 重新写入排序后的round数据（移除original_index字段）
    with open(round_output_path, 'w', encoding='utf-8') as f:
        for item in round_data:
            item_copy = item.copy()
            f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')

    # 保存失败的索引
    if failed_indices:
        with open(failed_indices_path, 'w', encoding='utf-8') as f:
            for idx in sorted(failed_indices):
                f.write(f"{idx}\n")

    success_count = len(data) - len(failed_indices)
    print(f"\n处理完成！")
    print(f"成功: {success_count}/{len(data)}")
    print(f"失败: {len(failed_indices)}/{len(data)}")
    print(f"输出文件统计：")
    print(f"  Round data lines: {success_count}")
    if failed_indices:
        print(f"失败索引已保存到: {failed_indices_path}")
        print(f"错误详情已保存到: {error_log_path}")

def serper_search(query):
    """
    使用 Serper API 执行搜索

    Args:
        query (str): 要搜索的查询词
        api_key (str): 你的 Serper API 密钥

    Returns:
        dict: API 返回的 JSON 结果
    """
    api_key = "REPLACE_WITH_YOUR_API_KEY"
    def contains_chinese_basic(text: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in text)

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    if contains_chinese_basic(query):
        payload = {
            "q": query,
            "location": "China",
            "gl": "cn",
            "hl": "zh-cn"
        }

    else:
        payload = {
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en"
        }

    success = False
    for i in range(5):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 检查请求是否成功
            res_json =  response.json()
            if "organic" in res_json:
                return res_json["organic"]
            else:
                return f"No results found for query: '{query}'. Use a less specific query."
        except Exception as e:
            print(f"请求出错: {e}")
    return f"No results found for query: '{query}'. Use a less specific query"

def call_google_search(subqueries: List[str]) -> Tuple[int,Dict[str,Any]]:
    assert isinstance(subqueries, List)
    docs = {}
    search_error = 0
    for q in subqueries:
        search_res = serper_search(q)
        if isinstance(search_res, str):
            search_error += 1
            docs[q] = []
            print(f'{q}搜索失败: {search_res}')
        else:
            docs[q] = search_res
    return search_error, docs

def format_search_results_google_tongyi(docs, global_index):
    all_content = ""
    for q, search_results in docs.items():
        web_snippets = list()
        if search_results:
            for page in search_results:
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{global_index}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)
                global_index += 1

            content = f"A Google search for '{q}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        else:
            content = f"No results found for query: '{q}'. Use a less specific query."
        all_content += "\n=======\n" + content
    return all_content, global_index

def format_search_results_google_ds(docs,global_index):
    webpage_tpl_ds_google = """[webpage {id} end]
[webpage title]{title}
[webpage url]{url}
[webpage date published]{date}
[webpage snippet begin]
{snippet}
[webpage snippet end]
[webpage content begin]
暂无详细信息
[webpage content end]
[webpage {id} end]
"""
    all_content = ""
    for q, search_results in docs.items():
        web_snippets = list()
        if search_results:
            for page in search_results:
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = webpage_tpl_ds_google.format(
                    id=global_index,
                    title=page['title'],
                    url=page['link'],
                    date=date_published,
                    snippet=snippet,
                )
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)
                global_index += 1
            content = f"A Google search for '{q}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        else:
            content = f"No results found for query: '{q}'. Use a less specific query."
        all_content += "\n=======\n" + content
    return all_content, global_index


if __name__ == "__main__":
    # print(serper_search('what is the capital of china'))
    # # python -u '${WORKSPACE}' > log/v41_0915.tx
    # import os
    # import requests
    # from openai import OpenAI
    # import json

    # client = OpenAI(
    #     # 请用知识引擎原子能力API Key将下行替换为：api_key="sk-xxx",
    #     api_key="REPLACE_WITH_YOUR_API_KEY", # 如何获取API Key：https://cloud.anonymous_org.com/document/product/1772/115970
    #     base_url="https://api.lkeap.cloud.anonymous_org.com/v1",
    # )
    content = """The current date is 2025-08-28, Thursday. Your primary task is to solve the user's questions, leveraging the appropriate tools as needed.
    [Core Instruction: Language Consistency]
    You MUST write your entire response in the same language as the user's question.
    If the user asks in Chinese, you must think and write in Chinese.
    If the user asks in French, you must think and write in French.
    Do not switch to English unless the user's question was in English. This applies to your reasoning, explanations, and the final answer.
    [Core Instruction: Citation Formatting]
    If the search tool was used and you are referencing a specific webpage in your final answer (the content after <｜search▁end｜>), you must use the format "[citation:x]", where x is the corresponding webpage number.
    You must first engage in step-by-step, rigorous thinking and reasoning before using the appropriate tool or providing the final answer.
    [Core Instruction: Search Budget Management]
    You have a maximum of 5 search attempts total. To use this budget efficiently:
    1. If you reach the 5-search limit, provide an answer based on available results and clearly indicate the limitations of your response due to search constraints.
    2. Plan your search strategy before executing to prevent wasting search budget on one task at the expense of others.
    3. Avoid searching 3 or more similar queries to prevent search budget waste.

    ## Tools
    You have access to the following tools:

    ### search
    Description: Search the web for information. The result will be a string.

    Parameters: {{"type": "object", "properties": {{"questions": {{"type": "string", "description": "Generate up to 5 search queries based on the requirements below:\n\n- Separate different queries with \"||\". For example: \"Elon Musk lawsuit OpenAI reasons||Elon Musk lawsuit OpenAI latest updates\".\n- All search queries should use generalized, search engine-friendly, and easily retrievable keywords to improve search efficiency and the relevance of results. Avoid directly extracting phrases from the question that are difficult to retrieve effectively. For example: Avoid using relative or ambiguous terms such as “third author” or “the following year.” - Instead, use more universal expressions like “author list” or “award history.”\n- For a single search query, keep them short - remove unnecessary auxiliary words, conjunctions, and interrogative words. Primarily use keyword combinations (separated by spaces). Ideally, they should not exceed 15 characters. Avoid using special punctuation marks.\n- When seeking information from non-English primary sources, adjust search query to retrieve content in the original language (e.g., chinese), even if your initial question is in English. This improves accuracy and avoids translation-related errors.\n- You can search 1 to 5 search queries in parallel each time. If the question involves comparing multiple entities (e.g., people, places, books, movies) or contains two or more sub-questions, please split it into separate queries for each entity or sub-question.\n- Do not re-search for queries that have already been searched."}}}}, "required": ["questions"]}}

    IMPORTANT: ALWAYS adhere to this exact format for tool use:
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

    Where:
    - `tool_call_name` must be an exact match to one of the available tools
    - `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
    - For multiple tool calls, chain them directly without separators or spaces

    # The user's message is: 我是一个16岁的男高中生 我要和我70岁的爷爷婆婆在一起旅游 我的体力还不错 但是老人已经70岁了 他们需要去室内的景点 最好有坐的地方 还能喝茶 有空调吹 请给我规划一下成都到重庆当天来回的一下午（13:00-18:00）的行程 帮我个人快速去打卡重庆沙坪坝区、渝中区、南岸区、江北区（市中心的5个景点）包括咖啡馆 citywalk路线连接景点 同时帮我规划我带老人在重庆室内的地方休憩的路线"""*100
    messages= [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': content}
    ]
    text = ''.join([msg['content'] for msg in messages])
    print("输入token数:", count_tokens(text))
    call_dsv31_vendor_b_thinking(messages)
    # # completion = client.chat.completions.create(
    # #     model="deepseek-v3.1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    # #     messages=messages,
    # #     extra_body={
    # #         "thinking": {
    # #             "type": "disabled"
    # #         }
    # #     },
    # #     stream=False
    # # )
    # # print(completion.choices[0].message.content)
    # print(call_dsv31_vendor_b(messages))
    # # pass
    # docs = call_google_search(["During which tour was Ghost's Papa Nihil resurrected?"])
    # print(docs)
    # print(call_dsv31_terminus_thinking(messages))
