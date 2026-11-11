import os
import sys

project_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_src not in sys.path:
    sys.path.insert(0, project_src)
    
import json
from typing import List, Union
import requests
import time 
import tiktoken
from utils.utils import call_dsv31_terminus_vendor_b, load_prompt

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
JINA_API_KEYS = os.getenv("JINA_API_KEYS", "jina_b4842818a2814a298db29dc1834d22e6_14EMP8RZVY3m3ydB0cK99BdlcHe")
EXTRACTOR_PROMPT = load_prompt(os.path.join(root_dir, "prompts/Tongyi-DeepResearch-30B-A3B/extractor_prompt.txt"))

def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


def _call_visit(url_list: List[str], goal: str) -> str:
    """
    调用批量爬虫API获取网页内容
    
    Args:
        url_list: URL列表
        goal: 目标描述
        
    Returns:
        str: API返回的文本内容或错误信息
    """
    # 参数验证
    if not isinstance(url_list, list) or not url_list:
        return "[ERROR]调用参数错误"
     
    if not isinstance(goal, str) or not goal.strip():
        return "[ERROR]调用参数错误"
    
    # 验证URL列表中的每个元素都是字符串
    for url in url_list:
        if not isinstance(url, str) or not url.strip():
            return "[ERROR]调用参数错误"
              
    # API配置
    api_url = os.getenv("KNOWGPO_LEGACY_VISIT_URL", "https://example.invalid/visit")
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "url": url_list,
        "goal": goal
    }
    
    # 指数退让重试配置
    max_retries = 3
    base_delay = 1  # 基础延迟时间（秒）
    
    for attempt in range(max_retries + 1):
        try:
            # 设置超时时间
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30  # 30秒超时
            )
            
            # 检查HTTP状态码
            if response.status_code == 200:
                # 尝试解析JSON响应
                try:
                    result = response.json()
                    print(f"[_call_visit] API返回类型: {type(result)}")
                    # 如果返回的是字符串，直接返回
                    if isinstance(result, str):
                        print(f"[_call_visit] 返回字符串, 长度: {len(result)}")
                        return result
                    # 如果返回的是字典，检查特定字段
                    elif isinstance(result, dict):
                        print(f"[_call_visit] 返回字典, keys: {list(result.keys())}")
                        # 优先检查content字段（API返回格式：{"res_info": {...}, "content": "..."}）
                        if 'content' in result:
                            content = result['content']
                            print(f"[_call_visit] content字段长度: {len(content) if content else 0}")
                            if not content or len(content.strip()) == 0:
                                print(f"[_call_visit] 警告: content为空! 完整响应: {json.dumps(result, ensure_ascii=False)[:500]}")
                            return content if content else ""
                        # 检查是否有text字段
                        elif 'text' in result:
                            text = result['text']
                            print(f"[_call_visit] text字段长度: {len(text) if text else 0}")
                            return text if text else ""
                        else:
                            print(f"[_call_visit] 无content/text字段, 返回完整JSON")
                            return json.dumps(result, ensure_ascii=False, indent=2)
                    else:
                        print(f"[_call_visit] 未知返回类型: {type(result)}")
                        return str(result)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，直接返回文本内容
                    print(f"[_call_visit] 非JSON响应, 返回原始文本, 长度: {len(response.text)}")
                    return response.text
            else:
                # HTTP状态码不是200，记录错误并重试
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 指数退让
                    time.sleep(delay)
                    continue
                else:
                    return "[ERROR]调用异常"
                            
        except Exception as e:
            # 其他未预期的异常
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # 指数退让
                time.sleep(delay)
                continue
            else:
                return "[ERROR]调用异常"
    
    return "[ERROR]调用异常"

def call_visit(params: Union[str, dict], **kwargs) -> str:
    try:
        url = params["url"]
        goal = params["goal"]
    except:
        return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

    start_time = time.time()
    
    # Create log folder if it doesn't exist
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)

    try:
        response = _call_visit(url,goal)
        return response.strip()
    except:
            
        if isinstance(url, str):
            response = readpage_jina(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url: 
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        cur_response = readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()
        
def call_server(msgs, max_retries=2):
    # 使用（匿名机构）云 Terminus 接口，保持与原逻辑一致的返回字符串
    answer, _ = call_dsv31_terminus_vendor_b(messages=msgs, enable_thinking=False, max_retries=max_retries)
    content = answer if isinstance(answer, str) else ""
    if content:
        try:
            json.loads(content)
        except:
            # 尝试从字符串中提取 JSON
            left = content.find('{')
            right = content.rfind('}')
            if left != -1 and right != -1 and left <= right:
                content = content[left:right+1]
        return content
    return ""
    
def jina_readpage(url: str) -> str:
    """
    Read webpage content using Jina service.
    
    Args:
        url: The URL to read
        goal: The goal/purpose of reading the page
        
    Returns:
        str: The webpage content or error message
    """
    max_retries = 3
    timeout = 50
    
    for attempt in range(max_retries):
        response = None
        released = False
        try:
            
            response = requests.get(
                f"https://r.jina.ai/{url}",
                timeout=timeout
            )
            
            released = True

            if response.status_code == 200:
                webpage_content = response.text
                return webpage_content
            else:
                print(response.text)
                print(f"jina readpage error, status code: {response.status_code}")
                # 发生限流/封禁时按监视器建议等待
                if response.status_code in (429, 451):
                    
                    time.sleep(min(wait_s, 10.0))
                else:
                    time.sleep(0.5)
       
        except Exception as e:
            print(f"jina readpage exception: {str(e)}")
            time.sleep(0.5)
            
                
            
    return "[visit] Failed to read page."

# 这里的summary模型我暂时用deepseek-v3.1-terminus来替代
def html_readpage_jina(url: str) -> str:
    max_attempts = 8
    for attempt in range(max_attempts):
        content = jina_readpage(url)
        service = "jina"     
        # print(service)
        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            return content
    return "[visit] Failed to read page."

def readpage_jina(url: str, goal: str) -> str:
    """
    Attempt to read webpage content by alternating between jina and aidata services.
    
    Args:
        url: The URL to read
        goal: The goal/purpose of reading the page
        
    Returns:
        str: The webpage content or error message
    """

    summary_page_func = call_server
    max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))

    content = html_readpage_jina(url)

    if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
        content = truncate_to_tokens(content, max_tokens=95000)
        messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
        parse_retry_times = 0
        raw = summary_page_func(messages, max_retries=max_retries)
        summary_retries = 3
        while len(raw) < 10 and summary_retries >= 0:
            truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
            status_msg = (
                f"[visit] Summary url[{url}] " 
                f"attempt {3 - summary_retries + 1}/3, "
                f"content length: {len(content)}, "
                f"truncating to {truncate_length} chars"
            ) if summary_retries > 0 else (
                f"[visit] Summary url[{url}] failed after 3 attempts, "
                f"final truncation to 25000 chars"
            )
            print(status_msg)
            content = content[:truncate_length]
            extraction_prompt = EXTRACTOR_PROMPT.format(
                webpage_content=content,
                goal=goal
            )
            messages = [{"role": "user", "content": extraction_prompt}]
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries -= 1

        parse_retry_times = 2
        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()
        while parse_retry_times < 3:
            try:
                raw = json.loads(raw)
                break
            except:
                raw = summary_page_func(messages, max_retries=max_retries)
                parse_retry_times += 1
        
        if parse_retry_times >= 3:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
            useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

        if len(useful_information) < 10 and summary_retries < 0:
            print("[visit] Could not generate valid summary after maximum retries")
            useful_information = "[visit] Failed to read page"
        
        return useful_information

    # If no valid content was obtained after all retries
    else:
        useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
        useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
        useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        return useful_information


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Heard_Island_and_McDonald_Islands"
    goal = "测试"
    params = {"url": [url], "goal": goal}
    # content = readpage_jina(url, goal)
    # print(content)
    print(call_visit(params))
