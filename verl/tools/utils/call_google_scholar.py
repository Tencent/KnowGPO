import requests
from typing import List,Tuple,Dict,Any

def serper_scholar_search(query):
    """
    使用 Serper API 执行学术搜索
    
    Args:
        query (str): 要搜索的查询词
        api_key (str): 你的 Serper API 密钥
    
    Returns:
        dict: API 返回的 JSON 结果
    """
    api_key = "REPLACE_WITH_YOUR_API_KEY"

    url = "https://google.serper.dev/scholar"
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
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
            if i == 4:
                return f"Google Scholar Timeout, return None, Please try again later."
    return f"No results found for query: '{query}'. Use a less specific query"

def call_google_scholar(subqueries: List[str]) -> Tuple[int,Dict[str,Any]]:
    assert isinstance(subqueries, List)
    docs = {}
    search_error_num = 0
    for q in subqueries:
        search_res = serper_scholar_search(q)
        if isinstance(search_res, str) or (isinstance(search_res, list) and len(search_res)==0):
            search_error_num += 1
            print(f'{q}搜索失败: {search_res}') # 如果失败会保存成：f"Google Scholar Timeout, return None, Please try again later." 或者 f"No results found for query: '{query}'. Use a less specific query"
        docs[q] = search_res
    return search_error_num, docs

def format_search_results_scholar_for_tongyi(docs):
    all_content = ""
    for q, search_results in docs.items():
        web_snippets = list()
        if isinstance(search_results, list) and len(search_results) > 0:
            idx = 0
            for page in search_results:
                idx += 1
                date_published = ""
                if "year" in page:
                    date_published = "\nDate published: " + str(page["year"])

                publicationInfo = ""
                if "publicationInfo" in page:
                    publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]
                
                link_info = "no available link"
                if "pdfUrl" in page: 
                    link_info = page["pdfUrl"]
                
                citedBy = ""
                if "citedBy" in page:
                    citedBy = "\ncitedBy: " + str(page["citedBy"])

                redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            content = f"A Google Scholar for '{q}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
        else:
            # 处理字符串或空列表的情况，确保content总是字符串
            if isinstance(search_results, str):
                content = search_results
            else:
                content = f"No results found for query: '{q}'. Use a less specific query."
        
        # 确保content是字符串类型
        if not isinstance(content, str):
            content = str(content)
            
        all_content += "\n=======\n" + content
    return all_content

if __name__ == "__main__":
    """
        示例用法：
        1. serper_scholar_search: 输入query, 输出docs list
        2. call_google_scholar: 输入query list, 输出一个dict,包含每一个query对应的docs list
        3. format_search_results_scholar_for_tongyi: 输入dict，输出搜索结果组织成的文本：
            对于每一个query:
                A Google Scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" 
            不同的query之间用 "\n=======\n"隔开
    """
    result = serper_scholar_search("transformer attention mechanism")
    print(result)
    search_error, all_query_search_results = call_google_scholar(["transformer attention mechanism","BERT pre-training"])
    all_content = format_search_results_scholar_for_tongyi(all_query_search_results)
    print('*'*10)
    print(search_error)
    print('*'*10)
    print(all_query_search_results)
    print('*'*10)
    print(all_content)
    print('*'*10)
