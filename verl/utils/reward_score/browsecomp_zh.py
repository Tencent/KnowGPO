# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import logging

from verl.utils.reward_score.api_client_factory import create_api_client, call_model_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 从browsecomp_zh_prompt.txt中提取的评分prompt
JUDGE_PROMPT_EN = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].  Put the extracted answer as  'None'  if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer],
        focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer.
        Do not comment on any background to the problem, do not attempt to solve the problem,
        do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer]  given  above,
        or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e.
        if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
"""

JUDGE_PROMPT_CN = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。
"""

class BrowseCompZHEvaluator:
    def __init__(
        self,
        grader_model_name: str = "gpt-4o",
        language: str = "zh",
        api_provider: str = "vendor_a",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        global_timeout: float = 600.0,
    ):
        self.grader_model_name = grader_model_name
        self.language = language.lower()
        self.api_provider = api_provider
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.fallback_models = fallback_models
        self.global_timeout = global_timeout
        # Keep client for backward compatibility, but grade_answer uses call_model_api
        self.client = create_api_client(
            provider=api_provider,
            api_key=api_key,
            base_url=api_base_url,
        )
        logger.info(
            f"BrowseCompZHEvaluator initialized: provider={api_provider}, "
            f"model={grader_model_name}, language={language}, "
            f"fallback_models={fallback_models}, global_timeout={global_timeout}"
        )

    def get_prompt_template(self):
        if self.language == "zh":
            return JUDGE_PROMPT_CN
        elif self.language == "en":
            return JUDGE_PROMPT_EN
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def grade_answer(self, question: str, correct_answer: str, response: str) -> Tuple[Dict, str]:
        """Grade an answer using the unified call_model_api with full failover.

        Uses api_client_factory.call_model_api() which supports:
          - Multi-model fallback (fallback_models)
          - 429 rate limit handling (does not consume retries)
          - Global timeout protection
          - Random model selection for load spreading
        """
        prompt = self.get_prompt_template().format(
            question=question,
            response=response,
            correct_answer=correct_answer
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            answer_text, _ = call_model_api(
                messages=messages,
                provider=self.api_provider,
                model=self.grader_model_name,
                api_key=self.api_key,
                base_url=self.api_base_url,
                enable_thinking=False,
                max_retries=3,
                temperature=0,
                max_tokens=500,
                fallback_models=self.fallback_models,
                global_timeout=self.global_timeout,
            )

            if not answer_text:
                logger.warning("grade_answer: call_model_api returned empty content")
                return {
                    'extracted_final_answer': 'None',
                    'correct': 'no',
                    'confidence': '100',
                    'reasoning': 'ERROR: API returned empty content after all retries'
                }, 'ERROR: API returned empty content'

            model_output = answer_text.strip()
            result = self.parse_judge_output(model_output)
            return result, model_output

        except Exception as e:
            logger.error(f"grade_answer failed: {e}")
            return {
                'extracted_final_answer': 'None',
                'correct': 'no',
                'confidence': '100',
                'reasoning': f"ERROR: {str(e)}"
            }, f"ERROR: {str(e)}"

    def parse_judge_output(self, output: str) -> Dict:
        """解析judge模型的输出，提取各个字段"""
        result = {
            'extracted_final_answer': 'None',
            'correct': 'no',
            'confidence': '100',
            'reasoning': ''
        }

        # 提取extracted_final_answer
        extracted_match = re.search(r'extracted_final_answer:\s*(.+?)(?=\n|$)', output, re.IGNORECASE | re.DOTALL)
        if extracted_match:
            result['extracted_final_answer'] = extracted_match.group(1).strip()

        # 提取correct
        correct_match = re.search(r'correct:\s*(yes|no)', output, re.IGNORECASE)
        if correct_match:
            result['correct'] = correct_match.group(1).lower()

        # 提取confidence
        confidence_match = re.search(r'confidence:\s*(\d+(?:\.\d+)?)', output, re.IGNORECASE)
        if confidence_match:
            result['confidence'] = confidence_match.group(1)

        # 提取reasoning
        reasoning_match = re.search(r'reasoning:\s*(.+?)(?=\n|$)', output, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()

        return result

# Global evaluator instance (default: vendor_a provider)
# This will be re-initialized when compute_score is called with api_config
_default_evaluator = BrowseCompZHEvaluator()
_evaluator_cache = {}



# def normalize_answer(s):
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def em_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer == normalized_prediction:
#             score = 1
#             break
#     return score


# def subem_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer in normalized_prediction:
#             score = 1
#             break
#     return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # 添加对None和空字符串的处理
    if solution_str is None or solution_str.strip() == "":
        return None

    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_question(solution_str):
    """Extract the question from the solution string."""
    # 提取当前用户问题（支持多种格式）
    q = None
    try:
        # 尝试第一种格式：Question: ...
        q_pattern = r"Question: (.*?)\nassistant"
        matches = re.findall(q_pattern, solution_str, re.DOTALL)
        if matches:
            q = matches[0].strip()
        else:
            # 尝试第二种格式：The user's message is: ...
            q_pattern = r"The user's message is:(.*?)assistant"
            matches = re.findall(q_pattern, solution_str, re.DOTALL)
            if matches:
                q = matches[0].strip()
    except Exception as e:
        logger.error(f"提取query时出错: {e}")
    return q


def _get_evaluator(
    api_provider: str = "vendor_a",
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    grader_model_name: str = "gpt-4o",
    language: str = "zh",
    fallback_models: Optional[List[str]] = None,
    global_timeout: float = 600.0,
) -> BrowseCompZHEvaluator:
    """Get or create a cached evaluator instance for the given configuration."""
    global _evaluator_cache, _default_evaluator

    fb_key = ",".join(fallback_models) if fallback_models else "none"
    cache_key = f"{api_provider}:{api_key or 'default'}:{api_base_url or 'default'}:{grader_model_name}:{language}:{fb_key}:{global_timeout}"
    if cache_key not in _evaluator_cache:
        _evaluator_cache[cache_key] = BrowseCompZHEvaluator(
            grader_model_name=grader_model_name,
            language=language,
            api_provider=api_provider,
            api_key=api_key,
            api_base_url=api_base_url,
            fallback_models=fallback_models,
            global_timeout=global_timeout,
        )
    return _evaluator_cache[cache_key]


def compute_score(solution_str, ground_truth, extra_info, api_config=None):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        extra_info: extra information dict (must contain 'question')
        api_config: optional dict with API configuration:
            - api_provider: 'vendor_a' or 'vendor_b' (default: 'vendor_a')
            - api_key: API key (default: provider default)
            - api_base_url: base URL override (default: provider default)
            - grader_model_name: model name for grading (default: 'gpt-4o')
            - language: language for judge prompt (default: 'zh')
    """
    answer = extract_solution(solution_str=solution_str)
    question = extra_info.get('question', None)

    if question is None or answer is None:
        return 0

    # Get evaluator based on api_config
    if api_config:
        eval_instance = _get_evaluator(
            api_provider=api_config.get('api_provider', 'vendor_a'),
            api_key=api_config.get('api_key'),
            api_base_url=api_config.get('api_base_url'),
            grader_model_name=api_config.get('grader_model_name', 'gpt-4o'),
            language=api_config.get('language', 'zh'),
            fallback_models=api_config.get('fallback_models'),
            global_timeout=api_config.get('global_timeout', 600.0),
        )
    else:
        eval_instance = _default_evaluator

    judge_result, _ = eval_instance.grade_answer(question, ground_truth['target'], answer)
    if judge_result['correct'] == 'yes':
        score = 1.0
    else:
        score = 0.0

    # do_print = random.randint(1, 64) == 1
    do_print = True

    if do_print:
        import json
        log_info = {
            "golden_answers": ground_truth['target'],
            "extracted_answer": answer,
            "solution_string": solution_str,
            "question": question,
            "judge_result": judge_result,
            "score": score,
        }
        # print(json.dumps(log_info, ensure_ascii=False))

    return score
