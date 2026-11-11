import time
import random
import httpx
from openai import OpenAI


def call_vendor_b_api(messages, enable_thinking=False, max_retries=3, base_delay=1, backoff_factor=2, max_delay=60):
    thinking = "disabled"
    if enable_thinking:
        thinking = "enabled"

    attempt = 0
    while True:
        try:
            client = OpenAI(
                api_key="REPLACE_WITH_YOUR_API_KEY",
                base_url="https://api.lkeap.cloud.anonymous_org.com/v1",
                http_client=httpx.Client(
                    proxy="http://localhost:3128",
                ),
            )
            completion = client.chat.completions.create(
                model="deepseek-r1", #3.1-terminus
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

            is_rate_limit_error = False
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit_error = True
            elif "429" in str(e) or "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                is_rate_limit_error = True

            if not is_rate_limit_error and attempt > max_retries:
                print(f"Non-rate-limit error occurred {max_retries} times. Returning empty content.")
                return "", ""

            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter

            if is_rate_limit_error:
                print(f"Rate limit error (429). Waiting {total_delay:.2f}s before retry...")
            else:
                print(f"Waiting {total_delay:.2f}s before retry ({attempt}/{max_retries})...")

            time.sleep(total_delay)


if __name__ == "__main__":
    messages = [{"role": "user", "content": "Hello, please reply with a short greeting."}]
    answer, reason = call_vendor_b_api(messages)
    print(f"Answer: {answer}")