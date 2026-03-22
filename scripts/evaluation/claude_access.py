import asyncio
import time
from openai import OpenAI,AsyncOpenAI

aplai_api_key = "sk-wUlatpu9VP0l1SSI3PI8TD1kV8HAGkpXQy9xk5NWantLsMuE"
# 同步客户端
client = OpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
# 异步客户端
async_client = AsyncOpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
model_name = "claude-haiku-4-5-20251001"
MAX_RETRIES = 4

def get_completion(prompt, system=""):
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role":"system","content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.1
            )
            response = completion.choices[0].message.content
            print(f"✅Claude的API返回了结果: {response}",flush=True)
            return response
        except Exception as e:
            error_message = str(e)
            if attempt == MAX_RETRIES - 1:
                print(f"Claude的API返回了错误结果: {error_message}",flush=True)
                return None
            wait_time = 2 ** attempt
            print(f"Claude的API返回了错误结果: {error_message}, {wait_time}秒后重试...",flush=True)
            time.sleep(wait_time)

async def get_completion_async(prompt, system=""):
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role":"system","content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.1
            )
            response = completion.choices[0].message.content
            print(f"✅Claude的API返回了结果: {response}",flush=True)
            return response
        except Exception as e:
            error_message = str(e)
            if attempt == MAX_RETRIES - 1:
                print(f"Claude的API返回了错误结果: {error_message}",flush=True)
                return None
            wait_time = 2 ** attempt
            print(f"Claude的API返回了错误结果: {error_message}, {wait_time}秒后重试...",flush=True)
            await asyncio.sleep(wait_time)