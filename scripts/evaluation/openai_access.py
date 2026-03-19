from openai import OpenAI, AsyncOpenAI, APIError, APIConnectionError, RateLimitError, BadRequestError
import time
import threading
import asyncio

# 配置API密钥和客户端
aplai_api_key = "sk-5EGVAxfuWBBlBJdwFZzO7N7O30YiMDzXkMljcmMRmiwrDG5q"
async_client = AsyncOpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
client = OpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
model_name = "gpt-5-nano-2025-08-07"
# model_name = "deepseek-v3-2-exp"
def get_oai_completion(prompt,developer):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "developer",
                    "content": developer
                },
                {
                    "role": "user", 
                    "content": prompt.strip()
                },
            ],
            max_tokens=512,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        gpt_output = response.choices[0].message.content
        return gpt_output
    except APIError as e:
        # 处理API错误
        error_message = str(e)
        print(f"OpenAI的API返回了错误结果: {error_message}")
        if "connection error" in error_message.lower():
            print("说明连接不上client的服务器，服务器连接不了外网，查看服务器是否配置了代理？")
        raise
    
async def get_oai_completion_async(prompt, developer):
    """异步版本的API调用函数"""
    try:
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "developer",
                    "content": developer
                },
                {
                    "role": "user",
                    "content": prompt.strip()
                },
            ],
            max_tokens=512,
            # reasoning_effort="minimal",
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
        gpt_output = response.choices[0].message.content
        return gpt_output
    except APIError as e:
        # 处理API错误
        error_message = str(e)
        print(f"OpenAI的API返回了错误结果: {error_message}")
        if "connection error" in error_message.lower():
            print("说明连接不上client的服务器，服务器连接不了外网，查看服务器是否配置了代理？")
        elif "Input must have at least 1 token" in error_message:
            print("[错误] 输入内容为空或太短，无法生成有效token")
        raise
    except APIConnectionError as e:
        # 处理连接错误
        print(f"[连接失败！] 连接OpenAI的API失败: {e}")
        raise
    except RateLimitError:
        print("[流量限制] 超出速率限制，稍后重试")
        raise
    except BadRequestError as e:
        print(f"[无效请求] 参数错误: {error_message}")
        raise
    except Exception as e:
        print(f"[未知错误] 发生未知错误: {e}")
        raise

def call_chatgpt(prompt,developer="", max_retries=5):
    for attempt in range(max_retries):
        try:
            ans = get_oai_completion(prompt,developer).strip()
            
            # ans非空则请求成功
            if not ans:
                print(f"⚠️第{attempt+1}/{max_retries}次请求生成的响应为None")
                raise Exception("生成的响应为空！！！")

            return ans
        except Exception as e:
            # 在最后一次尝试失败时抛出异常
            if attempt == max_retries - 1:
                print(f"❌ 多次重试后仍失败，请检查网络或 API 配额。")
                raise
            # 指数退避重试
            wait = 2 ** attempt
            print(f"⚠️ 出错: {e}, {wait} 秒后重试...")
            time.sleep(wait)
    return ""

async def call_chatgpt_async(prompt, developer="", max_retries=5):
    """异步版本的ChatGPT调用函数，包含重试机制"""
    for attempt in range(max_retries):
        try:
            ans = (await get_oai_completion_async(prompt, developer)).strip()
            
            # ans非空则请求成功
            if not ans:
                print(f"⚠️异步第{attempt+1}/{max_retries}次请求生成的响应为None")
                raise Exception("生成的响应为空！！！")

            return ans
        except Exception as e:
            # 在最后一次尝试失败时抛出异常
            if attempt == max_retries - 1:
                print(f"❌ 异步多次重试后仍失败，请检查网络或 API 配额。")
                raise
            # 指数退避重试
            wait = 2 ** attempt
            print(f"⚠️ 异步出错: {e}, {wait} 秒后重试...")
            await asyncio.sleep(wait)
    return ""