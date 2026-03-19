from openai import OpenAI,AsyncOpenAI

aplai_api_key = "sk-wUlatpu9VP0l1SSI3PI8TD1kV8HAGkpXQy9xk5NWantLsMuE"
# 同步客户端
client = OpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
# 异步客户端
async_client = AsyncOpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
model_name = "claude-haiku-4-5-20251001"
def get_completion(prompt,system=""):
    input_str = f"{system}\n{prompt}"
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": input_str}
            ],
            max_tokens=512
        )
        return completion.choices[0].message.content
    except Exception as e:
        # 处理API错误
        error_message = str(e)
        print(f"Claude的API返回了错误结果: {error_message}")

async def get_completion_async(prompt, system=""):
    input_str = f"{system}\n{prompt}"
    try:
        completion = await async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": input_str}
            ],
            max_tokens=256
        )
        return completion.choices[0].message.content
    except Exception as e:
        # 处理API错误
        error_message = str(e)
        print(f"Claude的API返回了错误结果: {error_message}")