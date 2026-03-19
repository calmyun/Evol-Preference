from openai import OpenAI,AsyncOpenAI

aplai_api_key = "sk-qYKGucMk03cq2pozYLR4YKKUsOJsYLJFZ7poxDHbMaq1V4JH"
# 同步客户端
client = OpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
# 异步客户端
async_client = AsyncOpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
model_name = "gemini-3.1-flash-lite-preview"
def get_completion(prompt,system=""):
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512
        )
        return completion.choices[0].message.content
    except Exception as e:
        # 处理API错误
        error_message = str(e)
        print(f"gemini的API返回了错误结果: {error_message}")

async def get_completion_async(prompt, system=""):
    try:
        completion = await async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256
        )
        return completion.choices[0].message.content
    except Exception as e:
        # 处理API错误
        error_message = str(e)
        print(f"gemini的API返回了错误结果: {error_message}")