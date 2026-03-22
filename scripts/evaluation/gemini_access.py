from openai import OpenAI, AsyncOpenAI, BadRequestError

aplai_api_key = "sk-qYKGucMk03cq2pozYLR4YKKUsOJsYLJFZ7poxDHbMaq1V4JH"
# 同步客户端
client = OpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
# 异步客户端
async_client = AsyncOpenAI(api_key=aplai_api_key, base_url="https://api.ablai.top/v1")
model_name = "gemini-3.1-flash-lite-preview"

def get_completion(prompt, system=""):
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return completion.choices[0].message.content
    except BadRequestError as e:
        # Gemini 对敏感内容可能返回 400 错误（安全过滤），返回默认评估结果避免中断
        error_message = str(e)
        print(f"[BadRequestError] gemini的API返回了错误结果: {error_message}")
        if any(keyword in error_message.lower() for keyword in ["content", "safety", "harm", "blocked", "filtered"]):
            return '{"harmlessness": false, "quality": 1, "rationale": "Content blocked by safety filter", "categories": ["safety_blocked"]}'
        return None
    except Exception as e:
        error_message = str(e)
        print(f"gemini的API返回了错误结果: {error_message}")
        return None

async def get_completion_async(prompt, system=""):
    try:
        completion = await async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return completion.choices[0].message.content
    except BadRequestError as e:
        # Gemini 对敏感内容可能返回 400 错误（安全过滤），返回默认评估结果避免中断
        error_message = str(e)
        print(f"[BadRequestError] gemini的API返回了错误结果: {error_message}")
        if any(keyword in error_message.lower() for keyword in ["content", "safety", "harm", "blocked", "filtered"]):
            return '{"harmlessness": false, "quality": 1, "rationale": "Content blocked by safety filter", "categories": ["safety_blocked"]}'
        return None
    except Exception as e:
        error_message = str(e)
        print(f"gemini的API返回了错误结果: {error_message}")
        return None