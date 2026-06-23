import time
from openai import OpenAI

# ----------------- 配置区 -----------------
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"  # 不需要鉴权时通常填 "EMPTY"
MODEL_NAME = "Qwen3.5-4B-AWQ-4bit" 
PROMPT = "请事无巨细的讲述注意力机制，不少于500字。"
# ------------------------------------------

print(f"初始化 OpenAI 客户端，连接到 {API_BASE_URL}...")
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

print("\n" + "="*50)
print(f"模型({MODEL_NAME}) 通过 OpenAI API 进行请求验证！")
print(f"测试提示词: '{PROMPT}'")
print("="*50 + "\n")

history = [
    {"role": "system", "content": "你是一个人工智能助手。请给出清晰、完整的回答。"},
    {"role": "user", "content": PROMPT}
]

print("Qwen API 生成中...", end="", flush=True)

# ------------------ 计时开始 ------------------
start_time = time.time()

try:
    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        max_tokens=512,  # 为 Prompt 留出 Context 长度
        temperature=0.7,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    end_time = time.time()
    # ------------------ 计时结束 ------------------

    # 提取新生成的文本与usage统计
    response_text = chat_response.choices[0].message.content
    
    # 获取 Tokens
    if chat_response.usage and hasattr(chat_response.usage, 'completion_tokens') and chat_response.usage.completion_tokens:
        completion_tokens = chat_response.usage.completion_tokens
    else:
        completion_tokens = len(response_text)

    # 计算性能指标
    generation_time = end_time - start_time
    tps = completion_tokens / generation_time if generation_time > 0 else 0

    # 打印该轮的吞吐指标
    print("\n" + "-" * 50)
    print(f"   生成 Token 数 : {completion_tokens}")
    print(f"   推理精准耗时  : {generation_time:.2f} 秒")
    print(f"   推理速度      : {tps:.2f} Tokens/s")
    print("-" * 50)

except Exception as e:
    print(f"\n\n[错误] API 请求失败: {e}")
