import requests
import json
import time
import random

API_URL = "http://127.0.0.1:5530/chat/stream"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "my-secret-token-2026"
}

# 设计不同的测试场景
SCENARIOS = [
    "你好，请自我介绍一下。",  # 正常提问（无工具）
    "今天上海的天气怎么样？出门需要带伞吗？",  # 触发单一工具
    "帮我查一下北京明天的天气，然后对比一下广州现在的天气。", # 复杂逻辑（可能多次触发工具）
    "999乘以888等于多少？", # 考验数学能力（无相关工具）
    "What is the weather like in shang hai today?", # 英文工具触发
]

def make_request(req_id):
    question = random.choice(SCENARIOS)
    payload = {
        "question": question,
        "thread_id": f"sim_thread_{req_id}",
        "max_tokens": 1024
    }
    
    print(f"\n[{req_id}/20] 发送请求: {question}")
    
    start_time = time.time()
    
    # 接收流式响应
    try:
        with requests.post(API_URL, headers=HEADERS, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # 简单打印前几个字示意
                    if "data: " in decoded_line and "[DONE]" not in decoded_line:
                        data = json.loads(decoded_line.replace("data: ", ""))
                        print(data.get("token", ""), end="", flush=True)
            
            print(f"\n=> 请求完成，耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"\n=> 请求失败: {e}")

if __name__ == "__main__":
    for i in range(1, 21):
        make_request(i)
        time.sleep(1) # 稍微停顿，模拟真实用户频率
    print("\n✅ 20次请求模拟完成，请登录 Langfuse 控制台查看！")