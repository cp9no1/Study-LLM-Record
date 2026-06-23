import time
import concurrent.futures
from threading import Lock
from openai import OpenAI

# ----------------- 配置区 -----------------
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"              # 不需要鉴权时通常填 "EMPTY"
MODEL_NAME = "Qwen3.5-4B-AWQ-4bit"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)
CONCURRENT_REQUESTS = 10       # 并发数量（线程数），此处自定义改动
TOTAL_REQUESTS = 50            # 总计请求次数
MAX_TOKENS = 512              # 模型最大生成的 token 限制 (为 prompt 留出空间)
PROMPT = "请事无巨细的讲述注意力机制，不少于500字。"
# ------------------------------------------

# 全局统计变量与锁
success_count = 0
failed_count = 0
total_completion_tokens = 0
stat_lock = Lock()

def send_request(request_id):
    """
    单个请求的任务函数
    """
    messages = [
        {"role": "user", "content": PROMPT}
    ]
    
    start_time = time.time()
    try:
        # 发送请求，使用 OpenAI SDK
        chat_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.95,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        
        # 解析返回的数据提取 usage。OpenAI 对象有 usage 属性
        completion_tokens = chat_response.usage.completion_tokens if chat_response.usage else 0

        latency = time.time() - start_time
        
        # 安全地更新全局变量
        global success_count, total_completion_tokens
        with stat_lock:
            success_count += 1
            total_completion_tokens += completion_tokens
            
        print(f"请求 [{request_id}] 成功完成 | 耗时: {latency:.2f}s | 生成 Tokens: {completion_tokens}")
            
    except Exception as e:
        global failed_count
        with stat_lock:
            failed_count += 1
        print(f"请求 [{request_id}] 失败: {e}")

def main():
    print(f"开始压力测试...")
    print(f"目标地址: {API_BASE_URL} (OpenAI API)")
    print(f"并发线程数: {CONCURRENT_REQUESTS}")
    print(f"总请求数: {TOTAL_REQUESTS}")
    print("-" * 50)
    
    # 记录压测总起点时间
    benchmark_start_time = time.time()
    
    # 使用线程池发起并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        # 提交所有任务到线程池
        futures = [executor.submit(send_request, i) for i in range(1, TOTAL_REQUESTS + 1)]
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
            
    # 计算总耗时
    total_time = time.time() - benchmark_start_time
    
    # 计算每秒生成的 Token 数 
    # 仅针对模型生成的 completion_tokens，这代表了当前推理服务器真实的生成吞吐量
    tps = total_completion_tokens / total_time if total_time > 0 else 0
    
    # ---------------- 打印数据统计 ----------------
    print("\n" + "=" * 50)
    print("压测结果统计")
    print("=" * 50)
    print(f"总耗时:            {total_time:.2f} 秒")
    print(f"成功请求数:        {success_count}")
    print(f"失败请求数:        {failed_count}")
    print(f"总生成 Tokens:    {total_completion_tokens} Tokens")
    print("-" * 50)
    print(f"系统生成吞吐量:    {tps:.2f} Tokens/s")
    print("=" * 50)

if __name__ == "__main__":
    main()
