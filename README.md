# 🟢 预备阶段：环境与基础准备

## 1. 安装 CUDA、Anaconda，配置 PyTorch 环境

创建支持 Qwen3.5 的 conda 环境：
```bash
conda create -n LLM311 python=3.11 -y
```

安装支持 GPU 的 pytorch：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 2. 通过 pip 在 conda 环境中安装相关库

安装 ModelScope、accelerate、transformers、tiktoken：
- **ModelScope**：社区模型库，下载模型代码与权重。
- **accelerate**：分布式训练与推理库，主要用于控制模型在 CPU 和 GPU 间的内存分配及异构推理加速。
- **tiktoken**：OpenAI 开源的高效 BPE 分词库，主要作用是将文本转换为模型可理解的 Token IDs 序列（随后才转为张量计算）。
- **transformers**：Hugging Face 提供的基础模型架构库，包含统一的高层 API 来加载运行各大预训练模型。

```bash
pip install accelerate modelscope tiktoken 
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
```

安装量化库 bitsandbytes，准备手动量化模型：
```bash
pip install bitsandbytes  
```

编码问题需要先设置环境变量：
```powershell
$env:CMAKE_ARGS="-DCMAKE_CXX_FLAGS='/utf-8'"
```

然后再安装 llama-cpp-python：
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir
```

## 3. 使用 ModelScope 下载 Qwen3.5-4B 及unsloth的 GGUF 模型

```bash
modelscope download --model QwenLM/Qwen3.5-4B --local_dir ./Qwen3.5-4B
modelscope download --model unsloth/Qwen3.5-4B-GGUF --local_dir ./Qwen3.5-4B-GGUF
```

## 4. 使用 transformers 库加载 Qwen3.5-4B 模型

代码示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import torch

model_path = r"E:\workPro\LLM\Qwen3.5-4B" 

print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    local_files_only=True, 
    trust_remote_code=True
)

# 1. 配置 4-bit 量化参数
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 开启 4-bit 量化加载
    bnb_4bit_compute_dtype=torch.float16, # 计算时仍然使用半精度保证效果
    bnb_4bit_quant_type="nf4",          # 使用 nf4 量化类型
    bnb_4bit_use_double_quant=True      # 开启双重量化，进一步节省显存
)

print("正在使用 4-bit 量化加载模型，请稍候...")
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    quantization_config=quantization_config # 传入量化配置
)

# 2. 初始化流式输出器 (打字机效果)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 3. 初始化对话历史
history = [
    {"role": "system", "content": "你是一个人工智能助手。请给出清晰、完整的回答。"}
]

print("\n" + "="*50)
print("模型加载完毕！现在可以开始对话了。")
print("输入 'quit' 或 'exit' 退出对话。")
print("="*50 + "\n")

# 4. 进入循环对话
while True:
    user_input = input("\nYou: ")
    
    # 退出机制
    if user_input.lower() in ['quit', 'exit']:
        print("结束对话，再见！")
        break
        
    # 防止输入空字符
    if not user_input.strip():
        continue

    # 将用户的问题加入对话历史
    history.append({"role": "user", "content": user_input})

    # 套用 Qwen 的官方对话模板（包含历史记录）
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 转换成张量送入模型所在的设备 (GPU/CPU)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Qwen: ", end="")
    
    # 5. 模型生成回答
    # 使用 no_grad 节省显存
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,                  # 生成的最大长度
            streamer=streamer,                    # 开启流式打字机输出
            pad_token_id=tokenizer.eos_token_id,  # 消除 pad_token 的警告
            do_sample=True,                       # 开启采样，让回答更自然
            temperature=0.7                       # 控制回答的随机性
        )

    # 6. 从生成的 ID 中提取刚才新生成的回答内容，存入历史记录，供下一轮对话使用
    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    history.append({"role": "assistant", "content": response_text})
```

# 🟡 第一阶段：推理服务与加速优化

## 1. 使用 Ollama 运行 Qwen3.5-4B 模型

首先新建一个没有任何后缀名的文本文件，命名为 `Modelfile`，并写入以下内容：
```dockerfile
FROM "E:\workPro\LLM\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf"
```

然后运行：
```bash
ollama create MyQwen4B -f Modelfile
```

最后运行：
```bash
ollama run MyQwen4B 
```

## 2. 使用 llama-cpp-python 库运行 Qwen3.5-4B 模型

```python
from llama_cpp import Llama

model_path = r"E:\workPro\LLM\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf" 

print("正在加载 GGUF 模型...")
# 初始化模型
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, # -1 表示把所有层卸载到 GPU 上加速，如果显存不够就写具体的数字比如 20
    n_ctx=2048,      # 上下文长度
    verbose=False    # 是否打印详细日志
)

print("模型加载完毕！输入 'quit' 退出。")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    if not user_input.strip():
        continue

    # Qwen 的 Prompt 模板
    prompt = f"<|im_start|>system\n你是一个人工智能助手。请给出清晰、完整的回答。<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    print("Qwen: ", end="")
    stream = llm(
        prompt,
        max_tokens=1024,
        stop=["<|im_end|>"],
        stream=True
    )
    
    for output in stream:
        print(output["choices"][0]["text"], end="", flush=True)
    print()
```

## 3. 安装 WSL2 与 vLLM
安装 WSL2 (Windows Subsystem for Linux) 和 Ubuntu，并运行 vLLM。

### 3.1 安装 WSL2
必须先启用“适用于 Linux 的 Windows 子系统”可选功能
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

必须启用 虚拟机平台 可选功能
```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

下载并安装 Linux 内核更新包
https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi

将 WSL 2 设置为默认版本
```powershell
wsl --set-default-version 2
```

最后在Microsoft Store中搜索Ubuntu并安装

### 3.2 在 Ubuntu 中安装 vLLM
先下载uv环境
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

使用uv创建虚拟环境
```bash
uv venv LLM311 --python 3.11
```

激活虚拟环境
```bash
source LLM311/bin/activate
```

使用uv安装vllm
```bash
uv pip install vllm
```

## 4. 开启 API Server 模式
开启 API Server 模式，用 OpenAI 兼容的格式向模型发请求。

### 4.1 下载Qwen3.5-4B-AWQ-4bit模型并使用vLLM运行

```bash
modelscope download --model cyankiwi/Qwen3.5-4B-AWQ-4bit --local_dir ./Qwen3.5-4B-AWQ-4bit
```

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen3.5-4B-AWQ-4bit \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048
```
--gpu-memory-utilization 0.6（告诉 vLLM 最多只能用 60% 的显卡显存）

**遇到报错：**
```text
RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
```
原因分析：缺少 C 编译器导致。

**解决办法：**
```bash
sudo apt update
sudo apt install build-essential
```

**遇到报错：**
```text
(EngineCore pid=8981) File ".../vllm/v1/worker/gpu_model_runner.py", line 5595, in profile_cudagraph_memory
...
(EngineCore pid=8981) torch.AcceleratorError: CUDA error: unknown error
```

原因分析：vLLM 会默认尝试开启 PyTorch 的高级加速特性（如 `torch.compile` 和 CUDA Graphs）。这个过程会预先捕获 GPU 的计算图并分配内存池（即日志里的 `profile_cudagraph_memory`）。
在 WSL2（Windows 子系统）中，虚拟化的 GPU 显存映射机制在处理量化模型（AWQ）的复杂底层算子编译和内存捕获时，非常容易触发驱动层的崩溃，从而抛出 `CUDA error: unknown error`。

**解决办法：**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen3.5-4B-AWQ-4bit \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048 \
    --enforce-eager
```
`--enforce-eager` 的作用：
强制使用 Eager 模式执行。它会告诉 vLLM：
- 跳过所有复杂的底层图编译（Disable `torch.compile`）
- 不要使用 CUDAGraph 捕获
- 直接以最基础、最标准的 PyTorch 模式一句一句运行模型

**遇到报错：**
```text
(EngineCore pid=5731) INFO 04-01 11:04:49 [gpu_worker.py:456] Available KV cache memory: -0.8 GiB
(EngineCore pid=5731) ERROR 04-01 11:04:49 [core.py:1099] EngineCore failed to start.
...
(EngineCore pid=5731) ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
```

原因分析：KV cache 显存占用过大，爆显存了。

**解决办法：修改 `max-model-len` 大小**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen3.5-4B-AWQ-4bit \
    --gpu-memory-utilization 0.6 \
    --max-model-len 1024 \
    --enforce-eager
```
使用官方推荐的指令：
```bash
vllm serve Qwen3.5-4B-AWQ-4bit --port 8000 --tensor-parallel-size 1 --max-model-len 1024 --reasoning-parser qwen3 --language-model-only --enforce-eager --gpu-memory-utilization 0.6
```
- `--tensor-parallel-size 1`：显卡数量，这里只有一张显卡，所以是 1
- `--reasoning-parser qwen3`：指定推理解析器
- `--language-model-only`：只加载语言模型

成功运行：
```text
(APIServer pid=11527) INFO 04-01 15:49:51 [api_server.py:580] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:37] Available routes are:
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=11527) INFO 04-01 15:49:51 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=11527) INFO:     Started server process [11527]
(APIServer pid=11527) INFO:     Waiting for application startup.
(APIServer pid=11527) INFO:     Application startup complete.
```

### 4.2 Benchmark
Python 多线程脚本，模拟 50 个用户同时并发请求。然后对比直接用 transformers 原生推理和用 vLLM 推理的吞吐量差异。

编写python脚本，使用openai的api，对模型进行请求：

```python
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
```

**运行结果：**

```text
开始压力测试...
目标地址: http://localhost:8000/v1 (OpenAI API)
并发线程数: 10
总请求数: 50
--------------------------------------------------
请求 [5] 成功完成 | 耗时: 29.10s | 生成 Tokens: 382
请求 [1] 成功完成 | 耗时: 29.25s | 生成 Tokens: 386
请求 [9] 成功完成 | 耗时: 29.39s | 生成 Tokens: 388
请求 [6] 成功完成 | 耗时: 29.84s | 生成 Tokens: 399
请求 [10] 成功完成 | 耗时: 29.84s | 生成 Tokens: 399
请求 [8] 成功完成 | 耗时: 30.33s | 生成 Tokens: 406
请求 [7] 成功完成 | 耗时: 32.26s | 生成 Tokens: 455
请求 [3] 成功完成 | 耗时: 32.38s | 生成 Tokens: 457
请求 [2] 成功完成 | 耗时: 32.63s | 生成 Tokens: 463
请求 [4] 成功完成 | 耗时: 32.77s | 生成 Tokens: 466
请求 [15] 成功完成 | 耗时: 13.90s | 生成 Tokens: 358
请求 [12] 成功完成 | 耗时: 15.20s | 生成 Tokens: 390
请求 [14] 成功完成 | 耗时: 15.25s | 生成 Tokens: 392
请求 [13] 成功完成 | 耗时: 16.36s | 生成 Tokens: 419
请求 [11] 成功完成 | 耗时: 17.46s | 生成 Tokens: 446
请求 [16] 成功完成 | 耗时: 16.27s | 生成 Tokens: 423
请求 [20] 成功完成 | 耗时: 15.37s | 生成 Tokens: 402
请求 [18] 成功完成 | 耗时: 16.57s | 生成 Tokens: 432
请求 [17] 成功完成 | 耗时: 17.15s | 生成 Tokens: 446
请求 [19] 成功完成 | 耗时: 17.04s | 生成 Tokens: 444
请求 [22] 成功完成 | 耗时: 15.07s | 生成 Tokens: 359
请求 [21] 成功完成 | 耗时: 16.60s | 生成 Tokens: 397
请求 [25] 成功完成 | 耗时: 15.07s | 生成 Tokens: 358
请求 [24] 成功完成 | 耗时: 16.55s | 生成 Tokens: 395
请求 [23] 成功完成 | 耗时: 18.04s | 生成 Tokens: 432
请求 [26] 成功完成 | 耗时: 17.50s | 生成 Tokens: 416
请求 [30] 成功完成 | 耗时: 16.13s | 生成 Tokens: 379
请求 [27] 成功完成 | 耗时: 18.43s | 生成 Tokens: 437
请求 [28] 成功完成 | 耗时: 18.43s | 生成 Tokens: 436
请求 [29] 成功完成 | 耗时: 17.96s | 生成 Tokens: 424
请求 [31] 成功完成 | 耗时: 16.06s | 生成 Tokens: 406
请求 [32] 成功完成 | 耗时: 15.89s | 生成 Tokens: 402
请求 [33] 成功完成 | 耗时: 14.94s | 生成 Tokens: 378
请求 [34] 成功完成 | 耗时: 15.55s | 生成 Tokens: 395
请求 [37] 成功完成 | 耗时: 13.74s | 生成 Tokens: 355
请求 [35] 成功完成 | 耗时: 17.36s | 生成 Tokens: 442
请求 [36] 成功完成 | 耗时: 17.74s | 生成 Tokens: 456
请求 [38] 成功完成 | 耗时: 15.31s | 生成 Tokens: 397
请求 [40] 成功完成 | 耗时: 15.47s | 生成 Tokens: 401
请求 [39] 成功完成 | 耗时: 17.41s | 生成 Tokens: 448
请求 [41] 成功完成 | 耗时: 14.42s | 生成 Tokens: 377
请求 [42] 成功完成 | 耗时: 14.85s | 生成 Tokens: 391
请求 [43] 成功完成 | 耗时: 14.89s | 生成 Tokens: 393
请求 [46] 成功完成 | 耗时: 13.46s | 生成 Tokens: 357
请求 [44] 成功完成 | 耗时: 16.10s | 生成 Tokens: 425
请求 [45] 成功完成 | 耗时: 14.82s | 生成 Tokens: 392
请求 [49] 成功完成 | 耗时: 13.61s | 生成 Tokens: 363
请求 [47] 成功完成 | 耗时: 16.21s | 生成 Tokens: 431
请求 [50] 成功完成 | 耗时: 14.07s | 生成 Tokens: 380
请求 [48] 成功完成 | 耗时: 18.62s | 生成 Tokens: 491

==================================================
压测结果统计
==================================================
总耗时:            100.51 秒
成功请求数:        50
失败请求数:        0
总生成 Tokens:    20466 Tokens
--------------------------------------------------
系统生成吞吐量:    203.63 Tokens/s
==================================================
```

### 4.3 单次请求对比
```python
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

```
**运行结果：**
```text
初始化 OpenAI 客户端，连接到 http://localhost:8000/v1...

==================================================
模型(Qwen3.5-4B-AWQ-4bit) 通过 OpenAI API 进行请求验证！
测试提示词: '请事无巨细的讲述注意力机制，不少于500字。'
==================================================

Qwen API 生成中...
--------------------------------------------------
   生成 Token 数 : 439
   推理精准耗时  : 27.05 秒
   推理速度      : 16.23 Tokens/s
--------------------------------------------------
```

### 4.4 使用 transformers 原生推理服务

直接使用 `transformers serve`：

```bash
transformers serve --force-model Qwen3.5-4B --port 8000 --continuous-batching
```

```text
Loading Qwen3.5-4B@main
The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 723/723 [00:04<00:00, 161.25it/s]
Some parameters are on the meta device because they were offloaded to the cpu.
INFO:     Started server process [20089]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
[Request received] Model: Qwen3.5-4B, CB: True
Continuous batching is not supported for non-text-only models. Falling back to regular generate.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Passing `generation_config` together with generation-related arguments=({'return_dict_in_generate'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation.
INFO:     127.0.0.1:36990 - "POST /v1/chat/completions HTTP/1.1" 200 OK
Qwen3.5-4B was removed from memory after 300 seconds of inactivity
```


```python
import time
from openai import OpenAI

# ----------------- 配置区 -----------------
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"  # 不需要鉴权时通常填 "EMPTY"
MODEL_NAME = "Qwen3.5-4B" 
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
```

**运行结果：**
```text
初始化 OpenAI 客户端，连接到 http://localhost:8000/v1...

==================================================
模型(Qwen3.5-4B) 通过 OpenAI API 进行请求验证！
测试提示词: '请事无巨细的讲述注意力机制，不少于500字。'
==================================================
Qwen API 生成中...

--------------------------------------------------
   生成 Token 数 : 732
   推理精准耗时  : 287.91 秒
   推理速度      : 2.54 Tokens/s
--------------------------------------------------
```

### 结论
使用 vLLM 可以显著提升吞吐量。当前对比并不严谨（vLLM 加载的是量化后的模型，而 transformers 加载的是原始模型）。以下是尝试过的组合总结：

#### 1. vLLM + 完整模型 (Qwen3.5-4B)
- **结果**：直接失败（模型加载期 CUDA error）。
- **本质原因**：物理显存不足。
- **机制解析**：4B 半精度模型本身需要约 8GB 显存。vLLM 在启动时不仅要全量加载权重，还会根据 `--gpu-memory-utilization` 预先占用一大块显存用于 KV Cache。因为物理显存已经提前占用了大部分，所以连静态权重都放不下，直接在起步阶段报错。此时即使加上 `--enforce-eager`（关闭 CUDA Graph 预编译以节省一点点显存），也无法正常加载。

#### 2. vLLM + 量化模型 (Qwen3.5-4B-AWQ-4bit)
- **结果**：完美运行。
- **本质原因**：静态体积压缩 + 专属算子直接计算。
- **机制解析**：量化将静态权重压缩到了约 2.5GB，而且 vLLM 内置了专门优化的 AWQ/Marlin 底层 CUDA 算子。在生成文本时，无需解包，直接在 4-bit 数据上进行矩阵乘法。因此，动态显存占用平稳，能正常运行。

#### 3. Transformers + 完整模型 (Qwen3.5-4B)
- **结果**：不报错，但速度极慢。
- **本质原因**：触发了“显存不足则转投内存”的机制。
- **机制解析**：Transformers 偏向于“保证跑通代码”。当检测到 8GB 显存装不下 8GB 的模型时，会自动将一部分模型层切分并卸载到系统物理内存中。计算时，数据频繁在 GPU 和 CPU 之间通过 PCIe 搬运。虽然能运行，但是极慢，几乎不可用。

#### 4. Transformers + 量化模型 (Qwen3.5-4B-AWQ-4bit)
- **结果**：收到推理请求时会 CUDA error / OOM。
- **本质原因**：保底机制失效 + 动态解包导致显存瞬间飙升。
- **机制解析**：因为静态权重只有 2.5GB，Transformers 判定 8GB 显存安全，于是没有触发 CPU Offload 机制，将模型全量放在了 GPU 上。但是，由于缺乏高效的 4-bit 计算算子，当真正开始生成文本时，为了计算，会在显存中强行把 4-bit 临时“解压”回 16-bit 甚至 32-bit。这引发了巨大的显存占用，瞬间耗尽剩余的可用显存，导致在推理时崩溃。

## 5. AWQ 或 GPTQ 量化

### 5.1 AWQ 与 GPTQ 的区别
这两种技术都属于训练后量化，通常用于将 16-bit (FP16/BF16) 的模型权重压缩到 4-bit，以大幅降低显存占用并提升推理速度。主要区别在于量化策略和性能侧重：

#### GPTQ (Generative Pre-trained Transformer Quantization)

**原理**：主要关注权重本身。通过计算二阶信息（海森矩阵的逆）来评估哪些权重的误差对模型整体影响最小，然后依次进行量化并补偿误差。

**特点**：压缩率高，生态支持非常成熟。但是纯基于权重的静态分析，可能在某些极端场景下损失部分精度。

#### AWQ (Activation-aware Weight Quantization)

**原理**：不仅关注权重，还关注激活值 (Activations)。AWQ 观察模型在少量真实数据上的运行情况，找出大约 1% 对模型输出起决定性作用的“显著权重 (Salient Weights)”，保持这些权重为 FP16，只对剩余的 99% 进行 4-bit 量化。

**特点**：通常能比 GPTQ 保留更好的模型精度（困惑度更低），尤其在较小参数量的模型上表现更佳。此外，AWQ 在许多现代推理框架（如 vLLM）中的硬件执行效率极高。

### 5.2 量化

当前环境：
```text
Using Python 3.11.15 environment at: LLM311
Name: transformers
Version: 5.5.0.dev0
```
transformers 5.5.0.dev0 暂时还不支持使用autoawq量化qwen3.5，需要等待更新,而且autoawq已经停止维护了。

退出环境当前环境，重新创建适配的环境：
```bash
deactivate
uv venv LLM_Quant --python 3.11
source LLM_Quant/bin/activate
uv pip install torch torchvision torchaudio
uv pip install transformers==4.44.2 peft==0.11.1 auto-gptq optimum
```

暂不支持，待后续更新后尝试量化。

# 🟠 第二阶段：RAG
目标： 解决大模型的幻觉问题，能基于私有知识库回答问题。

## 1. 密集检索（Dense Retrieval）和稀疏检索（Sparse Retrieval）的区别

#### 密集检索（Dense Retrieval）

**原理**：算两个连续向量之间的余弦相似度或内积。

**特点**：密集检索能捕捉到深层意图。即使用户输入的措辞很模糊，只要意思对，模型就能把它找出来。这极大地提升了问答系统的“人性化”程度。

**代表技术**： 双编码器（Bi-Encoders）、DPR（Dense Passage Retrieval）。

#### 稀疏检索（Sparse Retrieval）

**原理**：稀疏检索基于关键词匹配。它将文档表示为高维（维度等同于词表大小，通常为 $10^5$ 以上）且极其稀疏的向量。

**特点**：稀疏检索在处理精确匹配和特定术语方面表现出色。如果用户的问题包含特定的专有名词或缩写，稀疏检索能更准确地找到包含这些词的文档。

**代表技术**： TF-IDF, BM25（目前最常用的工业标准）。

## 2. 使用 Python 的 sentence-transformers 库加载模型，输入两句相似的话和一句不相关的话，计算它们之间的余弦相似度

安装必要的库：
```bash
uv pip install sentence-transformers scikit-learn
```

运行代码：
```python
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from sklearn.metrics.pairwise import cosine_similarity

# 1. 从国内 ModelScope 高速下载模型到本地缓存，并获取本地路径
model_dir = snapshot_download('Xorbits/bge-small-zh-v1.5') 

# 2. 从本地路径加载模型
model = SentenceTransformer(model_dir)

# 3. 定义测试文本
sentences = [
    "去理发。",  # 句子 A
    "做头部组织缔结术。",  # 句子 B
    "今天晚上我想吃饭。" # 句子 C
]

# 4. 生成向量 (Embeddings)
embeddings = model.encode(sentences)

# 5. 计算余弦相似度
similarity_ab = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
similarity_ac = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
similarity_bc = cosine_similarity([embeddings[1]], [embeddings[2]])[0][0]

# 6. 打印结果
print(f"句子 A: {sentences[0]}")
print(f"句子 B: {sentences[1]}")
print(f"句子 C: {sentences[2]}")
print("-" * 30)
print(f"A vs B (相似): {similarity_ab:.4f}")
print(f"A vs C (无关): {similarity_ac:.4f}")
print(f"B vs C (无关): {similarity_bc:.4f}")
```

**运行结果：**
```text
句子 A: 去理发。
句子 B: 做头部组织缔结术。
句子 C: 今天晚上我想吃饭。
------------------------------
A vs B (相似): 0.6209
A vs C (无关): 0.4353
B vs C (无关): 0.3446
```

**结果分析：**
- **相似句子 (A vs B)**：余弦相似度高达 **0.6209**。尽管没有一个词是完全相同的（"理发" vs "头部组织缔结术"），但模型捕捉到了深层的语义关联，因此得分更高。
- **无关句子 (A vs C / B vs C)**：得分更低。这表明模型成功地区分了“理发”和“吃饭”这两个完全不同的概念。

## 3. 文档切分与解析 (Chunking)
Chunk Size（块大小）和 Chunk Overlap（块重叠）如何影响检索质量。

#### Chunk Size（块大小）
块大小决定了每次检索返回多少文本。如果块太小，可能无法包含足够的信息来回答复杂问题；如果块太大，可能会引入不相关的噪声，降低检索精度。

#### Chunk Overlap（块重叠）
块重叠是指相邻块之间共享的文本量。适当的重叠可以确保在块边界处的信息不会丢失，从而提高检索的召回率。

## 4. 使用LlamaIndex写一段 Python 脚本，加载并切分 PDF。
先补充环境：
```bash
uv pip install llama-index llama-index-readers-file pymupdf
```

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import os

# 1. 设置本地 PDF 文件路径
pdf_path = "./HCA716 HCA726 倾角传感器.pdf"

if not os.path.exists(pdf_path):
    print(f"错误：未找到文件 {pdf_path}")
    exit()

# 2. 加载 PDF 文档
# SimpleDirectoryReader 是 LlamaIndex 最通用的加载器
print("正在使用 LlamaIndex 加载 PDF...")
reader = SimpleDirectoryReader(input_files=[pdf_path])
documents = reader.load_data()

# 3. 初始化切分器 (Node Parser)
# LlamaIndex 的 SentenceSplitter 会自动识别句子边界
# 针对中文，它在处理句号、感叹号等方面表现很稳
splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 4. 执行切分，将 Document 转换为 Nodes
nodes = splitter.get_nodes_from_documents(documents)

# 5. 打印结果
print(f"原始文档页数: {len(documents)}")
print(f"切分后的 Node 数量: {len(nodes)}")
print("-" * 30)

# 打印前 2 个 Node 的内容和元数据
for i, node in enumerate(nodes[:2]):
    print(f"--- Node {i+1} (ID: {node.node_id}) ---")
    # LlamaIndex 会自动提取页码等元数据
    page_label = node.metadata.get('page_label', '未知')
    print(f"[来源页码: {page_label}]")
    print(node.get_content())
    print()
```

**运行结果：**
```text
正在使用 LlamaIndex 加载 PDF...
原始文档页数: 15
切分后的 Node 数量: 30
------------------------------
--- Node 1 (ID: 253dc813-70b5-44e4-9759-15d514cd33e3) ---
[来源页码: 1]
HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
1 / 15
V2.8
HCA716/HCA726
全温补倾角传感器
技术手册 专利号：ZL201730610076.0
CE认证:ATSZAHE180205007
RoSH认证号：18300RC20410801

--- Node 2 (ID: 67c44093-1e27-4d91-8d5c-12e901c76587) ---
[来源页码: 2]
HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
2 / 15
瑞芬资质认证
○ 高新技术企业（证书编号：GR201844204379）
○ 深圳市专精特新企业（编号：SZ20210879）
○ 质量管理体系认证：IATF16949：2016（证书编号：T178487）
○ GJB9001C-2017标准 武器装备质量管理体系认证（注册号：02622J31799R0M）
○ CE认证:ATSZAHE180205007
○ RoHS认证号：18300RC20410801
○ 中国国家知识产权外观专利权（专利号：ZL 201730610076.0）
○ 修订日期：2025-5-27
注：产品功能、参数、外观等将随技术升级而调整，购买时请与本司售前业务联系确认。
```

## 5. 向量数据库部署与写入
将 4 切分好的 nodes，通过任务 2 的 Embedding 模型转化为向量，并附带元数据（Metadata，如：页码、文档名），一并存入 ChromaDB 中。并且，输入一个问题，从库里检索出 Top-3 最相似的文本块

### 5.1 什么是向量数据库？
向量数据库是一种特殊的数据库，它使用向量嵌入（Vector Embeddings）作为数据存储单位。与传统关系型数据库存储结构化数据（如文本、数字）不同，向量数据库存储的是数据的数学表示形式，即向量。这些向量通常由深度学习模型生成，能够捕捉数据的语义信息。

### 5.2 为什么需要向量数据库？
传统的数据库在处理非结构化数据时存在局限性。例如，在关系型数据库中，要查找与某个概念相关的文档，需要使用复杂的 SQL 查询和关键词匹配，这种方法无法捕捉数据的语义信息，导致检索结果不准确。而向量数据库能够通过计算向量之间的相似度来查找相关数据，从而实现更精准的语义搜索。

先补充环境：
```bash
uv pip install chromadb llama-index-vector-stores-chroma llama-index-embeddings-huggingface
```
```python
import os
import chromadb
from modelscope import snapshot_download
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("正在检查并加载 BGE 向量模型...")
model_dir = snapshot_download('Xorbits/bge-small-zh-v1.5')

# 利用GPU进行 CUDA 加速
embed_model = HuggingFaceEmbedding(
    model_name=model_dir, 
    device='cuda' 
)

# 全局设置 LlamaIndex 使用上方定义的 Embedding 模型
Settings.embed_model = embed_model
# 暂时禁用 LLM（因为这一步只是向量化存库，不需要大模型参与推理）
Settings.llm = None 

print("正在读取和切分 PDF 文档...")
pdf_path = "./HCA716 HCA726 倾角传感器.pdf"

if not os.path.exists(pdf_path):
    print(f"错误：未找到文件 {pdf_path}")
    exit()

reader = SimpleDirectoryReader(input_files=[pdf_path])
documents = reader.load_data()

splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

# 这里尝试手动给每个 Node 追加额外的业务标签
for node in nodes:
    node.metadata["project_phase"] = "knowledge_base_init"
    node.metadata["source_type"] = "academic_paper"

print("正在连接 Chroma 数据库...")
# 数据将持久化保存在当前目录下的 chroma_db 文件夹中
db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)

# 创建或获取一个集合（Collection，类似于关系型数据库中的表）
chroma_collection = chroma_client.get_or_create_collection("my_pdf_collection")

# 执行向量化并存入数据库
print("开始生成向量并写入数据库，这可能需要几秒钟...")

# 配置 LlamaIndex 的存储上下文，指向 Chroma
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 这一步是核心：VectorStoreIndex 会自动遍历所有 Nodes，
# 调用 embed_model 计算出多维向量，然后连同文本、元数据一起存入 Chroma
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context
)

print("-" * 30)
print(f"成功！已将 {len(nodes)} 个块（包含文本、向量和元数据）持久化存入 {db_path} 目录。")

# 输入问题并检索 Top-3 相似文本块
print("\n" + "="*40)
print("🔍 开始检索测试")
print("="*40)

query_text = input("请输入你的问题: ")
print(f"\n用户提问: {query_text}\n")

# 将 Index 转换为 Retriever (检索器)，并设置相似度最高的前 3 个 (Top-K)
retriever = index.as_retriever(similarity_top_k=3)

# 执行检索
# 这一步会自动将 query_text 向量化，并在 Chroma 库中进行 KNN (K近邻) 搜索
retrieved_nodes = retriever.retrieve(query_text)

for i, node_with_score in enumerate(retrieved_nodes):
    # LlamaIndex 返回的是 NodeWithScore 对象，包含原始的 Node 和匹配得分
    node = node_with_score.node
    score = node_with_score.score
    
    print(f"Top {i+1} 匹配")
    # 得分通常是距离或相似度度量（具体取决于底层的 Chroma 距离计算公式，默认为 L2 平方距离或余弦相似度）
    print(f"得分: {score:.4f}") 
    print(f"来源页码: {node.metadata.get('page_label', '未知')}")
    print(f"业务标签: {node.metadata.get('project_phase', '无')}")
    print(f"内容: {node.get_content().strip()}")
    print("-" * 30)
```

**运行结果**
```text
正在读取和切分 PDF 文档...
正在连接 Chroma 数据库...
开始生成向量并写入数据库，这可能需要几秒钟...
------------------------------
成功！已将 30 个块（包含文本、向量和元数据）持久化存入 ./chroma_db 目录。

========================================
🔍 开始检索测试
========================================
请输入你的问题: 倾斜仪的精度是多少？

用户提问: 倾斜仪的精度是多少？

Top 1 匹配
得分: 0.4614
来源页码: 3
业务标签: knowledge_base_init
内容: HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
3 / 15
▶ 产品介绍
HCA716/HCA726 是瑞芬科技针对工业现场控制领域推出小体积全温补精度高的单/双轴倾
角仪，采用 RS485/RS232 串行通行接口。内置精度高的 24bit A/D 差分转换器，通过 5 阶滤波
算法，从而可以测量传感器输出相对于水平面的倾斜和俯仰角度。该产品集成高新技 MEMS 倾
角单元，测量范围±180°，全量程精度 0.01°，可轻松实现双轴与单轴倾角测量。产品属于真正
工业级产品，性能可靠稳定，扩展性好，多种输出可供选择。适合应用于古建筑、危房、古墙等
监测以及大量程精度高的测量工业现场。
▶ 主要特性
★ 单/双轴倾角测量 ★ 量程±1～±180°可选 ★ 精度：0.008°
★ DC 9～36V 宽电压输入 ★ 宽温工作-40～+85℃ ★ 分辨率 0.
------------------------------
Top 2 匹配
得分: 0.4538
来源页码: 4
业务标签: knowledge_base_init
内容: HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
4 / 15
分辨率：指传感器在测量范围内能够检测和分辨出的被测量的最小变化值。
精度最大绝对误差：是指产品在量程内，对多个角度点测量，取测量值与实际角度误差的最大值。
精度均方根误差：是指产品在量程内，对角度多次测量(16 次以上)，取测量值与实际角度误差的均方根差。
零点温度系数：是指传感器零值状态下，在其额定工作温度范围内相对常温的示值变化率。
灵敏度温度系数：是指传感器在其额定工作温度范围内，满量程示值相对于常温满量程示值的百分比，随
温度的变化率。
▶ 性能指标
HCA716/HCA726 条件 参数 单位
测量范围 ±10 ±30 ±60 ±90 ±180 °
测量轴 X Y X Y X Y X Y X 轴
分辨率 0.001 0.001 0.001 0.001 0.
------------------------------
Top 3 匹配
得分: 0.4494
来源页码: 5
业务标签: knowledge_base_init
内容: HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
5 / 15
▶ 订购信息
例：HCA716S-10-232-68 ：单轴/水平安装/±10°测量范围/RS232 信号输出/68 协议。
注：垂直安装测量只有单轴 X 轴。
▶ 机械参数
○ 连 接 器：1m 直线引线（可定制）
○ 防护等级：IP67
○ 外壳材质：铝合金磨沙氧化
○ 安 装：四颗 M4 螺丝
▶ 工作原理
采用进口核心控制单元，运用电容微型摆锤原理。利用地球重力原理，当倾角单元倾斜时，地球
重力在相应的摆锤上会产生重力的分量，相应的电容量会变化，通过对电容量处量放大，滤波，
转换之后得出倾角。
------------------------------
```
**结果分析：**
- 检索结果与预期一致：用户输入“倾斜仪的精度是多少？”，系统成功检索到了包含精度信息的文档片段。

- 检索质量分析：Top 1-3 匹配结果显示，能够准确识别与问题相关的文本块。检索到的内容包含了“全量程精度 0.01°”、“精度：0.008°”等关键信息。

## 6. 知识库问答 (RAG)
引入框架，把“检索”和“生成”连接起来，让模型能够回答基于知识库的问题。

接入知识库上下文会增加，启动vllm时，需要增加max_tokens参数，否则会报错：
```bash
vllm serve Qwen3.5-4B-AWQ-4bit --port 8000 --tensor-parallel-size 1 --max-model-len 4096 --reasoning-parser qwen3 --language-model-only --enforce-eager --gpu-memory-utilization 0.6
```

```python
import time
import chromadb
from openai import OpenAI
from modelscope import snapshot_download
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# vLLM API 配置
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3.5-4B-AWQ-4bit" 

# 用户提问
USER_QUERY = "安装这款倾斜仪有哪些注意事项？"

print("正在加载 BGE 向量模型与 Chroma 数据库...")
# 加载 BGE 模型用于把提问转成向量
model_dir = snapshot_download('Xorbits/bge-small-zh-v1.5')
Settings.embed_model = HuggingFaceEmbedding(model_name=model_dir, device='cuda')
Settings.llm = None  # 检索阶段不需要 LlamaIndex 自带的 LLM

# 连接 Chroma 数据库
db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
chroma_collection = chroma_client.get_collection("my_pdf_collection")

# 挂载检索器
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
retriever = index.as_retriever(similarity_top_k=3)

# 执行检索并构建 RAG Prompt
print(f"\n正在检索问题: '{USER_QUERY}'")
retrieved_nodes = retriever.retrieve(USER_QUERY)

# 提取检索到的文本并拼接
context_texts = []
for i, node_with_score in enumerate(retrieved_nodes):
    context_texts.append(f"--- 资料 {i+1} ---\n{node_with_score.node.get_content().strip()}")

# 合并字符串文本
joined_context = "\n\n".join(context_texts)

print("\n构建的上下文 (Context) 如下:")
print(joined_context)

# 核心：构建 RAG 专属 Prompt
system_prompt = """你是一个严谨的人工智能助手。
请仔细阅读下方提供的参考资料，并根据这些资料回答用户的提问。
如果参考资料中没有包含回答该问题所需的信息，请诚实地回答“根据提供的资料，我无法回答这个问题”，不要捏造事实。"""

user_prompt = f"""参考资料：
{joined_context}

我的问题是：{USER_QUERY}"""

history = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# 请求 vLLM 进行生成
print(f"\n初始化 OpenAI 客户端，连接到 {API_BASE_URL}...")
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

print("\nQwen 模型思考与生成中...", end="", flush=True)

start_time = time.time()
try:
    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        max_tokens=1024,
        temperature=0.3, # RAG 场景下，temperature 通常调低(如0.1-0.3)，减少幻觉
        top_p=0.90,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    end_time = time.time()

    response_text = chat_response.choices[0].message.content
    completion_tokens = chat_response.usage.completion_tokens if chat_response.usage else len(response_text)
    generation_time = end_time - start_time
    tps = completion_tokens / generation_time if generation_time > 0 else 0

    print("\n\n" + "="*50)
    print("最终回答:")
    print(response_text)
    print("="*50)
    
    print(f"\n性能指标: 生成 {completion_tokens} Tokens | 耗时 {generation_time:.2f}s | 速度 {tps:.2f} Tokens/s")

except Exception as e:
    print(f"\n\n[错误] API 请求失败: {e}")
```

**运行结果：**
```text
正在加载 BGE 向量模型与 Chroma 数据库...
Downloading Model from https://www.modelscope.cn to directory: /home/justin/.cache/modelscope/hub/models/Xorbits/bge-small-zh-v1.5
2026-04-03 16:55:52,826 - modelscope - INFO - Target directory already exists, skipping creation.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 71/71 [00:00<00:00, 10330.80it/s]
BertModel LOAD REPORT from: /home/justin/.cache/modelscope/hub/models/Xorbits/bge-small-zh-v1___5
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
LLM is explicitly disabled. Using MockLLM.

正在检索问题: '安装这款倾斜仪有哪些注意事项？'

构建的上下文 (Context) 如下:
--- 资料 1 ---
HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
7 / 15
▶ 安装注意事项
请按照正确的方法进行安装倾角传感器，不正确的安装会导致测量误差，尤其注意一“面”，二“线”：
1）传感器的安装面与被测量面固定须紧密、平整、稳定，如安装面出现不平容易造成传感器测
量夹角误差。
2)传感器轴线与被测量轴线必须平行，两轴线尽可能不要产生夹角。
▶ 安装方向
安装时应保持传感器安装面与被测目标面平行，并减少动态和加速度对传感器的影响。本产品可
水平安装也可以垂直安装（垂直安装选型只可适用单轴），安装方式请参考下面示意图：

--- 资料 2 ---
HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
5 / 15
▶ 订购信息
例：HCA716S-10-232-68 ：单轴/水平安装/±10°测量范围/RS232 信号输出/68 协议。
注：垂直安装测量只有单轴 X 轴。
▶ 机械参数
○ 连 接 器：1m 直线引线（可定制）
○ 防护等级：IP67
○ 外壳材质：铝合金磨沙氧化
○ 安 装：四颗 M4 螺丝
▶ 工作原理
采用进口核心控制单元，运用电容微型摆锤原理。利用地球重力原理，当倾角单元倾斜时，地球
重力在相应的摆锤上会产生重力的分量，相应的电容量会变化，通过对电容量处量放大，滤波，
转换之后得出倾角。

--- 资料 3 ---
008°
★ DC 9～36V 宽电压输入 ★ 宽温工作-40～+85℃ ★ 分辨率 0.001°
★ IP67 防护等级 ★ 高抗振性能>100g ★ 直接引线式接口
▶ 应用范围
★ 古建筑、危房监测 ★ 桥梁与大坝监测 ★ 工程车辆调平
★ 医疗设备角度控制 ★ 地下钻机姿态导航 ★ 铁路轨距尺、轨距仪测平
★ 矿业机械、石油钻井设备 ★ 地质设备倾斜监测 ★ 定向卫星通讯天线的俯仰角测
量

初始化 OpenAI 客户端，连接到 http://localhost:8000/v1...

Qwen 模型思考与生成中...

==================================================
最终回答:
根据提供的参考资料，安装这款倾角传感器（HCA716/HCA726）的注意事项如下：

**1. 安装原则：一“面”，二“线”**
*   **关于“面”：** 传感器的安装面与被测量面必须固定紧密、平整且稳定。如果安装面出现不平，容易造成测量夹角误差。
*   **关于“线”：** 传感器轴线与被测量轴线必须保持平行，两轴线之间尽可能不要产生夹角。

**2. 安装方向要求**
*   **平行度：** 安装时应保持传感器安装面与被测目标面平行。
*   **减少干扰：** 安装时应尽量减少动态和加速度对传感器的影响。
*   **安装方式：** 本产品可水平安装，也可垂直安装（注：垂直安装选型仅适用于单轴）。

**3. 机械安装细节**
*   **固定方式：** 使用四颗 M4 螺丝进行固定。
*   **线缆：** 配备 1m 直线引线（也可定制）。

**4. 环境适应性**
*   **防护等级：** 产品具备 IP67 防护等级，适合在恶劣环境下使用。
*   **工作温度：** 宽温工作范围在 -40℃ 至 +85℃。
*   **抗振性能：** 具有高抗振性能（>100g）。
==================================================

性能指标: 生成 297 Tokens | 耗时 11.50s | 速度 25.82 Tokens/s
```

**结果分析：**
- 检索结果与预期一致：用户输入“安装这款倾斜仪有哪些注意事项？”，系统成功检索到了包含安装注意事项的文档片段。

- 检索质量分析：Top 1 匹配结果显示，能够准确识别与问题相关的文本块。检索到的内容包含了“安装注意事项”、“安装面与被测量面固定须紧密、平整、稳定”、“传感器轴线与被测量轴线必须平行”等关键信息。

- 生成质量分析：Qwen 模型根据检索到的上下文生成了结构清晰、内容准确的回答，准确传达了安装时的关键要求。

## 7. 对抗“长尾效应”与工程落地
解决“检索不到”和“胡说八道”的问题

### 7.1 多路召回与重排 Rerank
引入 BM25 算法作为补充，实现“向量 + 关键词”的双路召回，然后根据召回的相关片段，用 Reranker 模型（如 BAAI/bge-reranker-base）对这些片段与用户问题的相关性进行二次精确打分，筛选出最优质的 Top-3。实现重排后准确率的提升

先补充环境：
```bash
uv pip install llama-index-retrievers-bm25
```

**纯 Chroma 数据库默认不存储完整的内存文档结构（Docstore），这会导致 LlamaIndex 原生的 BM25 无法直接挂载**
所以先删除之前的 chroma_db 文件夹，然后运行：

```python
import os
import pickle
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from modelscope import snapshot_download

pdf_path = "./HCA716 HCA726 倾角传感器.pdf"
persist_dir = "./storage" # 本地持久化总目录

model_dir = snapshot_download('Xorbits/bge-small-zh-v1.5')
Settings.embed_model = HuggingFaceEmbedding(model_name=model_dir, device='cuda')
Settings.llm = None

print("正在读取和切分文档...")
reader = SimpleDirectoryReader(input_files=[pdf_path])
documents = reader.load_data()
splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)

print("正在初始化持久化存储引擎...")
db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
chroma_collection = chroma_client.get_or_create_collection("my_pdf_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 文档与索引存储 (LlamaIndex 自带的本地文件存储)
docstore = SimpleDocumentStore()
index_store = SimpleIndexStore()

# 将这些组件打包成 StorageContext
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore,
    index_store=index_store,
)

# 将 Nodes 注册进 docstore (非常重要)
storage_context.docstore.add_documents(nodes)

print("正在计算向量并存入数据库...")
index = VectorStoreIndex(nodes, storage_context=storage_context)

print("正在将所有索引持久化到磁盘...")
os.makedirs(persist_dir, exist_ok=True)
storage_context.persist(persist_dir=persist_dir)

print("建库完成！")
```

**运行结果：**
```text
Downloading Model from https://www.modelscope.cn to directory: /home/justin/.cache/modelscope/hub/models/Xorbits/bge-small-zh-v1.5
2026-04-03 17:33:37,773 - modelscope - INFO - Target directory already exists, skipping creation.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 71/71 [00:00<00:00, 11170.13it/s]
BertModel LOAD REPORT from: /home/justin/.cache/modelscope/hub/models/Xorbits/bge-small-zh-v1___5
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
LLM is explicitly disabled. Using MockLLM.
正在读取和切分文档...
正在初始化持久化存储引擎...
正在计算向量并存入数据库...
正在将所有索引持久化到磁盘...
建库完成！
```

补充环境：
```bash
uv pip install sentence-transformers
```

```python
import os
import time
import pickle
import chromadb
from openai import OpenAI
from modelscope import snapshot_download
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from sentence_transformers import CrossEncoder
from llama_index.retrievers.bm25 import BM25Retriever

API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3.5-4B-AWQ-4bit" 
USER_QUERY = "安装这款倾斜仪有哪些注意事项？"
persist_dir = "./storage"

print("1. 正在加载 BGE 模型...")
model_dir = snapshot_download('Xorbits/bge-small-zh-v1.5')
Settings.embed_model = HuggingFaceEmbedding(model_name=model_dir, device='cuda')
Settings.llm = None

print("2. 从磁盘恢复数据库与检索器...")

# A. 载入 Chroma 向量库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_collection("my_pdf_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# B. 恢复本地 Docstore (这里面存放了所有完整的文本内容)
storage_context = StorageContext.from_defaults(
    persist_dir=persist_dir,
    vector_store=vector_store
)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
vector_retriever = index.as_retriever(similarity_top_k=10)

# C. 从 Docstore 提取节点，秒级构建 BM25 检索器
# storage_context.docstore.docs 是一个包含了所有切分块的字典
nodes = list(storage_context.docstore.docs.values())
print(f"   -> 从本地磁盘成功加载了 {len(nodes)} 个文档块，正在挂载 BM25 引擎...")
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

# 融合器 (RRF算法): 自动合并两路结果，去重并重新打分，总计输出 10 个候选片段
hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,  # 不使用 LLM 重写 Query，节约算力
    mode="reciprocal_rerank",
)

print("3. 正在加载 BGE Reranker 重排模型...")
# 使用bge-reranker-base 模型进行重排
reranker_model_dir = snapshot_download('BAAI/bge-reranker-base')

# 直接使用 sentence-transformers 的底层接口，避免当前的兼容性问题
cross_encoder = CrossEncoder(reranker_model_dir, max_length=512)
print(f"\n4. 正在执行双路召回与重排: '{USER_QUERY}'")
initial_nodes = hybrid_retriever.retrieve(USER_QUERY)

print("   -> 正在使用 CrossEncoder 进行精准打分...")
reranker_model_dir = snapshot_download('BAAI/bge-reranker-base')

# 加载交叉编码器模型 (自动调用 GPU)
cross_encoder = CrossEncoder(reranker_model_dir, max_length=512)

# 核心原理：构建 (提问, 资料) 的文本对
pairs = [[USER_QUERY, node.node.get_content().strip()] for node in initial_nodes]

# 批量预测相关性得分 (越相关的片段，打分越高)
scores = cross_encoder.predict(pairs)

# 将得分写回 LlamaIndex 的 Node 中
for node, score in zip(initial_nodes, scores):
    node.score = float(score)

# 按照得分从高到低进行排序 (降序)
initial_nodes.sort(key=lambda x: x.score, reverse=True)

# 4096 上下文 可以截取最精准的 Top-4
reranked_nodes = initial_nodes[:4]
context_texts = []
for i, node_with_score in enumerate(reranked_nodes):
    context_texts.append(f"--- 资料 {i+1} (相关度得分: {node_with_score.score:.3f}) ---\n{node_with_score.node.get_content().strip()}")

joined_context = "\n\n".join(context_texts)

print("\n构建的重排后上下文 (Context) 如下:")
print(joined_context)

system_prompt = """你是一个严谨的人工智能助手。
请仔细阅读下方提供的参考资料，并根据这些资料回答用户的提问。
如果参考资料中没有包含回答该问题所需的信息，请诚实地回答“根据提供的资料，我无法回答这个问题”，不要捏造事实。"""

user_prompt = f"参考资料：\n{joined_context}\n\n我的问题是：{USER_QUERY}"

history = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

print(f"\n5. 初始化 OpenAI 客户端，连接至 vLLM ({API_BASE_URL})...")
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

print("\nQwen 模型思考与生成中...", end="", flush=True)
start_time = time.time()
try:
    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        max_tokens=1024,
        temperature=0.3,
        top_p=0.90,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    end_time = time.time()

    response_text = chat_response.choices[0].message.content
    completion_tokens = chat_response.usage.completion_tokens if chat_response.usage else len(response_text)
    generation_time = end_time - start_time
    tps = completion_tokens / generation_time if generation_time > 0 else 0

    print("\n\n" + "="*50)
    print("最终回答:")
    print(response_text)
    print("="*50)
    
    print(f"\n性能指标: 生成 {completion_tokens} Tokens | 耗时 {generation_time:.2f}s | 速度 {tps:.2f} Tokens/s")

except Exception as e:
    print(f"\n\n[错误] API 请求失败: {e}")
```

**运行结果：**
```text
1. 正在加载 BGE 模型...
Downloading Model from https://www.modelscope.cn to directory: /home/justin/.cache/modelscope/hub/models/Xorbits/bge-small-zh-v1.5
2026-04-03 18:51:11,969 - modelscope - INFO - Target directory already exists, skipping creation.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 71/71 [00:00<00:00, 15386.77it/s]
BertModel LOAD REPORT from: /home/justin/.cache/modelscope/hub/models/Xorbits/bge-small-zh-v1___5
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
LLM is explicitly disabled. Using MockLLM.
2. 从磁盘恢复数据库与检索器...
   -> 从本地磁盘成功加载了 30 个文档块，正在挂载 BM25 引擎...
3. 正在加载 BGE Reranker 重排模型...
Downloading Model from https://www.modelscope.cn to directory: /home/justin/.cache/modelscope/hub/models/BAAI/bge-reranker-base
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3141.45it/s]
XLMRobertaForSequenceClassification LOAD REPORT from: /home/justin/.cache/modelscope/hub/models/BAAI/bge-reranker-base
Key                             | Status     |  | 
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

4. 正在执行双路召回与重排: '安装这款倾斜仪有哪些注意事项？'
   -> 正在使用 CrossEncoder 进行精准打分...
Downloading Model from https://www.modelscope.cn to directory: /home/justin/.cache/modelscope/hub/models/BAAI/bge-reranker-base
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 2988.58it/s]
XLMRobertaForSequenceClassification LOAD REPORT from: /home/justin/.cache/modelscope/hub/models/BAAI/bge-reranker-base
Key                             | Status     |  | 
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

构建的重排后上下文 (Context) 如下:
--- 资料 1 (相关度得分: 0.891) ---
HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
7 / 15
▶ 安装注意事项
请按照正确的方法进行安装倾角传感器，不正确的安装会导致测量误差，尤其注意一“面”，二“线”：
1）传感器的安装面与被测量面固定须紧密、平整、稳定，如安装面出现不平容易造成传感器测
量夹角误差。
2)传感器轴线与被测量轴线必须平行，两轴线尽可能不要产生夹角。
▶ 安装方向
安装时应保持传感器安装面与被测目标面平行，并减少动态和加速度对传感器的影响。本产品可
水平安装也可以垂直安装（垂直安装选型只可适用单轴），安装方式请参考下面示意图：

--- 资料 2 (相关度得分: 0.389) ---
HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
5 / 15
▶ 订购信息
例：HCA716S-10-232-68 ：单轴/水平安装/±10°测量范围/RS232 信号输出/68 协议。
注：垂直安装测量只有单轴 X 轴。
▶ 机械参数
○ 连 接 器：1m 直线引线（可定制）
○ 防护等级：IP67
○ 外壳材质：铝合金磨沙氧化
○ 安 装：四颗 M4 螺丝
▶ 工作原理
采用进口核心控制单元，运用电容微型摆锤原理。利用地球重力原理，当倾角单元倾斜时，地球
重力在相应的摆锤上会产生重力的分量，相应的电容量会变化，通过对电容量处量放大，滤波，
转换之后得出倾角。

--- 资料 3 (相关度得分: 0.150) ---
008°
★ DC 9～36V 宽电压输入 ★ 宽温工作-40～+85℃ ★ 分辨率 0.001°
★ IP67 防护等级 ★ 高抗振性能>100g ★ 直接引线式接口
▶ 应用范围
★ 古建筑、危房监测 ★ 桥梁与大坝监测 ★ 工程车辆调平
★ 医疗设备角度控制 ★ 地下钻机姿态导航 ★ 铁路轨距尺、轨距仪测平
★ 矿业机械、石油钻井设备 ★ 地质设备倾斜监测 ★ 定向卫星通讯天线的俯仰角测
量

--- 资料 4 (相关度得分: 0.056) ---
HCA716/HCA726 全温补倾角传感器
○倾角传感器 ○三维电子罗盘 ○数显水平仪 ○加速度计 ○陀螺仪 ○寻北仪 ○INS&IMU
瑞芬科技 SINCE2008 工业传感 · 智创未来
14 / 15
六．设置传感器通讯字符格式:
设置传感器通讯字符格式： 从机响应：
传感器地址 01H 传感器地址 01H
功能码 06H 功能码 06H
地址
00H 寄存器
地址
00H
09H 09H
传感器更改通
讯字符格式
00 H
传感器的新格式
00H
01H 01H
CRC 9808 CRC 9808
设置传感器通讯字符格式应用举例：
主机发送 01 H 06 H 00 H 09 H 00 H 01H 98H 08H
从机回复 01 H 06 H 00 H 09 H 00 H 01H 98 H 08H
以上举例为把字节 格式 设置成 ：一个起始位+8 个数据位 无奇偶校验 +1 个停止位
重新上电后有效。出厂默认是 一个起始位+8 个数据位 偶校验 +1 个停止位
注：0009 为寄存器地址，该寄存器控制传感器通讯字符格式。
0000H:一个起始位+8 个数据位 偶校验 +1 个停止位
0001H:一个起始位+8 个数据位 无校验 +1 个停止位
七.

5. 初始化 OpenAI 客户端，连接至 vLLM (http://localhost:8000/v1)...

Qwen 模型思考与生成中...

==================================================
最终回答:
根据提供的参考资料，安装这款 HCA716/HCA726 全温补倾角传感器时，有以下主要注意事项：

**1. 安装面要求（“面”与“线”）**
*   **紧密、平整、稳定：** 传感器的安装面与被测量面必须固定紧密、平整且稳定。如果安装面不平，容易造成测量夹角误差。
*   **轴线平行：** 传感器轴线与被测量轴线必须保持平行，两轴线之间尽可能不要产生夹角。

**2. 安装方向与姿态**
*   **保持平行：** 安装时应保持传感器安装面与被测目标面平行。
*   **减少干扰：** 安装时应尽量减少动态和加速度对传感器的影响。
*   **安装方式：** 本产品既可以水平安装，也可以垂直安装（注：垂直安装仅适用于单轴选型）。

**3. 机械固定**
*   **螺丝固定：** 安装时需要使用四颗 M4 螺丝进行固定。

**4. 线缆处理**
*   传感器配备 1 米直线引线（也可定制），安装时需确保引线连接稳固。

**5. 环境适应性**
*   该传感器具有 IP67 防护等级，高抗振性能（>100g），宽温工作范围（-40～+85℃），安装时应考虑其工作环境是否符合这些参数要求。
==================================================

性能指标: 生成 298 Tokens | 耗时 11.85s | 速度 25.15 Tokens/s
```

## 8. 总结与工程复盘

### 8.1 RAG 架构设计与持久化

#### BM25 与 Chroma 的隔阂：引入 SimpleDocumentStore
- **遇到的问题**：Chroma 作为纯向量数据库，默认丢弃了文档层级的完整树状结构，导致 LlamaIndex 的 BM25 无法直接挂载语料。
- **深层原因**：工业界通常是“向量存 Milvus/Chroma，文本存 ES”。为了在本地模拟这种解耦，不能依赖极其耗费内存的动态提取。
- **解决方案**：引入 LlamaIndex 的 `StorageContext`，在离线建库时，将向量存入 Chroma，同时将完整的文本节点切块序列化并存入本地磁盘（Docstore JSON）。

#### 序列化报错：放弃 Pickle 保存 Cython 对象
- **遇到的问题**：尝试用 pickle 持久化 `BM25Retriever` 时，报 `TypeError: self.cobj cannot be converted` 错误。
- **深层原因**：BM25 底层使用了 C 语言编写的 `PyStemmer`（用于词根提取），Python 的原生 Pickle 无法打包 C 语言的内存指针对象。
- **解决方案**：不再持久化 BM25 引擎本身，而是利用上一步存好的 Docstore，在服务冷启动时，直接从磁盘将纯文本 JSON 拉入系统内存中实时重建 BM25 索引。利用物理硬件资源充足的优势，避开底层库的序列化 Bug。

### 8.2 工程化重构

#### Reranker 依赖版本冲突问题：抛弃 LlamaIndex 高级封装
- **遇到的问题**：经历了缺包、提示从 GitHub 安装、最后在执行重排时触发 `AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model` 异常。
- **深层原因**：LlamaIndex 频繁的模块重构，以及其依赖的 `FlagEmbedding` 库在版本更替中，并不兼容当下在 vLLM 中运行的最新版 `transformers`。这种底层依赖冲突在 AI 落地中非常常见。
- **最终解决方案**：引入纯粹的底层推理库 `sentence-transformers`，手动编写降级的 `CrossEncoder` 预测逻辑：构建文本对 `[Query, Doc]` -> `predict` 打分 -> `sort` 降序排序 -> 切片获取 `Top-4`，从而成功剥离对脆弱高层 API 的依赖。

# 🟣 第三阶段：Agent
目标： 掌握 Function Calling 机制，理解 ReAct 思想，并熟练使用工业级状态图框架构建可控智能体。

## 1.Function Calling
Agent与大模型之间的一种交互方式，Agent可以调用外部工具来获取信息，然后将信息返回给大模型，大模型再根据信息生成回答。

**详细步骤：**
1.Agent发请求给模型，并附上它可用的工具。
2.模型返回一个 tool call。
3.应用代码执行这个工具。
4.把工具执行结果再发回模型。
5.模型产出最终回答，或者继续请求更多工具，如此循环。

## 2.tools 和 tool_choice
tools 和 tool_choice 是 OpenAI API 中用于实现 Function Calling 的参数。

### 2.1 tools
tools 是一个列表，用于指定 Agent 可以使用的工具。

### 2.2 tool_choice
tool_choice 是一个字符串，用于指定 Agent 使用工具的方式，可选值有：
- "auto": 自动选择工具。
- "none": 不使用工具。
- "tool_name": 使用指定的工具。
甚至也可以是一组工具的子集，表示优先使用这些工具。

## 3.实战：调用高德 API 获取指定地区的天气信息
要启用function calling，需要使用vllm serve的--enable-auto-tool-choice和--tool-call-parser参数

```bash
vllm serve Qwen3.5-4B-AWQ-4bit \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only \
  --enforce-eager \
  --gpu-memory-utilization 0.6
```


```python
import json
import requests
from openai import OpenAI


class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {
            "city": city,
            "key": self.api_key,
            "extensions": extensions,   # base=实况, all=预报
            "output": "JSON",
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "1":
            raise RuntimeError(
                f"Amap API error: {data.get('info')} ({data.get('infocode')})"
            )
        return data

    def get_current_weather(self, city: str) -> dict:
        return self._request(city, "base")

    def get_weather_forecast(self, city: str) -> dict:
        return self._request(city, "all")


client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

AMAP_KEY = ""


def get_current_weather(city: str) -> dict:
    api = GaodeAPI(AMAP_KEY)
    return api.get_current_weather(city)


def get_weather_forecast(city: str) -> dict:
    api = GaodeAPI(AMAP_KEY)
    return api.get_weather_forecast(city)


TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current real-time weather conditions of a city. Use this ONLY for questions about the exact weather right now (e.g., current temperature, current wind, current humidity",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, for example: 北京, 上海, Shenzhen"
                }
            },
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_weather_forecast",
        "description": "Get the multi-day weather forecast of a city, including today's overall summary and upcoming days. Use this for questions about high/low temperatures, day/night weather, tomorrow, and future days.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, for example: 北京, 上海, Shenzhen"
                }
            },
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    }
]


def run_tool(tool_name: str, arguments_json: str):
    args = json.loads(arguments_json)
    city = args["city"]

    if tool_name == "get_current_weather":
        raw_res = get_current_weather(city)
        # 只保留 lives 里的关键信息
        if "lives" in raw_res:
            return raw_res["lives"][0] 
        return raw_res

    if tool_name == "get_weather_forecast":
        raw_res = get_weather_forecast(city)
        # 只给模型看关键的 casts 列表
        if "forecasts" in raw_res:
            forecast = raw_res["forecasts"][0]
            return {
                "city": forecast["city"],
                "casts": [
                    {k: v for k, v in c.items() if "float" not in k}
                    for c in forecast["casts"]
                ]
            }
        return raw_res


def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue
        response = client.responses.create(
            model="Qwen3.5-4B-AWQ-4bit",
            input=user_input,
            tools=TOOLS,
            tool_choice="auto",
        )

        while True:
            function_calls = [item for item in response.output if item.type == "function_call"]

            if not function_calls:
                print("模型最终回答：")
                print(response.output_text)
                break

            tool_outputs = []
            for call in function_calls:
                print(f"模型调用工具: {call.name}")
                print(f"工具参数: {call.arguments}")

                result = run_tool(call.name, call.arguments)

                # 避免把敏感字段打出来
                safe_result = json.loads(json.dumps(result, ensure_ascii=False))
                for key in ("key", "sec_code", "sec_code_debug"):
                    safe_result.pop(key, None)

                print("工具返回：")
                print(json.dumps(safe_result, ensure_ascii=False, indent=2))

                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            response = client.responses.create(
                model="Qwen3.5-4B-AWQ-4bit",
                input=[
                    {"role": "user", "content": user_input},
                    *response.output,
                    *tool_outputs,
                ],
                tools=TOOLS,
                tool_choice="auto",
            )


if __name__ == "__main__":
    main()
```

**输出结果**
```text
You: 今天明天上海的天气怎么样？
模型调用工具: get_weather_forecast
工具参数: {"city": "上海"}
工具返回：
{
  "city": "上海市",
  "casts": [
    {
      "date": "2026-04-09",
      "week": "4",
      "dayweather": "中雨",
      "nightweather": "大雨",
      "daytemp": "29",
      "nighttemp": "17",
      "daywind": "南",
      "nightwind": "南",
      "daypower": "1-3",
      "nightpower": "1-3"
    },
    {
      "date": "2026-04-10",
      "week": "5",
      "dayweather": "阴",
      "nightweather": "阴",
      "daytemp": "25",
      "nighttemp": "14",
      "daywind": "西北",
      "nightwind": "西北",
      "daypower": "1-3",
      "nightpower": "1-3"
    },
    {
      "date": "2026-04-11",
      "week": "6",
      "dayweather": "小雨",
      "nightweather": "中雨",
      "daytemp": "17",
      "nighttemp": "14",
      "daywind": "东",
      "nightwind": "东",
      "daypower": "1-3",
      "nightpower": "1-3"
    },
    {
      "date": "2026-04-12",
      "week": "7",
      "dayweather": "小雨",
      "nightweather": "阴",
      "daytemp": "19",
      "nighttemp": "16",
      "daywind": "东南",
      "nightwind": "东南",
      "daypower": "1-3",
      "nightpower": "1-3"
    }
  ]
}
模型最终回答：


根据上海最新天气预报：

**今天（4月9日）：**
- 🌧️ **白天：** 中雨，**夜间：** 大雨
- 🌡️ **气温：** 白天 29°C → 夜间 17°C
- 💨 **风向：** 南风（白天、夜间风力1-3级）
- ⚠️ **重点：** 明天会有大雨天气，建议带雨具，注意出行安全

**明天（4月10日）：**
- ☁️ **白天：** 阴天，**夜间：** 阴天
- 🌡️ **气温：** 白天 25°C → 夜间 14°C
- 💨 **风向：** 西北风（白天、夜间风力1-3级）
- ✅ **说明：** 明天天气较凉爽，但仍偏阴，建议携带雨具以防夜间降雨

**总结**：今天和明天上海以阴雨天气为主，明天尤其注意防范大雨，6月份 temperature 统计数据仅供参考，实际气温可能略有波动。建议近期多关注实时天气更新，合理安排出行。

如需更详细的后续天气信息（如后几天），可随时说明！
```

## 4.ReAct
ReAct 是一种把**推理（Reason）和行动（Act）**交替组织起来的提示词技巧：

```text
你是一个会使用工具解决问题的助手。你可以在推理过程中决定是否调用工具。

你可用的工具有：
1. Search：搜索互联网信息
2. Lookup：查询指定文档或知识库中的内容
3. Calculator：进行数值计算

请严格按照以下格式作答：

Question: 用户问题
Thought: 你当前的思考。说明你要解决什么子问题，是否需要调用工具。
Action: 你要调用的工具名，必须是 [Search, Lookup, Calculator] 之一；如果已经可以回答，则不要写 Action。
Action Input: 工具输入内容
Observation: 工具返回结果
...（Thought / Action / Action Input / Observation 可以重复多轮）
Thought: 我已经可以给出最终答案
Final Answer: 针对原问题的最终回答

要求：
- 优先基于 Observation 作答，不要凭空猜测
- 每次 Action 都要服务于当前子问题
- 信息足够时立刻停止调用工具
- Final Answer 要简洁、明确、直接回答用户问题

Question: {input}
......
```

## 5.LangGraph
将应用定义为一个**图 (Graph)**，节点是执行特定任务的**函数或 Agent**，边是**控制流**，且引入了跨节点的**全局状态 (State)** 和**循环 (Cycles)**。

先补充环境
```bash
uv pip install langgraph langchain-openai requests
```

```python
import json
import requests
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {
            "city": city,
            "key": self.api_key,
            "extensions": extensions,
            "output": "JSON",
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "1":
            raise RuntimeError(f"Amap API error: {data.get('info')} ({data.get('infocode')})")
        return data

    def get_current_weather(self, city: str) -> dict:
        return self._request(city, "base")

    def get_weather_forecast(self, city: str) -> dict:
        return self._request(city, "all")


AMAP_KEY = "" 

# 使用 @tool 装饰器，LangChain 会自动从函数签名和 docstring 中提取 JSON Schema
@tool
def get_current_weather(city: str) -> dict:
    """获取指定城市的当前实时天气情况。用于询问当前的温度、风力、湿度等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_current_weather(city)
    if "lives" in raw_res:
        return raw_res["lives"][0]
    return raw_res

@tool
def get_weather_forecast(city: str) -> dict:
    """获取指定城市的多日天气预报。用于询问今天整体、明天或未来的天气、最高/最低温等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_weather_forecast(city)
    if "forecasts" in raw_res:
        forecast = raw_res["forecasts"][0]
        return {
            "city": forecast["city"],
            "casts": [
                {k: v for k, v in c.items() if "float" not in k}
                for c in forecast["casts"]
            ]
        }
    return raw_res

tools = [get_current_weather, get_weather_forecast]

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model="Qwen3.5-4B-AWQ-4bit"
)
llm_with_tools = llm.bind_tools(tools)

# 定义一个包含 messages 列表的 State (直接使用内置的 MessagesState)
def agent_node(state: MessagesState):
    """大模型节点：将历史消息传递给 LLM，并返回新的 AIMessage"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 初始化图构建器
builder = StateGraph(MessagesState)

# 添加节点
builder.add_node("agent", agent_node)
# 创建一个工具执行节点 (使用内置的 ToolNode，它会自动处理 tool_calls)
builder.add_node("tools", ToolNode(tools))

# 定义图的边
builder.add_edge(START, "agent")

# 设置条件边 (Conditional Edge)
# tools_condition 是 LangGraph 内置的路由：
# 如果 agent 节点返回了 tool_calls -> 流转到 "tools"
# 如果没有 -> 流转到 END
builder.add_conditional_edges("agent", tools_condition)

# 工具执行完毕后，返回给大模型继续思考
builder.add_edge("tools", "agent")

# 人类确认机制
# 使用 MemorySaver 持久化状态图，以便我们可以在特定节点前暂停它
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # 在流转到 "tools" 节点前打断点
)


def main():
    # 配置线程 ID，LangGraph 依赖这个来区分不同的对话状态
    thread_config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue

        # 向图中推入用户的消息并开始流转
        events = graph.stream(
            {"messages": [("user", user_input)]}, 
            thread_config, 
            stream_mode="values"
        )
        
        # 打印初始的流转过程 (通常是 Agent 给出的回答或 Tool call 的请求)
        for event in events:
            last_message = event["messages"][-1]
            # 简单过滤，避免把用户的输入再打印一遍
            if last_message.type == "ai" and last_message.content:
                print(f"Agent: {last_message.content}")

        # 检查图是否在断点处暂停
        state = graph.get_state(thread_config)
        
        # 如果 state.next 包含 "tools"，说明它被我们设置的 interrupt_before 拦截了
        while state.next and state.next[0] == "tools":
            last_message = state.values["messages"][-1]
            tool_calls = last_message.tool_calls
            
            tool_names = [tc["name"] for tc in tool_calls]
            print(f"\n[拦截] 大模型请求执行工具: {tool_names}")
            print(f"工具参数: {json.dumps(tool_calls[0]['args'], ensure_ascii=False)}")
            
            # 提示终端用户确认
            decision = input("是否允许执行？(Y/N): ")
            
            if decision.lower() == 'y':
                print("授权通过，继续执行...")   
                # 传入 None 并使用同样的 thread_config，图会从断点处继续往下流转 (执行 tools -> 回到 agent)
                events = graph.stream(None, thread_config, stream_mode="values")
                for event in events:
                    last_msg = event["messages"][-1]
                    if last_msg.type == "ai" and last_msg.content:
                        print(f"Agent: {last_msg.content}")
                
                # 刷新状态，看是否还有后续动作
                state = graph.get_state(thread_config)
            
            else:
                print("已拒绝执行该工具。")
                # 如果拒绝，必须向状态中手动注入一条假装是 Tool 返回的拒绝消息。
                # 否则大模型会卡在等待工具返回的状态里。
                denial_messages = [
                    ToolMessage(
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        content="用户拒绝了该工具的执行。系统警告：你现在没有任何真实的天气数据，严禁利用自己的知识库猜测或编造天气情况！请直接向用户说明没有数据。"
                    )
                    for tc in tool_calls
                ]
                
                # 将拒绝消息更新到状态图中，假装是 tools 节点执行的结果，并直接跳回到 agent 节点
                graph.update_state(thread_config, {"messages": denial_messages}, as_node="tools")
                
                # 再次 stream，让 agent 根据“被拒绝”这个事实继续说话
                events = graph.stream(None, thread_config, stream_mode="values")
                for event in events:
                    last_msg = event["messages"][-1]
                    if last_msg.type == "ai" and last_msg.content:
                        print(f"Agent: {last_msg.content}")
                
                break # 拒绝后跳出当前确认循环

if __name__ == "__main__":
    main()
```    

**运行结果**
```
You: 明天上海和北京的天气怎么样？

[拦截] 大模型请求执行工具: ['get_weather_forecast', 'get_weather_forecast']
工具参数: {"city": "上海"}
是否允许执行？(Y/N): y
授权通过，继续执行...
Agent: 

根据天气预报，明天两地的天气情况如下：

### **上海**
- **天气**：中雨（白天），大雨（夜间）
- **温度**：日间最高 29°C，夜间最低 17°C
- **风力**：南风 1-3 级
- **出行建议**：明天上海的天气以降水为主，请携带雨具，注意防雨，并避免安排户外高强度活动，及时关注天气变化。

### **北京**
- **天气**：阴雨（白天），多云（夜间）
- **温度**：日间最高 14°C，夜间最低 8°C
- **风力**：东风 1-3 级
- **出行建议**：北京明天的天气较为凉爽，且有雨，但也仍有转晴的可能。建议穿着便于防风保暖的衣物，出门携带雨具。

**总结**：明天上海天气偏雨季且较热，北京则凉爽多雨。请根据您的行程调整出行安排。

You: 杭州呢？

[拦截] 大模型请求执行工具: ['get_weather_forecast']
工具参数: {"city": "杭州"}
是否允许执行？(Y/N): n
已拒绝执行该工具。
Agent: 

关于您的天气数据查询需求，我暂时无法获取相关数据。建议您通过其他方式获取最新的天气信息，例如查看当地气象网站、手机天气预报应用或向当地气象台咨询。祝您生活愉快！
```

# 🟣 第四阶段：部署上线与运维 LLMOps

> 目标：完成真正的“工程落地”，让系统能够面向真实用户，稳定且可监测。

## 1. 基础接口定义与 Pydantic 校验：将 RAG/Agent 逻辑接入 FastAPI

### 1.1 Pydantic
Pydantic 是 Python 的数据验证库。它利用 Python 的类型提示（Type Hints，比如 str, int, List）来强制校验数据。如果你规定一个字段必须是整数，但外部传入了字符串 "123"，Pydantic 会自动将其转换为整数 123；如果传入了 "abc"，它则会直接报错拦截。

### 1.2 FastAPI
FastAPI 是一个现代、快速的 Web 框架，专门用于构建 API（应用程序编程接口）。它基于异步编程（async/await）设计，并且天然集成了 Pydantic。

### 1.3 `api_server.py`

使用 Pydantic 定义严格的数据模型：`ChatRequest`（包含历史对话、用户提问、超参数）和 `ChatResponse`，确保 API 能拦截非法或恶意请求。

关键点：

- `ChatRequest` / `ChatResponse` 都使用 Pydantic 模型
- `model_config = ConfigDict(extra="forbid")`
- `question` / `history` / `max_tokens` 都有限制
- `response_model=ChatResponse`
- 路由挂载 `API Key` 校验和 `Content-Type` 校验
- 禁止客户端直接传 `system` 角色

```python
import json
import requests
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field, ConfigDict, field_validator
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. 基础接口定义与 Pydantic 校验
class Message(BaseModel):
    role: str = Field(..., description="角色，仅允许 user 或 assistant")
    content: str = Field(..., description="消息内容")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v == "system":
            raise ValueError("出于安全考虑，禁止客户端直接传递 'system' 角色。")
        if v not in ["user", "assistant"]:
            raise ValueError("角色必须是 'user' 或 'assistant'。")
        return v

class ChatRequest(BaseModel):
    # 严格模式：拒绝任何未在模型中定义的额外字段
    model_config = ConfigDict(extra="forbid")
    
    question: str = Field(..., min_length=1, max_length=1000, description="用户的当前提问")
    history: List[Message] = Field(default_factory=list, max_length=50, description="历史对话上下文")
    max_tokens: Optional[int] = Field(default=1024, le=4096, description="最大生成 token 数量限制")
    thread_id: str = Field(default="default_thread", description="用于 LangGraph 记忆的会话 ID")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    thread_id: str = Field(..., description="当前会话的 ID")
    usage: dict = Field(default_factory=dict, description="Token 消耗情况（预留）")

# 2. 天气工具与大模型配置
class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {"city": city, "key": self.api_key, "extensions": extensions, "output": "JSON"}
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "1":
            raise RuntimeError(f"Amap API error: {data.get('info')} ({data.get('infocode')})")
        return data

    def get_current_weather(self, city: str) -> dict:
        return self._request(city, "base")

    def get_weather_forecast(self, city: str) -> dict:
        return self._request(city, "all")

AMAP_KEY = ""

@tool
def get_current_weather(city: str) -> dict:
    """获取指定城市的当前实时天气情况。用于询问当前的温度、风力、湿度等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_current_weather(city)
    return raw_res["lives"][0] if "lives" in raw_res else raw_res

@tool
def get_weather_forecast(city: str) -> dict:
    """获取指定城市的多日天气预报。返回的数据包含今天整体、明天或未来的天气、最高/最低温等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_weather_forecast(city)
    if "forecasts" in raw_res:
        forecast = raw_res["forecasts"][0]
        return {
            "city": forecast["city"],
            "casts": [{k: v for k, v in c.items() if "float" not in k} for c in forecast["casts"]]
        }
    return raw_res

tools = [get_current_weather, get_weather_forecast]

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model="Qwen3.5-4B-AWQ-4bit"
)
llm_with_tools = llm.bind_tools(tools)

# 3. LangGraph 状态图构建 (移除终端中断机制)
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 4. FastAPI 路由与中间件拦截

app = FastAPI(title="LangGraph Agent API", version="1.0.0")

# 模拟的鉴权白名单
VALID_API_KEYS = {"my-secret-token-2026"}

def verify_headers(
    x_api_key: str = Header(..., description="API 鉴权 Key"),
    content_type: str = Header(..., description="必须为 application/json")
):
    """验证 Header：Content-Type 与 API Key"""
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_headers)])
async def chat_endpoint(request: ChatRequest):
    """
    接收用户提问，调用 LangGraph 智能体处理，并返回结果。
    """
    # 1. 组装 LangGraph 需要的 message 格式
    formatted_messages = []
    
    # 注入历史记录
    for msg in request.history:
        formatted_messages.append((msg.role, msg.content))
        
    # 注入当前问题
    formatted_messages.append(("user", request.question))

    # 2. 配置 Thread ID 以维持状态图记忆
    thread_config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # 3. 调用 Graph (同步调用，等待图完全执行完毕)
        # 注意：此处使用 invoke 而不是 stream，是为了直接获取最终状态
        final_state = graph.invoke(
            {"messages": formatted_messages}, 
            config=thread_config
        )
        
        # 4. 获取 Agent 的最后一条回复
        final_message = final_state["messages"][-1]
        
        return ChatResponse(
            answer=final_message.content,
            thread_id=request.thread_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行出错: {str(e)}")

if __name__ == "__main__":
    # 使用 Uvicorn 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

启动命令：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 5530
```

使用 Postman 测试：

```text
(LLM311) justin@DESKTOP-JENE0T2:~/workPro/LLM$ uvicorn api_server:app --host 0.0.0.0 --port 5530
INFO:     Started server process [3968]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5530 (Press CTRL+C to quit)
INFO:     127.0.0.1:49778 - "POST /chat HTTP/1.1" 200 OK
```
正常请求示意：

![正常请求示意](image.png)

未填写 `X-API-Key`：

![未填写 X-API-Key](image-1.png)

尝试注入系统提示词：

![尝试注入系统提示词](image-2.png)

尝试传入非法超参数：

![尝试传入非法超参数](image-3.png)

## 2. 实现流式输出（Streaming）

使用 FastAPI 的 `StreamingResponse` 和 `Server-Sent Events`（SSE）协议，改造代码，让大模型的输出像打字机一样，通过 API 一个 Token 一个 Token 地推送到客户端。

```python
import json
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. 基础接口定义与 Pydantic 校验
class Message(BaseModel):
    role: str = Field(..., description="角色，仅允许 user 或 assistant")
    content: str = Field(..., description="消息内容")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v == "system":
            raise ValueError("出于安全考虑，禁止客户端直接传递 'system' 角色。")
        if v not in ["user", "assistant"]:
            raise ValueError("角色必须是 'user' 或 'assistant'。")
        return v

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=1, max_length=1000)
    history: List[Message] = Field(default_factory=list, max_length=50)
    max_tokens: Optional[int] = Field(default=1024, le=4096)
    thread_id: str = Field(default="default_thread")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    thread_id: str = Field(..., description="当前会话的 ID")

# 2. 工具与大模型配置
class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {"city": city, "key": self.api_key, "extensions": extensions, "output": "JSON"}
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "1":
            raise RuntimeError(f"Amap API error: {data.get('info')} ({data.get('infocode')})")
        return data

    def get_current_weather(self, city: str) -> dict:
        return self._request(city, "base")

    def get_weather_forecast(self, city: str) -> dict:
        return self._request(city, "all")

AMAP_KEY = "" 

@tool
def get_current_weather(city: str) -> dict:
    """获取指定城市的当前实时天气情况。用于询问当前的温度、风力、湿度等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_current_weather(city)
    return raw_res["lives"][0] if "lives" in raw_res else raw_res

@tool
def get_weather_forecast(city: str) -> dict:
    """获取指定城市的多日天气预报。返回的数据包含今天整体、明天或未来的天气、最高/最低温等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_weather_forecast(city)
    if "forecasts" in raw_res:
        forecast = raw_res["forecasts"][0]
        return {
            "city": forecast["city"],
            "casts": [{k: v for k, v in c.items() if "float" not in k} for c in forecast["casts"]]
        }
    return raw_res

tools = [get_current_weather, get_weather_forecast]

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model="Qwen3.5-4B-AWQ-4bit",
    streaming=True # 开启底层模型的流式支持
)
llm_with_tools = llm.bind_tools(tools)

# 3. LangGraph 状态图构建
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 4. FastAPI 路由与 SSE 流式返回
app = FastAPI(title="LangGraph Agent API", version="1.0.0")

VALID_API_KEYS = {"my-secret-token-2026"}

def verify_headers(x_api_key: str = Header(...), content_type: str = Header(...)):
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_headers)])
async def chat_endpoint(request: ChatRequest):
    formatted_messages = [(msg.role, msg.content) for msg in request.history]
    formatted_messages.append(("user", request.question))
    thread_config = {"configurable": {"thread_id": request.thread_id}}
    final_state = await graph.ainvoke({"messages": formatted_messages}, config=thread_config)
    return ChatResponse(answer=final_state["messages"][-1].content, thread_id=request.thread_id)

# 流式输出 SSE 接口
@app.post("/chat/stream", dependencies=[Depends(verify_headers)])
async def chat_stream_endpoint(request: ChatRequest):
    # 组装消息列表
    formatted_messages = [(msg.role, msg.content) for msg in request.history]
    formatted_messages.append(("user", request.question))
    thread_config = {"configurable": {"thread_id": request.thread_id}}

    # 定义异步生成器函数
    async def event_generator():
        try:
            # astream 配合 stream_mode="messages" 能捕获到底层大模型的每一次消息增量 (Chunk)
            async for msg, metadata in graph.astream(
                {"messages": formatted_messages},
                config=thread_config,
                stream_mode="messages",
            ):
                # 条件 1: 必须是 agent 节点发出的消息 (排除了 tools 节点的消息)
                # 条件 2: 必须有 content 文本 (排除了 agent 请求调用工具时的内部不可见 json)
                if metadata.get("langgraph_node") == "agent" and msg.content:
                    
                    # 组装我们要发给前端的 JSON 数据块
                    chunk_data = json.dumps({"token": msg.content}, ensure_ascii=False)
                    
                    # 严格遵守 SSE 协议格式：以 "data: " 开头，以两个换行符 "\n\n" 结尾
                    yield f"data: {chunk_data}\n\n"

            # 模型生成完毕后，按照业界惯例发送 [DONE] 标识
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    # 使用 StreamingResponse 返回生成器，并指定媒体类型为 text/event-stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5530)
```

结果：

```text
(LLM311) justin@DESKTOP-JENE0T2:~/workPro/LLM$ curl -N -X POST http://127.0.0.1:5530/chat/stream \
-H "Content-Type: application/json" \
-H "X-API-Key: my-secret-token-2026" \
-d '{"question": "北京明天的天气怎么样？", "thread_id": "test_stream_01"}'
data: {"token": "\n\n"}

data: {"token": "\n\n"}

data: {"token": "根据"}

data: {"token": "天气预报"}

data: {"token": "，"}

data: {"token": "明天"}

data: {"token": "（"}

data: {"token": "5"}

data: {"token": "月"}

data: {"token": "5"}

data: {"token": "日"}

data: {"token": "）"}

data: {"token": "北京"}

data: {"token": "天气"}

data: {"token": "情况"}

data: {"token": "如下"}

data: {"token": "："}

data: {"token": "\n\n"}

data: {"token": "|"}

data: {"token": " "}

data: {"token": "时间"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "天气"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "白天"}

data: {"token": "温度"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "夜间"}

data: {"token": "温度"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "白天"}

data: {"token": "风力"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "夜间"}

data: {"token": "风力"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "风力"}

data: {"token": "等级"}

data: {"token": " |"}

data: {"token": "\n"}

data: {"token": "|"}

data: {"token": "------"}

data: {"token": "|"}

data: {"token": "------"}

data: {"token": "|"}

data: {"token": "----------"}

data: {"token": "|"}

data: {"token": "----------"}

data: {"token": "|"}

data: {"token": "----------"}

data: {"token": "|"}

data: {"token": "----------"}

data: {"token": "|"}

data: {"token": "----------"}

data: {"token": "|"}

data: {"token": "\n"}

data: {"token": "|"}

data: {"token": " "}

data: {"token": "白天"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "晴"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "2"}

data: {"token": "8"}

data: {"token": "℃"}

data: {"token": " |"}

data: {"token": " -"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "南"}

data: {"token": " |"}

data: {"token": " -"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "1"}

data: {"token": "-"}

data: {"token": "3"}

data: {"token": "级"}

data: {"token": " |"}

data: {"token": "\n"}

data: {"token": "|"}

data: {"token": " "}

data: {"token": "夜间"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "多云"}

data: {"token": " |"}

data: {"token": " -"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "1"}

data: {"token": "7"}

data: {"token": "℃"}

data: {"token": " |"}

data: {"token": " -"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "南"}

data: {"token": " |"}

data: {"token": " "}

data: {"token": "1"}

data: {"token": "-"}

data: {"token": "3"}

data: {"token": "级"}

data: {"token": " |"}

data: {"token": "\n\n"}

data: {"token": "明天的"}

data: {"token": "天气"}

data: {"token": "不错"}

data: {"token": "，"}

data: {"token": "以"}

data: {"token": "晴天"}

data: {"token": "为主"}

data: {"token": "，"}

data: {"token": "白天"}

data: {"token": "会比较"}

data: {"token": "温暖"}

data: {"token": "（"}

data: {"token": "最高"}

data: {"token": "2"}

data: {"token": "8"}

data: {"token": "℃"}

data: {"token": "），"}

data: {"token": "夜间"}

data: {"token": "稍"}

data: {"token": "凉"}

data: {"token": "（"}

data: {"token": "最低"}

data: {"token": "1"}

data: {"token": "7"}

data: {"token": "℃"}

data: {"token": "），"}

data: {"token": "风力"}

data: {"token": "较小"}

data: {"token": "（"}

data: {"token": "1"}

data: {"token": "-"}

data: {"token": "3"}

data: {"token": "级"}

data: {"token": "），"}

data: {"token": "是个"}

data: {"token": "适合"}

data: {"token": "出去"}

data: {"token": "游玩"}

data: {"token": "的美好"}

data: {"token": "天气"}

data: {"token": "。"}

data: [DONE]
```

## 3. 本地部署 Langfuse

### 3.1 Langfuse

Langfuse 是一个开源的 LLM 工程平台，专门解决大模型应用“难以调试、难以评估、难以追踪成本”的痛点。

在 Langfuse 中，有三个极其核心的概念：

- `Trace`（追踪）：对应用户的一次完整请求。比如用户问“上海天气如何？”到拿到最终结果，这整个生命周期就是一个 Trace。

- `Span`（跨度）：Trace 内部的子任务。比如“Agent 思考”“调用高德地图 API”“组装最终回答”，这些步骤各自是一个 Span。

- `Generation`（生成）：特指调用大模型生成文本的动作。这里会精确记录你发给大模型的 Prompt、大模型回复的内容、消耗了多少 Token，以及关键的性能指标（如 `TTFT` 和 `Token/s`）。

配置 Langfuse：

在一个新建的空目录（例如 `~/langfuse_local`）下，新建 `docker-compose.yml`：

```yaml
version: '3.5'
services:
  langfuse-server:
    image: langfuse/langfuse:latest
    depends_on:
      - db
      - clickhouse
      - minio-setup
      - redis
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
      - NEXTAUTH_SECRET=my_super_secret_key_for_langfuse_2026
      - SALT=my_super_secret_salt
      - NEXTAUTH_URL=http://localhost:3000
      - TELEMETRY_ENABLED=${TELEMETRY_ENABLED:-true}
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=${LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES:-false}
      - CLICKHOUSE_URL=http://clickhouse:8123
      - CLICKHOUSE_MIGRATION_URL=clickhouse://clickhouse:9000
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=langfuse123
      - CLICKHOUSE_CLUSTER_ENABLED=false
      - LANGFUSE_S3_EVENT_UPLOAD_BUCKET=langfuse-events
      - LANGFUSE_S3_EVENT_UPLOAD_REGION=us-east-1
      - LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT=http://minio:9000
      - LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID=minioadmin
      - LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=minioadmin
      - LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE=true
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_AUTH=langfuse_redis_123

  redis:
    image: redis:7
    command: redis-server --requirepass langfuse_redis_123
    ports:
      - "6379:6379"

  db:
    image: postgres:15
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=postgres
    ports:
      - "5432:5432"
    volumes:
      - database_data:/var/lib/postgresql/data

  clickhouse:
    image: clickhouse/clickhouse-server:23.11
    ports:
      - "8123:8123"
      - "9000:9000"
    environment:
      - CLICKHOUSE_DB=default
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=langfuse123
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
    volumes:
      - clickhouse_data:/var/lib/clickhouse

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9005:9000" 
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data

  minio-setup:
    image: minio/mc
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      sleep 5;
      /usr/bin/mc alias set myminio http://minio:9000 minioadmin minioadmin;
      /usr/bin/mc mb myminio/langfuse-events || true;
      /usr/bin/mc anonymous set public myminio/langfuse-events || true;
      exit 0;
      "

  langfuse-worker:
    image: langfuse/langfuse-worker:latest
    depends_on:
      - db
      - clickhouse
      - redis
      - minio-setup
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
      - SALT=my_super_secret_salt
      - TELEMETRY_ENABLED=${TELEMETRY_ENABLED:-true}
      - CLICKHOUSE_URL=http://clickhouse:8123
      - CLICKHOUSE_MIGRATION_URL=clickhouse://clickhouse:9000
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=langfuse123
      - CLICKHOUSE_CLUSTER_ENABLED=false
      - LANGFUSE_S3_EVENT_UPLOAD_BUCKET=langfuse-events
      - LANGFUSE_S3_EVENT_UPLOAD_REGION=us-east-1
      - LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT=http://minio:9000
      - LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID=minioadmin
      - LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=minioadmin
      - LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE=true
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_AUTH=langfuse_redis_123    
      
volumes:
  database_data:
    driver: local
  clickhouse_data:
    driver: local
  minio_data:
    driver: local
```

启动服务：

```bash
docker-compose up -d
```

浏览器打开 <http://localhost:3000>。

1. 注册一个本地账号。
2. 创建一个新项目，会生成一对公钥和私钥：

- `LANGFUSE_SECRET_KEY`（`sk-...`）
- `LANGFUSE_PUBLIC_KEY`（`pk-...`）

## 4. Trace 追踪

安装依赖：

```bash
uv pip install langchain langfuse
```

```python
import json
import requests
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-bd835943-a54c-4447-ba68-53ed25620001"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-ed77af4d-a40d-4773-ac8b-4bde1d0adf3a"
os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"

from langfuse.langchain import CallbackHandler

# 1. 基础接口定义与 Pydantic 校验
class Message(BaseModel):
    role: str = Field(..., description="角色，仅允许 user 或 assistant")
    content: str = Field(..., description="消息内容")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v == "system":
            raise ValueError("出于安全考虑，禁止客户端直接传递 'system' 角色。")
        if v not in ["user", "assistant"]:
            raise ValueError("角色必须是 'user' 或 'assistant'。")
        return v

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=1, max_length=1000)
    history: List[Message] = Field(default_factory=list, max_length=50)
    max_tokens: Optional[int] = Field(default=1024, le=4096)
    thread_id: str = Field(default="default_thread")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    thread_id: str = Field(..., description="当前会话的 ID")

# 2. 工具与大模型配置
class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {"city": city, "key": self.api_key, "extensions": extensions, "output": "JSON"}
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "1":
            raise RuntimeError(f"Amap API error: {data.get('info')} ({data.get('infocode')})")
        return data

    def get_current_weather(self, city: str) -> dict:
        return self._request(city, "base")

    def get_weather_forecast(self, city: str) -> dict:
        return self._request(city, "all")

AMAP_KEY = "" 

@tool
def get_current_weather(city: str) -> dict:
    """获取指定城市的当前实时天气情况。用于询问当前的温度、风力、湿度等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_current_weather(city)
    return raw_res["lives"][0] if "lives" in raw_res else raw_res

@tool
def get_weather_forecast(city: str) -> dict:
    """获取指定城市的多日天气预报。返回的数据包含今天整体、明天或未来的天气、最高/最低温等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_weather_forecast(city)
    if "forecasts" in raw_res:
        forecast = raw_res["forecasts"][0]
        return {
            "city": forecast["city"],
            "casts": [{k: v for k, v in c.items() if "float" not in k} for c in forecast["casts"]]
        }
    return raw_res

tools = [get_current_weather, get_weather_forecast]

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model="Qwen3.5-4B-AWQ-4bit",
    streaming=True # 开启底层模型的流式支持
)
llm_with_tools = llm.bind_tools(tools)

# 3. LangGraph 状态图构建
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

app = FastAPI(title="LangGraph Agent API with Langfuse", version="1.0.0")

VALID_API_KEYS = {"my-secret-token-2026"}

def verify_headers(x_api_key: str = Header(...), content_type: str = Header(...)):
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/chat/stream", dependencies=[Depends(verify_headers)])
async def chat_stream_endpoint(request: ChatRequest):
    formatted_messages = [(msg.role, msg.content) for msg in request.history]
    formatted_messages.append(("user", request.question))
    
    langfuse_handler = CallbackHandler()

    # 将追踪信息挂载到 LangGraph 的 metadata 配置中
    thread_config = {
        "configurable": {"thread_id": request.thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": request.thread_id,
            "langfuse_user_id": "justin_test_user",
            "langfuse_tags": ["streaming_api", "agent_v1"]
        }
    }

    async def event_generator():
        try:
            async for msg, metadata in graph.astream(
                {"messages": formatted_messages},
                config=thread_config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "agent" and msg.content:
                    chunk_data = json.dumps({"token": msg.content}, ensure_ascii=False)
                    yield f"data: {chunk_data}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5530)
```

模拟流量脚本：

```python
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
                    if "data: " in decoded_line and "[DONE]" not in decoded_line:
                        data = json.loads(decoded_line.replace("data: ", ""))
                        print(data.get("token", ""), end="", flush=True)
            
            print(f"\n=> 请求完成，耗时: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"\n=> 请求失败: {e}")

if __name__ == "__main__":
    for i in range(1, 21):
        make_request(i)
        time.sleep(1)
    print("\n✅ 20次请求模拟完成，请登录 Langfuse 控制台查看！")
```

## 5. 仪表盘数据分析

![Langfuse 仪表盘总览](image-4.png)

![Langfuse 延迟分析](image-5.png)

分析：

- Trace 数量完全对齐：第一张图显示总共有 22 条 Traces。这与之前运行的脚本（模拟了 20 次请求）加上手动测试的 2 次完全吻合。
- LangGraph 节点级耗时拆解较清晰：在第二张图的 `Generation latency percentiles` 中，Langfuse 成功把整个图（Graph）拆开了。
- `tools` 节点（高德 API）非常快：`p50`（中位数）耗时只有 `0.10s`。这说明网络请求和工具调用基本没有瓶颈。
- `agent / ChatOpenAI` 节点（大模型推理）是主要耗时来源：`p50` 耗时大约 `5.31s`，`p90`（前 90% 的请求）需要 `11.98s`。
- 模型识别正确：底部图表成功识别出了使用的模型名称 `Qwen3.5-4B-AWQ-4bit`。
