from llama_cpp import Llama

model_path = r"E:\workPro\LLM\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf" 

print("正在加载 GGUF 模型...")
# 初始化模型
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, # -1 表示把所有层卸载到 GPU 上加速，如果显存不够就写具体的数字比如 20
    n_ctx=2048,      # 上下文长度
    verbose=False
)

print("模型加载完毕！输入 'quit' 退出。")
while True:
    user_input = input("\n你: ")
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