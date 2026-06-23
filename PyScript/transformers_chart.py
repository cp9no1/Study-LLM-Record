from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import torch

model_path = r"E:\workPro\LLM\Qwen3.5-4B" 

print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    local_files_only=True, 
    trust_remote_code=True
)

# 配置 4-bit 量化参数
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
    user_input = input("\n你: ")
    
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
            max_new_tokens=1024,                  # 放宽生成的最大长度
            streamer=streamer,                    # 开启流式打字机输出
            pad_token_id=tokenizer.eos_token_id,  # 消除 pad_token 的警告
            do_sample=True,                       # 开启采样，让回答更自然
            temperature=0.7                       # 控制回答的随机性
        )

    # 6. 从生成的 ID 中提取刚才新生成的回答内容，存入历史记录，供下一轮对话使用
    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    history.append({"role": "assistant", "content": response_text})