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
retriever = index.as_retriever(similarity_top_k=3)#显卡显存不够，只能跑1024的上下文，参考太多会超

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