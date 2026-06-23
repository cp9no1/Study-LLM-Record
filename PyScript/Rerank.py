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