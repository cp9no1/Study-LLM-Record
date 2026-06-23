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