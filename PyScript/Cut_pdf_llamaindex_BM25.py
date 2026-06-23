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