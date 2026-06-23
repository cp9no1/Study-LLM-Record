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