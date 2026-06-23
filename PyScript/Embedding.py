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