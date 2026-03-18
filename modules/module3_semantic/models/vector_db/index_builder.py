# c:\Users\Leo\Desktop\gemini\models\vector_db\index_builder.py
import sys
import json
import numpy as np
import faiss
from pathlib import Path

# 路径挂载
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from modules.module3_semantic.models.embeddings import FeatureExtractor

def build_index():
    # 👈 修改点：指向我们刚刚生成的 10000 条数据集
    DATA_PATH = PROJECT_ROOT / "semantic_safety_train.jsonl"
    INDEX_SAVE_PATH = PROJECT_ROOT / "vguard_hnsw.index"
    META_SAVE_PATH = PROJECT_ROOT / "vguard_metadata.json"

    if not DATA_PATH.exists():
        print(f"[!] 错误：找不到数据集 {DATA_PATH}，请确认 synthesizer.py 运行成功。")
        return

    extractor = FeatureExtractor()
    
    print(f"[*] 开始解析大规模数据集: {DATA_PATH.name}")
    
    embeddings_list = []
    metadata_dict = {}

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line)
            
            # 提取文本向量
            text_emb = extractor.encode_text([item['text']])[0]
            # 提取车况向量
            ctx_vec = extractor.encode_context(item['context'])
            
            # 执行多模态融合 (1.0 vs 5.0 权重)
            # 这里的融合方式必须和 reasoning.py 以及 train.py 严格一致！
            fusion_vec = np.hstack((text_emb * 1.0, ctx_vec * 5.0)).astype('float32')
            
            embeddings_list.append(fusion_vec)
            metadata_dict[i] = {
                "id": item['id'],
                "text": item['text'],
                "ground_truth_score": item['ground_truth_score'],
                "reason": item['reason']
            }
            
            if (i + 1) % 2000 == 0:
                print(f"[>] 已处理 {i+1} 条向量...")

    # 构建 FAISS HNSW 索引
    dimension = embeddings_list[0].shape[0]
    # HNSW32 代表每个点连接 32 个邻居，检索速度极快
    index = faiss.IndexHNSWFlat(dimension, 32)
    
    # 转换为 numpy 矩阵并归一化
    data_matrix = np.array(embeddings_list).astype('float32')
    faiss.normalize_L2(data_matrix)
    
    print("[*] 正在训练并构建 HNSW 图索引...")
    index.add(data_matrix)
    
    # 保存结果
    faiss.write_index(index, str(INDEX_SAVE_PATH))
    with open(META_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=4)
        
    print(f"【成功】建库完成！当前库内已存入 {index.ntotal} 条高维安全基准。")

if __name__ == "__main__":
    build_index()