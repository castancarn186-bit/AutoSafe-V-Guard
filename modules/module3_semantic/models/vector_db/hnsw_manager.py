# models/vector_db/hnsw_manager.py
import faiss
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any  # 👈 新增：为了兼容 Python 3.10 以下的版本

class HNSWManager:
    """
    负责管理 FAISS HNSW 向量索引的加载与实时检索
    """
    def __init__(self):
        # 向上找三层，精准定位到 gemini 根目录
        self.db_dir = Path(__file__).resolve().parent.parent.parent
        self.index_path = self.db_dir / 'vguard_hnsw.index'
        self.meta_path = self.db_dir / 'vguard_metadata.json'
        
        self.index = None
        self.metadata_list = []
        
        print("[*] 正在加载 HNSW 向量数据库与元数据...")
        try:
            # 加载 FAISS 索引
            self.index = faiss.read_index(str(self.index_path))
            
            # 加载元数据。因为 FAISS 默认存的是 0, 1, 2... 的顺序整数 ID
            # 我们将字典的 values 转为列表，利用索引下标进行 O(1) 匹配
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                self.metadata_list = list(metadata_dict.values())
                
            print(f"[+] 向量库加载成功！当前库内包含 {self.index.ntotal} 个安全基准场景。")
        except FileNotFoundError:
            print("[!] 警告：未找到向量库文件，请先运行 vector_db/index_builder.py 建库。")
        except Exception as e:
            print(f"[!] 向量库加载失败：{e}")

    # 👈 修改点：将 dict | None 改为 Optional[Dict[str, Any]]，tuple 改为 Tuple
    def search(self, query_vector: np.ndarray, top_k: int = 1, threshold: float = 0.35) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        在 HNSW 图中检索最近邻特征点。
        
        :param query_vector: 多模态特征融合后的向量 (shape: [1, 394])
        :param top_k: 返回几个最相似的结果
        :param threshold: 距离阈值（越小代表越相似，大于此阈值认为未见过此场景）
        :return: (匹配到的元数据字典, 距离) 或 (None, 当前最小距离)
        """
        if self.index is None or len(self.metadata_list) == 0:
            return None, 999.0
            
        # 必须进行 L2 归一化，才能使 L2 距离等价于余弦相似度搜索
        faiss.normalize_L2(query_vector)
        
        # 检索最近的 top_k 个节点
        distances, indices = self.index.search(query_vector, top_k)
        
        best_dist = float(distances[0][0])
        best_idx = int(indices[0][0])
        
        # 判断是否在阈值内（即是否为“高置信度”的历史已知场景）
        if best_dist < threshold and best_idx < len(self.metadata_list):
            match_meta = self.metadata_list[best_idx]
            return match_meta, best_dist
            
        return None, best_dist