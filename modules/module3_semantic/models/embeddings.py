# modules/module_c_semantic/models/embeddings.py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class FeatureExtractor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化多语言文本编码器，支持中英双语指令的降维
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] 加载深度特征提取模型 [{model_name}] on {self.device}...")
        
        self.text_encoder = SentenceTransformer(model_name, device=self.device)
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension() # 通常为 384 维
        self.context_dim = 10 # 物理车况特征维度

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """批量提取文本的高维语义特征"""
        return self.text_encoder.encode(texts, convert_to_numpy=True)

    def encode_context(self, context: dict, param: float = None) -> np.ndarray:
        """
        将物理车况（速度、档位、天气等）编码为连续的浮点特征张量
        """
        # 1. 速度归一化 (0-200 km/h -> 0.0-1.0)
        speed_norm = min(context['speed'] / 200.0, 1.0)
        
        # 2. 档位 One-Hot 编码
        gear_map = {'P': [1,0,0,0], 'R': [0,1,0,0], 'N': [0,0,1,0], 'D': [0,0,0,1]}
        gear_vec = gear_map.get(context['gear'], [0,0,0,0])
        
        # 3. 天气 One-Hot 编码
        weather_map = {'sunny': [1,0,0,0], 'rainy': [0,1,0,0], 'snowy': [0,0,1,0], 'hail': [0,0,0,1]}
        weather_vec = weather_map.get(context['weather'], [0,0,0,0])
        
        # 4. 控制参数归一化 (如音量百分比，如果没有具体参数则默认为0)
        param_norm = (param / 100.0) if param is not None else 0.0
        
        return np.array([speed_norm] + gear_vec + weather_vec + [param_norm], dtype=np.float32)