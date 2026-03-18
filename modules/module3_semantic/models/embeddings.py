# modules/module_c_semantic/models/embeddings.py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class FeatureExtractor:
   def __init__(self, model_path=None):
        """
      初始化多语言文本编码器，支持中英双语指令的降维
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 默认使用 modules/semantic_model 目录（绝对路径）
        if model_path is None:
           model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'semantic_model'))
        
        print(f"[*] 加载深度特征提取模型 [{model_path}] on {self.device}...")
        
       # 检查路径是否存在
        if not os.path.exists(model_path):
          raise FileNotFoundError(f"找不到模型目录：{model_path}")
      
        self.text_encoder = SentenceTransformer(model_path, device=self.device)
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension() # 通常为 384 维
        self.context_dim = 10

   def encode_text(self, texts: list[str]) -> np.ndarray:
        """批量提取文本的高维语义特征"""
        return self.text_encoder.encode(texts, convert_to_numpy=True)

   def encode_context(self, context: dict, param: float = None) -> np.ndarray:
        """
        将物理车况（速度、档位、天气等）编码为连续的浮点特征张量
        """
        # 1. 速度归一化 (0-200 km/h -> 0.0-1.0)
        speed_norm = min(context['speed'] / 200.0, 1.0)

        speed = context['speed']
        bins = [0, 30, 60, 90, 120, float('inf')]
        speed_bin = np.digitize(speed, bins) - 1  # 返回 0,1,2,3,4
        speed_onehot = np.zeros(5, dtype=np.float32)
        speed_onehot[speed_bin] = 1.0

        # 2. 档位 One-Hot 编码
        gear_map = {'P': [1,0,0,0], 'R': [0,1,0,0], 'N': [0,0,1,0], 'D': [0,0,0,1]}
        gear_vec = gear_map.get(context['gear'], [0,0,0,0])
        
        # 3. 天气 One-Hot 编码
        weather_map = {'sunny': [1,0,0,0], 'rainy': [0,1,0,0], 'snowy': [0,0,1,0], 'hail': [0,0,0,1]}
        weather_vec = weather_map.get(context['weather'], [0,0,0,0])
        
        # 4. 控制参数归一化 (如音量百分比，如果没有具体参数则默认为0)
        param_norm = (param / 100.0) if param is not None else 0.0
        is_bad_weather = 1 if context['weather'] in ['rainy', 'snowy', 'hail'] else 0
        speed_weather_interact = speed_norm * is_bad_weather  # 交互特征
        # 打印各部分长度
        print(f"speed_norm 类型: {type(speed_norm)}")
        print(f"speed_onehot 长度: {len(speed_onehot.tolist())}")  # 应为 5
        print(f"gear_vec 长度: {len(gear_vec)}")  # 应为 4
        print(f"weather_vec 长度: {len(weather_vec)}")  # 应为 4
        print(f"param_norm: {param_norm}")

        feature_list = [speed_norm] + speed_onehot.tolist() + gear_vec + weather_vec + [param_norm,
                                                                   speed_weather_interact]
        print(f"合并后列表长度: {len(feature_list)}")  # 应为 15
        return np.array(feature_list, dtype=np.float32)


             # 返回数组

