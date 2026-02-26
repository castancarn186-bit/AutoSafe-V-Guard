import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg') # 告诉程序在后台绘图，不要弹出那个白色的预览窗口
class SemanticLibrary:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # 加载预训练语义嵌入模型
        print(f"正在加载语义模型：{model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.command_map = {} # 存储 {标准指令：[同义词列表]}
        self.embeddings = None
        self.raw_data = None

    def load_dataset(self, file_path):
        """加载初始车辆指令数据集"""
        if not os.path.exists(file_path):
            # 如果没有文件，生成一些模拟数据
            print("未找到数据文件，生成模拟车辆指令数据集...")
            data = {
                'command': [
                    '打开空调', '把空调开了', '空调太热了', '关闭车窗', '把窗户关上', 
                    '车窗留条缝', '导航去公司', '我要回家', '设置目的地为公司',
                    '加速', '开快点', '减速', '慢一点', '打开音乐', '放首歌'
                ]
            }
            self.raw_data = pd.DataFrame(data)
            self.raw_data.to_csv(file_path, index=False)
        else:
            self.raw_data = pd.read_csv(file_path)
        print(f"数据集加载完成，共 {len(self.raw_data)} 条指令。")

    def process_embeddings(self):
        """将文本指令转换为向量"""
        texts = self.raw_data['command'].tolist()
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        return self.embeddings

    def cluster_commands(self, n_clusters=5):
        """基于向量进行聚类，识别同义词组"""
        if self.embeddings is None:
            self.process_embeddings()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.embeddings)
        self.raw_data['cluster_id'] = labels
        
        # 构建映射关系
        for cluster_id in range(n_clusters):
            cluster_cmds = self.raw_data[self.raw_data['cluster_id'] == cluster_id]['command'].tolist()
            # 简单策略：取第一条作为标准指令
            standard_cmd = cluster_cmds[0]
            self.command_map[standard_cmd] = cluster_cmds
            
        return self.command_map

    def visualize_clusters(self, save_path='ui/assets/m3_clusters.png'):
        """使用 t-SNE 降维可视化指令分布"""
        if self.embeddings is None:
            raise ValueError("请先运行 process_embeddings")
        
        # 【中文字体设置】
        import matplotlib.font_manager as fm
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 自动计算合适的 perplexity 值
        n_samples = len(self.embeddings)
        perplexity_val = max(1, min(5, n_samples - 1))
        
        print(f"[调试信息] 当前样本数：{n_samples}, 使用 perplexity: {perplexity_val}")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
        reduced_embeds = tsne.fit_transform(self.embeddings)
        
        plt.figure(figsize=(12, 9))
        scatter = plt.scatter(reduced_embeds[:, 0], reduced_embeds[:, 1], 
                              c=self.raw_data['cluster_id'], cmap='viridis', alpha=0.7, s=100)
        
        # 为每个点添加中文指令标注
        for i, txt in enumerate(self.raw_data['command']):
            plt.annotate(txt, (reduced_embeds[i, 0], reduced_embeds[i, 1]), 
                        fontsize=9, alpha=0.8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
            
        plt.title("车辆指令语义聚类图 (t-SNE)", fontsize=16, fontweight='bold')
        plt.xlabel("降维维度 1", fontsize=12)
        plt.ylabel("降维维度 2", fontsize=12)
        
        # 设置颜色条中文标签
        cbar = plt.colorbar(scatter)
        cbar.set_label('聚类类别编号', fontsize=11)
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 高清保存
        print(f"✅ 语义聚类图已保存至：{save_path}")
        plt.close()

    def match_intent(self, input_text):
        """输入新指令，匹配到最接近的标准指令"""
        input_emb = self.model.encode([input_text], convert_to_numpy=True)
        # 计算与所有原始指令的余弦相似度
        similarities = np.dot(input_emb, self.embeddings.T) / (
            np.linalg.norm(input_emb) * np.linalg.norm(self.embeddings, axis=1)
        )
        best_idx = np.argmax(similarities)
        best_cmd = self.raw_data.iloc[best_idx]['command']
        cluster_id = self.raw_data.iloc[best_idx]['cluster_id']
        # 返回该类的标准指令
        standard_cmd = list(self.command_map.keys())[list(self.command_map.values()).index(
             next(v for k, v in self.command_map.items() if best_cmd in v)
        )]
        return standard_cmd, cluster_id, similarities[0][best_idx]
