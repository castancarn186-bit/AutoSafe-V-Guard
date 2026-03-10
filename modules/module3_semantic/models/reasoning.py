import sys
import torch
import numpy as np
from pathlib import Path
import time

# --- 路径环境配置 (针对当前层级修复) ---
CURRENT_DIR = Path(__file__).resolve().parent      # 这是 models 文件夹
PROJECT_ROOT = CURRENT_DIR.parent                  # 退回上一级，定位到 gemini 根目录

# 把项目根目录强行插入到系统寻址路径的最前面
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 动态路径设置：兼容 core 和 modules
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(CURRENT_DIR))

# 导入核心数据协议 (假设你在 core/protocol.py 中定义了这些)
from modules.module3_semantic.core.protocol import SemanticInput, RiskReport, RiskLevel, IntentCategory

# 导入子模块
from modules.module3_semantic.models.embeddings import FeatureExtractor
from modules.module3_semantic.models.risk_net import RiskNet
from vector_db.hnsw_manager import HNSWManager

class SemanticSafetyEngine:
    """
    语义冲突层核心引擎：RAG + Deep Learning 双轨制防线
    """
    def __init__(self):
        print("=== 初始化 V-Guard 语义安全决策引擎 ===")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 启动特征提取器 (翻译官)
        self.extractor = FeatureExtractor()
        
        # 2. 启动快思考系统 (向量数据库)
        self.vector_db = HNSWManager()
        
        # 3. 启动慢思考系统 (逻辑推理脑)
        self.risk_net = RiskNet().to(self.device)
        # 修复点 1：直接在 CURRENT_DIR (即 models 目录) 下找权重
        model_path = CURRENT_DIR / 'vguard_risknet_best.pth'
        
        try:
            self.risk_net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print("[+] 深度学习模型 (RiskNet) 权重加载完毕！")
        except FileNotFoundError:
            print(f"[!] 未找到 RiskNet 权重，请确认 {model_path} 是否存在。")
        finally:
            # 修复点 2：无论是否找到权重，都必须强行设置为推理模式，防止 BatchNorm 报错
            self.risk_net.eval()

        # 融合向量的权重 (必须与建库时保持一致)
        self.text_weight = 1.0
        self.context_weight = 5.0

    def evaluate(self, user_input: SemanticInput) -> RiskReport:
        # 1. 基础特征提取
        text_emb = self.extractor.encode_text([user_input.text])[0]
        ctx_vec = self.extractor.encode_context(user_input.context.model_dump())
        
        # 2. 向量库辅助阶段：不再直接 return，而是提取“经验特征”
        weighted_text = text_emb * self.text_weight
        weighted_ctx = ctx_vec * self.context_weight
        fusion_query = np.hstack((weighted_text, weighted_ctx)).astype(np.float32)
        fusion_query = np.expand_dims(fusion_query, axis=0)
        
        match_meta, _ = self.vector_db.search(fusion_query, threshold=0.2)
        
        # 如果没搜到，给一个中性的 0.5 分作为“无意见”
        vdb_score_val = match_meta['ground_truth_score'] if match_meta else 0.5
        
        # 3. 深度学习决策阶段：将 VDB 分数作为特征之一
        t_text = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_ctx = torch.tensor(ctx_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_vdb = torch.tensor([[vdb_score_val]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # 神经网络现在拥有：[语义理解] + [当前车况] + [相似历史参考]
            final_score = self.risk_net(t_text, t_ctx, t_vdb).item()

        # 4. 生成报告
        return self._build_report(final_score, vdb_score_val, match_meta)

    def _build_report(self, score: float, vdb_ref: float, match_meta: dict) -> RiskReport:
        level = RiskLevel.DANGER if score >= 0.7 else RiskLevel.WARNING if score >= 0.3 else RiskLevel.SAFE
        
        # 理由中可以体现这种融合逻辑
        reason = f"AI大脑综合研判：风险分{score:.2f}。"
        if match_meta:
            reason += f"（参考了历史相似案例：{match_meta['text']}）"
            
        return RiskReport(
            risk_score=round(score, 3),
            level=level,
            reason=reason,
            intent_category=IntentCategory.BODY_CONTROL
        )
        
    def _generate_reason_by_score(self, score: float) -> str:
        """为深度模型泛化出的分数生成可解释理由"""
        if score >= 0.8:
            return "深度模型评估为致命风险，环境特征与指令存在严重冲突。"
        elif score >= 0.6:
            return "深度模型评估为高风险，建议拦截操作并语音提示驾驶员。"
        elif score >= 0.3:
            return "深度模型评估为潜在警告，建议降级执行（如微调开度）。"
        else:
            return "深度模型评估为安全操作，允许放行。"

# ==========================================
# 4. 本地模拟测试 (Mock Test)
# ==========================================
if __name__ == "__main__":
    from core.protocol import VehicleContext, Language, WeatherCondition
    import time
    
    # 初始化总引擎
    engine = SemanticSafetyEngine()
    
    print("\n--- 模拟 ASR 数据接入 ---")
    # 模拟场景：车速 120km/h，下雨天，用户说“帮我把窗户开到底透透风”
    test_context = VehicleContext(
        speed=120.0, 
        speed_limit=120.0, 
        gear="D", 
        weather=WeatherCondition.RAINY,
        traffic_density="low",
        has_pedestrians=False,
        window_open=False
    )
    
    test_input = SemanticInput(
        text="帮我把窗户开到底透透风",
        language=Language.ZH,
        context=test_context
    )
    
    # 执行评估并计时
    start_time = time.time()
    report = engine.evaluate(test_input)
    cost_time = (time.time() - start_time) * 1000
    
    print(f"\n评估耗时: {cost_time:.2f} ms")
    print(f"安全打分: {report.risk_score}")
    print(f"风险评级: {report.level.value}")
    print(f"拦截原因: {report.reason}")