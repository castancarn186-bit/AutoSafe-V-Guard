#风险融合决策引擎	收集 A/B/C 的报告，执行加权算法，输出最终决策。
# core/engine.py
import os
import sys
import logging
import matplotlib

# 强制使用 Agg 后端，防止在后台运行时弹出窗口导致程序卡死
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.state import shared_state, ModuleReport

# 动态导入模块 3 的组件
# 假设你的项目结构中，core 和 modules 都在根目录下
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3_PATH = os.path.join(ROOT_DIR, "modules", "module3_semantic")
sys.path.append(os.path.join(M3_PATH, "src"))

try:
    from semantic_lib import SemanticLibrary
    from state_model import DrivingStateModel
    from risk_assessment import RiskManager
except ImportError as e:
    print(f"核心组件加载失败: {e}")


class VGuardEngine:
    """V-Guard 风险融合决策引擎 (集成语义防御模块)"""

    def __init__(self):
        print("🚀 [V-Guard Engine] 正在初始化离线安全决策内核...")

        # 1. 路径配置
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(ROOT_DIR, "modules", "semantic_model")
        self.data_path = os.path.join(M3_PATH, "data", "raw_commands.csv")
        # 获取当前文件所在目录的上一级（项目根目录）
        # 强制把输出目录设为 ui/assets
        self.output_dir = os.path.join(ROOT_DIR, "ui", "assets")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 2. 初始化语义组件
        # 强制离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.sem_lib = SemanticLibrary(model_name=self.model_path)
        self.sem_lib.load_dataset(self.data_path)
        self.sem_lib.process_embeddings()
        self.sem_lib.cluster_commands(n_clusters=5)

        # 交付物 1：初始化时生成一次语义聚类图（静态）
        self.generate_cluster_map()

        self.state_model = DrivingStateModel()
        self.risk_mgr = RiskManager(input_dim=16)

    def generate_cluster_map(self):
        """生成交付物 1：语义聚类图"""
        self.sem_lib.visualize_clusters()
        # 将生成的图保存到 UI 目录
        save_path = os.path.join(self.output_dir, "m3_clusters.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📊 [Engine] 已保存语义聚类图: {save_path}")

    async def run_pipeline(self, text):
        """
        核心防御流水线：ASR 文本 -> 语义匹配 -> 状态融合 -> 风险判定
        """
        if not text:
            return

        print(f"🛡️ [V-Guard] 正在对指令 '{text}' 进行多源风险评估...")

        # 1. 语义特征提取
        standard_cmd, cluster_id, confidence = self.sem_lib.match_intent(text)
        cmd_embedding = self.sem_lib.model.encode([standard_cmd])[0]

        # 2. 实时驾驶状态感知 (对接 shared_state 里的真实数据)
        current_speed = getattr(shared_state, 'vehicle_speed', 0.0)
        mock_sensor_data = {
            'speed': current_speed,
            'speed_limit': 80.0,
            'auto_mode': False,
            'surroundings': {'front_dist': 30.0, 'pedestrian_near': False,'lane_keeping': True}
        }
        self.state_model.update_from_sensors(mock_sensor_data)
        state_vector = self.state_model.get_state_vector()

        # 交付物 2：生成驾驶状态可视化图（动态刷新）
        self.state_model.visualize_state()

        # 3. 风险评分与决策
        input_tensor = self.risk_mgr.prepare_input(cmd_embedding, state_vector)
        risk_score = self.risk_mgr.evaluate(input_tensor)
        decision, _ = self.risk_mgr.generate_risk_matrix(risk_score)

        # 交付物 3：生成决策矩阵图（动态刷新）
        decision, _ = self.risk_mgr.generate_risk_matrix(risk_score)

        # 4. 更新系统状态，供 UI 显示
        report = ModuleReport(
            module_id="C",
            risk_score=float(risk_score),
            status="DENY" if decision == "DENY" else "PASS",
            reason=f"匹配意图: {standard_cmd}"
        )
        shared_state.update_module_report(report)

        return decision
'''
#仅接入第二部分
import os
import time
import json
import wave
import pyaudio
import threading
from datetime import datetime
from typing import Dict, Any
import os
import asyncio
import wave
import pyaudio
from datetime import datetime
from core.state import shared_state, ModuleReport

# 修复 DLL 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VGuardEngine:
    """V-Guard 核心安全防御引擎"""

    def __init__(self):
        print("🚀 [V-Guard Engine] 正在启动安全决策内核...")
        # 注意：这里不初始化 ASR 模型，因为我们在 app.py 的 session 中统一管理
        # 这样可以避免多次加载模型导致内存爆炸
        pass

    async def run_pipeline(self, audio_path=None):
        """
        核心防御流水线
        该方法由 app.py 调用。如果是模拟状态，它保持运行；如果是真实音频，它执行 ASR 风险判定。
        """
        if not audio_path:
            # 模拟模式下的基础逻辑（同步状态）
            await asyncio.sleep(0.1)
            return

        # 如果有音频路径，说明触发了真实防御检测
        # 注意：ASR 的实际推理逻辑已在 app.py 的 start_voice_defense 中由 to_thread 处理
        # 这里预留给后续成员 A (声学) 和成员 C (语义) 的集成接口
        pass

    def get_system_status(self):
        return "RUNNING" if shared_state.is_running else "STOPPED"


# 确保在文件末尾没有多余的、会导致报错的旧代码
'''
'''
import asyncio
import time
import random
from typing import List, Tuple
from core.protocol import RiskReport, DecisionType, AttackType
from core.state import shared_state
from hardware.gpio_ctrl import vguard_hw
from data.database_manager import db_manager
from modules.module2_ASR.asr_risk_model import ASRRiskModel
from .utils import VoiceCapture # 假设你把上面的录音类放到了 utils


class VGuardEngine:
    def __init__(self):
        # 初始化 B 成员的风险评估模型
        self.asr_analyzer = ASRRiskModel(model_size="base")
        self.recorder = VoiceCapture()

    def process_voice_command(self):
        """
        核心防御流水线：录音 -> ASR风险评估 -> 综合决策
        """
        # 1. 采集实时语音
        audio_file = self.recorder.record(seconds=3)

        # 2. 调用 Module 2 进行安全性评估
        # 这是 B 成员的核心贡献点，我们要拿到原始的 metrics
        asr_report = self.asr_analyzer.compute_risk(audio_file)

        # 3. 语义规范化处理（解决你遇到的繁简问题）
        # 在答辩时，这叫“感知一致性对齐”
        recognized_text = asr_report.get("text", "").replace("請", "请").replace("開", "开")

        # 4. 封装为系统级防御报告
        defense_decision = {
            "module_b": {
                "risk_score": asr_report["risk_score"],
                "confidence": asr_report["confidence_metrics"]["confidence_score"],
                "stability": asr_report["stability_metrics"]["stability_score"],
                "text": recognized_text
            },
            # 这里的 final_decision 逻辑体现了你的“网关”作用
            "final_status": "BLOCK" if asr_report["risk_score"] > 0.7 else "PASS"
        }

        return defense_decision

    async def _get_module_report(self, module_id: str) -> RiskReport:
        """
        模拟模块检测逻辑，集成了攻击注入演示后门
        """
        start_t = time.perf_counter()

        # 模拟不同模块的基础计算开销
        delay = {"A": 0.01, "B": 0.02, "C": 0.01}
        await asyncio.sleep(delay.get(module_id, 0.01) + random.uniform(0, 0.01))

        risk = random.uniform(0.05, 0.15)
        reason = "监测正常"

        # --- 攻击模拟后门：用于演示现场一键触发 ---
        active_attack = shared_state.active_attack

        if module_id == "A" and active_attack == AttackType.REPLAY.value:
            risk = 0.95
            reason = "警告：检测到高频重放攻击特征"
        elif module_id == "B" and active_attack == AttackType.ADVERSARIAL.value:
            risk = 0.85
            reason = "异常：ASR 置信度发生非线性坍塌"
        elif module_id == "C":
            # 逻辑校验：高速行驶拦截危险指令
            if shared_state.vehicle_speed > 80 and (
                    "打开" in shared_state.asr_text or "后备箱" in shared_state.asr_text):
                risk = 0.98
                reason = f"危险：当前车速 {shared_state.vehicle_speed}km/h，已拦截车门操作"
            elif shared_state.weather == "暴雨" and "关闭" in shared_state.asr_text and "雨刮" in shared_state.asr_text:
                risk = 0.90
                reason = "警告：极端天气下禁止关闭安全辅助设备"

        latency = (time.perf_counter() - start_t) * 1000

        return RiskReport(
            module_id=module_id,
            risk_score=risk,
            suggestion=DecisionType.BLOCK.value if risk > 0.8 else DecisionType.PASS.value,
            reason=reason,
            evidence={"latency": latency},
            latency_ms=round(latency, 2)
        )

    def _fusion_logic(self, reports: List[RiskReport]) -> Tuple[float, str]:
        """
        多源风险融合算法 [cite: 14]
        """
        total_score = 0.0

        # 策略：一票否决 + 加权平均 [cite: 9]
        for r in reports:
            if r.risk_score > 0.9:
                return r.risk_score, DecisionType.BLOCK.value

            w = self.weights.get(r.module_id, 0.3)
            total_score += r.risk_score * w

        # 判定决策 [cite: 15]
        if total_score > 0.7:
            decision = DecisionType.BLOCK.value
        elif total_score > 0.4:
            decision = DecisionType.WARN.value
        else:
            decision = DecisionType.PASS.value

        return round(total_score, 2), decision

'''
'''
# core/engine.py
import random
import time
from typing import List, Tuple
from core.protocol import RiskReport, DecisionType, SystemContext
from core.state import shared_state  # 引入共享状态，实现 UI 与引擎联动
from hardware.gpio_ctrl import vguard_hw  # 引入硬件控制（模拟器/真实树莓派）
from data.database_manager import db_manager  # 引入影子模式数据库


class VGuardEngine:
    def __init__(self):
        # 权重分配：A(声学) 30%, B(ASR) 30%, C(语义) 40%
        self.weights = {"A": 0.4, "B": 0.3, "C": 0.3}

    def generate_mock_reports(self) -> List[RiskReport]:
        """
        核心演示逻辑：基于 shared_state (车速、天气) 动态生成检测报告
        """
        speed = shared_state.vehicle_speed
        text = shared_state.asr_text
        weather = shared_state.weather

        # --- 模拟 第三部分：语义冲突检测 (C 模块) ---
        c_risk = 0.0
        c_reason = "逻辑合规"

        # 逻辑：高速拦截
        if speed > 80 and ("后备箱" in text or "门" in text):
            c_risk = 0.95
            c_reason = f"危险：高速({speed}km/h)禁止开启车门类设备"
        # 逻辑：恶劣天气拦截
        elif weather == "暴雨" and "关闭" in text and "雨刮" in text:
            c_risk = 0.85
            c_reason = "危险：极端天气下禁止关闭安全设备"
        else:
            c_risk = random.uniform(0.0, 0.1)

        # 模拟 第一部分：声学探测 (A 模块)
        report_a = RiskReport(
            module_id="A",
            risk_score=random.uniform(0.0, 0.2),
            suggestion=DecisionType.PASS.value,
            reason="声纹特征正常",
            evidence={"spectrogram": "normal"},
            latency_ms=random.uniform(5, 12)
        )

        # 模拟 第二部分：ASR 行为 (B 模块)
        report_b = RiskReport(
            module_id="B",
            risk_score=random.uniform(0.0, 0.2),
            suggestion=DecisionType.PASS.value,
            reason="识别置信度高",
            evidence={"confidence": 0.98},
            latency_ms=random.uniform(15, 25)
        )

        # 模拟 第三部分：最终生成 (C 模块)
        report_c = RiskReport(
            module_id="C",
            risk_score=c_risk,
            suggestion=DecisionType.BLOCK.value if c_risk > 0.8 else DecisionType.PASS.value,
            reason=c_reason,
            evidence={"policy_check": "failed" if c_risk > 0.5 else "passed"},
            latency_ms=random.uniform(5, 10)
        )

        return [report_a, report_b, report_c]

    async def run_fusion(self, reports: List[RiskReport]) -> Tuple[float, str]:
        """
        风险融合引擎：计算总分 -> 物理执行 -> 数据存证
        """
        if not reports:
            return 0.0, "System Idle"

        total_score = 0.0
        decision = DecisionType.PASS.value

        # 1. 逻辑判定：加权计算 + 一票否决
        for r in reports:
            if r.risk_score > 0.8:  # 触发一票否决
                total_score = r.risk_score
                decision = DecisionType.BLOCK.value
                break
            w = self.weights.get(r.module_id, 0.0)
            total_score += r.risk_score * w
        else:
            # 正常加权判定
            if total_score > 0.7:
                decision = DecisionType.BLOCK.value
            elif total_score > 0.4:
                decision = DecisionType.WARN.value
            else:
                decision = DecisionType.PASS.value

        # 2. 物理执行：调用硬件接口（亮灯/断开继电器）
        vguard_hw.set_status(decision)

        # 3. 数据存证：将高风险行为计入影子模式数据库
        if decision != DecisionType.PASS.value or total_score > 0.3:
            try:
                db_manager.save_log(total_score, decision, reports)
            except Exception as e:
                print(f"[Engine] 数据库存证失败: {e}")

        return round(total_score, 2), decision
'''
'''
import asyncio
from typing import List
from core.protocol import RiskReport

class VGuardEngine:
    def __init__(self, modules: List):
        self.modules = modules
        # 风险权重分配 (这是答辩时可以展开讲的“算法优化点”)
        self.weights = {
            "A": 0.4,  # 声学物理层风险最高 (直接攻击)
            "B": 0.3,  # ASR 行为
            "C": 0.3   # 语义逻辑
        }

    async def run_pipeline(self, audio_data: bytes, vehicle_state: dict):
        """
        核心调度流水线
        """
        context = {"audio": audio_data, "state": vehicle_state}

        # 1. 并发调用 A/B/C 模块 (并行计算提升响应速度)
        # 就像三个专家同时对病人进行诊断
        tasks = [mod.detect(context) for mod in self.modules]
        reports: List[RiskReport] = await asyncio.gather(*tasks)

        # 2. 风险融合算法 (Weighted Sum Model)
        # 公式: $TotalRisk = \sum (Score_i \times Weight_i)$
        total_risk = sum(
            r.risk_score * self.weights.get(r.module_id, 0.1)
            for r in reports
        )

        # 3. 最终决策判决
        if total_risk > 0.7:
            decision = "BLOCK"
        elif total_risk > 0.4:
            decision = "WARN"
        else:
            decision = "PASS"

        return {
            "final_decision": decision,
            "total_risk": round(total_risk, 2),
            "breakdown": [r.to_dict() for r in reports]
        }
'''