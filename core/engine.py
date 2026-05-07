# core/engine.py
import asyncio
import time
import logging
import librosa
from core.protocol import SystemContext, DetectionResult
from core.state import shared_state
from typing import Dict, Any

# 导入探测器
from modules.module1_acoustic.detector import AcousticDetector
from modules.module2_ASR.detector import ASRDetector
from modules.module3_semantic.detector import SemanticDetector

# 导入声学智能体
from modules.module1_acoustic.agent_a_acoustic import AcousticAgent


class VGuardEngine:
    def __init__(self):
        self.logger = logging.getLogger("VGuard.Engine")
        self.m1 = AcousticDetector(module_id='A')
        self.m2 = ASRDetector()
        self.m3 = SemanticDetector()
        self.m1.setup()
        self.m2.setup()
        self.m3.setup()
        self.weights = {"A": 0.30, "B": 0.2, "C": 0.5}

        # ---------- 新增：声学智能体集成 ----------
        # 定义一个异步队列，用于接收智能体的预警消息
        self._ambient_alert_queue = asyncio.Queue()
        # 创建声学智能体，并传入一个回调函数，用于将预警放入队列
        async def on_ambient_alert(payload: Dict[str, Any]):
            await self._ambient_alert_queue.put(payload)
            self.logger.warning(f"[环境预警] {payload}")

        self.acoustic_agent = AcousticAgent(
            config={
                "la_model_path": "modules/module1_acoustic/models/aasist_la.pth",
                "pa_model_path": "modules/module1_acoustic/models/aasist_pa.pth",
                "fusion_weights": {"la": 0.7, "pa": 0.3},
                "thresholds": {"PASS": 0.3, "CONFIRM": 0.6}
            },
            publish_callback=on_ambient_alert
        )
        # 启动智能体后台监听（如果当前事件循环已在运行，则需确保在异步环境中启动）
        # 在 __init__ 中无法直接 await，所以保存一个任务，在 engine 启动时再真正启动
        self._agent_start_task = None

    async def start_agent(self):
        """启动声学智能体（需在异步环境中调用）"""
        if self._agent_start_task is None:
            await self.acoustic_agent.start()
            asyncio.create_task(self._process_ambient_alerts())
            self.logger.info("声学智能体已启动")

    async def _process_ambient_alerts(self):
        """后台处理预警消息"""
        while True:
            alert = await self._ambient_alert_queue.get()
            # 可在此处做进一步处理：如记录到数据库、触发额外防御、修改共享状态等
            self.logger.info(f"处理环境预警: {alert}")
            # 示例：将预警存入共享状态（可自定义）
            shared_state.sound_alert = alert  # 假设 shared_state 已有此字段
            # 如果需要，也可以发布到上层
            # await self._publish_external(alert)

    async def analyze_risk_pipeline(self, audio_path: str, speed: float) -> dict:
        start_ts = time.perf_counter()
        try:
            audio_matrix, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            return {"total_risk": 1.0, "decision": "BLOCK", "asr_text": "音频加载失败"}

        # 构建全量上下文
        ctx = SystemContext(
            audio_frame=audio_matrix,
            speed=float(speed),
            weather=shared_state.weather,
            has_pedestrians=False
        )

        # 1. 并发执行 A 和 B
        async def run_a():
            return await asyncio.to_thread(self.m1.detect, ctx)

        async def run_b():
            return await asyncio.to_thread(self.m2.detect, audio_matrix)

        results_ab = await asyncio.gather(run_a(), run_b(), return_exceptions=True)

        # 处理 A 模块
        report_a = results_ab[0] if not isinstance(results_ab[0], Exception) else \
            DetectionResult("A", 0.5, "REVIEW", "声学层异常")

        # 处理 B 模块
        raw_b = results_ab[1]
        if isinstance(raw_b, Exception) or raw_b is None:
            report_b = DetectionResult("B", 0.5, "REVIEW", "ASR异常")
            extracted_text = "识别服务异常"
        elif isinstance(raw_b, dict):
            extracted_text = raw_b.get('text', '识别为空')
            score = raw_b.get('risk_score', 0.0)
            report_b = DetectionResult("B", score, "PASS" if score < 0.6 else "BLOCK", "ASR分析完成")
            report_b.metadata = {"text": extracted_text}
        else:
            report_b = raw_b
            extracted_text = getattr(report_b, "metadata", {}).get("text", "无文本")

        # 2. 串行执行 C 模块
        try:
            ctx.asr_text = extracted_text
            report_c = await asyncio.to_thread(self.m3.run, ctx)
        except Exception as e:
            report_c = DetectionResult("C", 0.8, "BLOCK", f"语义层异常: {e}")

        # 3. 风险融合与决策
        reports = [report_a, report_b, report_c]
        total_risk = 0.0
        for r in reports:
            total_risk += r.risk_score * self.weights.get(r.module_id, 0.33)

        any_block = False
        any_review = False
        for r in reports:
            if r.decision == "BLOCK" or r.risk_score >= 0.8:
                any_block = True
                break
            if r.decision == "REVIEW" or r.risk_score >= 0.5:
                any_review = True

        if any_block:
            decision = "BLOCK"
        elif any_review:
            decision = "REVIEW"
        else:
            decision = "PASS"

        latency = round((time.perf_counter() - start_ts) * 1000, 2)

        self._print_terminal_log(extracted_text, speed, reports, total_risk, decision, latency)

        return {
            "total_risk": total_risk, "decision": decision,
            "reports": reports, "latency_ms": latency, "asr_text": extracted_text
        }

    def _print_terminal_log(self, text, speed, reports, total, decision, latency):
        print("\n" + "═" * 60)
        print(f"🛡️  [V-Guard 安全网关流水线] 评估完成 | 耗时: {latency}ms")
        print(f"🎤 识别文本: '{text}' | 🚗 车速: {speed} km/h")
        print("-" * 60)
        for r in reports:
            print(f"  ▶ [模块 {r.module_id}] 风险分: {r.risk_score:.3f} | 建议: {r.decision:<5} | 详情: {r.reason}")
        print("-" * 60)
        print(f"⚖️  最终网关指数: {total:.3f} -> 执行动作: 【{decision}】")
        print("═" * 60 + "\n")
