# asr_agent.py
"""
ASRAgent - 自主决策的 ASR 智能体
功能：
- 接收 audio.ready 事件，进行 ASR 识别
- 置信度过低时主动请求重录
- 重复攻击检测
- 自学习纠错规则并持久化
- 发布转录结果和风险信息
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

import librosa
import numpy as np

from base_agent import BaseAgent
from asr_risk_model import EnhancedASRRiskModel, EnhancedConfig


@dataclass
class CommandHistory:
    """指令历史记录"""
    text: str
    original: str
    confidence: float
    risk_score: float
    timestamp: float


class ASRAgent(BaseAgent):
    """ASR 智能体 - 自主决策、自学习"""

    def __init__(self, bus, config: dict = None):
        super().__init__("ASRAgent", bus)

        self.config = config or {}

        # 初始化 ASR 风险模型（复用你现有的）
        asr_config = self._build_asr_config()
        self.risk_model = EnhancedASRRiskModel(asr_config)

        # 自学习纠错字典（运行时学习的新规则）
        self.learned_corrections: Dict[str, str] = {}
        self.corrections_file = "asr_learned_corrections.json"

        # 历史指令记录（用于重复攻击检测）
        self.recent_commands: deque = deque(maxlen=50)  # 最多保留50条
        self.repeat_window_seconds = 30  # 重复检测时间窗口（秒）
        self.repeat_threshold = 3  # 窗口内相同低置信度指令超过此次数告警

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "retry_requests": 0,
            "corrections_learned": 0,
            "repeat_attacks_detected": 0
        }

        # 加载之前学习的纠错规则
        self._load_learned_corrections()

        print(f"✅ ASRAgent 初始化完成")
        print(f"   - 纠错规则: {len(self.learned_corrections)} 条（自学习）")
        print(f"   - 历史记录容量: {self.recent_commands.maxlen}")

    def _build_asr_config(self) -> EnhancedConfig:
        """构建 ASR 风险模型配置"""
        return EnhancedConfig(
            model_size=self.config.get("model_size", "tiny"),
            enable_vad=self.config.get("enable_vad", False),
            enable_adversarial_defense=self.config.get("enable_adversarial_defense", True),
            defense_strength=self.config.get("defense_strength", "medium"),
            enable_postprocessing=self.config.get("enable_postprocessing", True),
            enable_phonetic_correction=self.config.get("enable_phonetic_correction", True),
            force_dict_match=self.config.get("force_dict_match", True),
            commands_json_path=self.config.get("commands_json_path", "commands.json")
        )

    async def subscribe_all(self):
        """订阅总线事件"""
        # 订阅音频就绪事件
        await self.bus.subscribe("audio.ready", self.handle_audio_ready)
        # 订阅重录响应（可选）
        await self.bus.subscribe("audio.retry_response", self.handle_retry_response)
        print("📡 ASRAgent 已订阅: audio.ready, audio.retry_response")

    async def handle_audio_ready(self, payload: dict):
        """
        处理音频就绪事件 - 核心决策逻辑

        决策流程：
        1. ASR 识别 + 风险分析
        2. 置信度过低 → 请求重录（不发布低质量结果）
        3. 重复攻击检测 → 发布告警
        4. 自学习纠错 → 更新词典
        5. 发布最终结果
        """
        audio_path = payload.get("audio_path")
        session_id = payload.get("session_id", "")

        if not audio_path or not os.path.exists(audio_path):
            await self.publish("asr.error", {
                "error": f"Audio file not found: {audio_path}",
                "session_id": session_id
            })
            return

        self.stats["total_processed"] += 1

        # 1. 加载音频
        audio, sr = self._load_audio(audio_path)
        if audio is None:
            await self.publish("asr.error", {
                "error": f"Failed to load audio: {audio_path}",
                "session_id": session_id
            })
            return

        # 2. ASR 识别
        result = self.risk_model.compute_risk(audio, sr)

        original_text = result.get("original_text", "")
        corrected_text = result.get("text", original_text)
        confidence = result.get("confidence", 0.0)
        risk_score = result.get("risk_score", 0.0)
        risk_level = result.get("risk_level", "未知")
        decision = result.get("decision", "接受")
        timings = result.get("timings", {})

        print(f"\n🎤 [ASRAgent] 处理音频: {os.path.basename(audio_path)}")
        print(f"   原始识别: '{original_text}'")
        print(f"   纠错后: '{corrected_text}'")
        print(f"   置信度: {confidence:.3f}, 风险分: {risk_score:.3f}")

        # ========== 3. 自主决策：置信度过低 → 请求重录 ==========
        if confidence < 0.6:
            self.stats["retry_requests"] += 1
            await self.publish("asr.request_retry", {
                "reason": "ASR confidence too low (<0.6), possible adversarial noise or poor audio",
                "original_text": original_text,
                "confidence": confidence,
                "session_id": session_id,
                "audio_path": audio_path
            })
            print(f"   ⚠️ 置信度过低({confidence:.2f})，已请求重录")
            return  # 不继续发布低质量结果

        # ========== 4. 重复攻击检测 ==========
        if self._detect_repeat_attack(corrected_text, confidence):
            self.stats["repeat_attacks_detected"] += 1
            await self.publish("asr.repeated_attack", {
                "text": corrected_text,
                "original_text": original_text,
                "confidence": confidence,
                "reason": f"短时间内相同低置信度指令出现超过{self.repeat_threshold}次",
                "session_id": session_id
            })
            print(f"   🚨 检测到重复攻击: '{corrected_text}'")

        # ========== 5. 自学习纠错 ==========
        if corrected_text != original_text:
            learned = self._learn_correction(original_text, corrected_text, confidence)
            if learned:
                await self.publish("asr.correction_applied", {
                    "from": original_text,
                    "to": corrected_text,
                    "confidence": confidence,
                    "session_id": session_id
                })
                print(f"   📝 自学习: '{original_text}' -> '{corrected_text}'")

        # ========== 6. 记录历史 ==========
        self._add_to_history(corrected_text, original_text, confidence, risk_score)

        # ========== 7. 发布最终结果 ==========
        await self.publish("text.transcribed", {
            "text": corrected_text,
            "original_text": original_text,
            "confidence": confidence,
            "session_id": session_id,
            "audio_path": audio_path,
            "timings_ms": timings
        })

        await self.publish("risk.asr", {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "decision": decision,
            "original_text": original_text,
            "corrected_text": corrected_text,
            "confidence": confidence,
            "session_id": session_id,
            "ambient_risk": payload.get("ambient_risk", 0.0)  # 接收环境风险上下文
        })

        print(f"   ✅ 已发布: text.transcribed, risk.asr")

    async def handle_retry_response(self, payload: dict):
        """处理重录响应（可选）"""
        session_id = payload.get("session_id")
        accepted = payload.get("accepted", False)

        if accepted:
            print(f"   📞 重录请求已接受，session: {session_id}")
        else:
            print(f"   ⚠️ 重录请求被拒绝，session: {session_id}")

    def _load_audio(self, audio_path: str, target_sr: int = 16000):
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            # 限制音频长度（最多10秒）
            max_samples = target_sr * 10
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            return audio, sr
        except Exception as e:
            print(f"❌ 加载音频失败 {audio_path}: {e}")
            return None, None

    def _learn_correction(self, wrong: str, correct: str, confidence: float) -> bool:
        """
        自学习纠错规则

        条件：
        - 置信度 > 0.8（只学习高置信度的纠正）
        - 错误文本长度 > 1
        - 不是已有的纠错规则
        """
        # 只学习高置信度的纠正
        if confidence < 0.8:
            return False

        # 避免学习无意义的纠正
        if len(wrong) < 2 or wrong == correct:
            return False

        # 检查是否已经是已知规则
        if wrong in self.learned_corrections or wrong in self._get_builtin_corrections():
            return False

        # 学习新规则
        self.learned_corrections[wrong] = correct
        self.stats["corrections_learned"] += 1

        # 实时注入到 risk_model 的 postprocess 中
        self._inject_correction_to_model(wrong, correct)

        # 持久化
        self._save_learned_corrections()

        return True

    def _get_builtin_corrections(self) -> set:
        """获取内置纠错规则（硬编码的）"""
        # 从 asr_risk_model.py 中提取的硬编码纠错
        builtin = {
            "把开", "我开", "玩开", "拨开", "关地", "完毕", "倒车", "柱门", "车创",
            "音了", "音亮", "温渡", "天创", "车双", "空桥"
        }
        return builtin

    def _inject_correction_to_model(self, wrong: str, correct: str):
        """
        将新学习的纠错规则注入到 risk_model 的 postprocess 中
        这样后续识别会立即生效
        """
        # 方法1: 如果 risk_model 有动态更新方法
        if hasattr(self.risk_model, 'add_correction'):
            self.risk_model.add_correction(wrong, correct)

        # 方法2: 直接修改 risk_model 的内部纠错字典
        # 注意：这取决于 asr_risk_model.py 的实现
        # 你可以在 EnhancedASRRiskModel 中添加一个 correction_rules 字典

    def _detect_repeat_attack(self, text: str, confidence: float) -> bool:
        """
        检测重复攻击

        判断条件：
        - 时间窗口内相同文本出现次数 >= repeat_threshold
        - 这些出现中置信度都偏低 (< 0.7)
        """
        now = time.time()

        # 统计时间窗口内相同文本的记录
        similar_commands = [
            cmd for cmd in self.recent_commands
            if cmd.text == text
               and now - cmd.timestamp < self.repeat_window_seconds
               and cmd.confidence < 0.7
        ]

        return len(similar_commands) >= self.repeat_threshold

    def _add_to_history(self, text: str, original: str, confidence: float, risk_score: float):
        """添加指令到历史"""
        self.recent_commands.append(CommandHistory(
            text=text,
            original=original,
            confidence=confidence,
            risk_score=risk_score,
            timestamp=time.time()
        ))

    def _load_learned_corrections(self):
        """加载之前学习的纠错规则"""
        if os.path.exists(self.corrections_file):
            try:
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    self.learned_corrections = json.load(f)
                print(f"📚 加载自学习纠错规则: {len(self.learned_corrections)} 条")
            except Exception as e:
                print(f"⚠️ 加载纠错规则失败: {e}")

    def _save_learned_corrections(self):
        """保存学习的纠错规则"""
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump(self.learned_corrections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存纠错规则失败: {e}")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            "learned_corrections_count": len(self.learned_corrections),
            "history_size": len(self.recent_commands)
        }

    async def start(self):
        """启动 Agent"""
        print(f"🚀 ASRAgent 启动，已处理 {self.stats['total_processed']} 个音频")

    def cleanup(self):
        """清理资源"""
        if hasattr(self.risk_model, 'cleanup'):
            self.risk_model.cleanup()
        print(f"🧹 ASRAgent 清理完成，最终统计: {self.get_stats()}")