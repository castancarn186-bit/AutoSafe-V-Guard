# asr_risk_model.py
"""
增强版ASR风险模型 - 后处理只在防御时生效
"""

import numpy as np
import time
import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import librosa
import zhconv

from asr_engine import create_asr_engine
from confidence_analyzer import ConfidenceAnalyzer

# 拼音纠错依赖
try:
    from pypinyin import lazy_pinyin, Style
    import Levenshtein

    PINYIN_AVAILABLE = True
except ImportError:
    PINYIN_AVAILABLE = False
    print("⚠️ 拼音纠错未安装，请运行: pip install pypinyin python-Levenshtein")

# VAD导入
try:
    import webrtcvad

    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("⚠️ 请安装 webrtcvad: pip install webrtcvad")


@dataclass
class EnhancedConfig:
    """增强配置"""
    model_size: str = "tiny"
    enable_vad: bool = False
    vad_aggressiveness: int = 0
    enable_volume_normalization: bool = True
    min_audio_duration: float = 1.5
    language: str = "zh"
    initial_prompt: str = "请识别以下中文车载语音指令："
    enable_postprocessing: bool = True
    alpha: float = 1.0
    enable_adversarial_defense: bool = True
    defense_noise_std: float = 0.0001
    enable_phonetic_correction: bool = True
    force_dict_match: bool = True


class VADProcessor:
    """VAD处理器"""

    def __init__(self, aggressiveness=0, sample_rate=16000):
        if not VAD_AVAILABLE:
            raise ImportError("请安装 webrtcvad")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio_int16 = (audio * 32767).astype(np.int16)

        if len(audio_int16) % self.frame_size != 0:
            padding = self.frame_size - (len(audio_int16) % self.frame_size)
            audio_int16 = np.pad(audio_int16, (0, padding), 'constant')

        num_frames = len(audio_int16) // self.frame_size
        speech_flags = []

        for i in range(num_frames):
            frame = audio_int16[i * self.frame_size:(i + 1) * self.frame_size]
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                speech_flags.append(is_speech)
            except:
                speech_flags.append(True)

        if len(speech_flags) > 2:
            for i in range(1, len(speech_flags) - 1):
                if speech_flags[i - 1] == 1 or speech_flags[i + 1] == 1:
                    speech_flags[i] = 1

        mask = np.repeat(speech_flags, self.frame_size)
        mask = mask[:len(audio)]

        result = audio.copy()
        result[~mask] = 0
        return result


class PhoneticCorrector:
    """拼音纠错器"""

    @staticmethod
    def similarity(text1: str, text2: str) -> float:
        """计算两个文本的拼音相似度"""
        if not text1 or not text2:
            return 0.0

        try:
            p1 = ''.join(lazy_pinyin(text1, style=Style.NORMAL))
            p2 = ''.join(lazy_pinyin(text2, style=Style.NORMAL))

            if not p1 or not p2:
                return 0.0

            dist = Levenshtein.distance(p1, p2)
            max_len = max(len(p1), len(p2))
            return 1 - dist / max_len if max_len > 0 else 0.0
        except:
            return 0.0

    @staticmethod
    def find_best_match(text: str, candidates: List[str], threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """找到最匹配的候选词"""
        best_match = None
        best_sim = threshold

        for cand in candidates:
            sim = PhoneticCorrector.similarity(text, cand)
            if sim > best_sim:
                best_sim = sim
                best_match = cand

        return best_match, best_sim


class EnhancedASRRiskModel:
    """增强版ASR风险模型"""

    def __init__(self, config: Optional[EnhancedConfig] = None):
        self.config = config or EnhancedConfig()

        print("\n" + "=" * 70)
        print("🚀 增强版ASR风险模型")
        print("=" * 70)
        print(f"模型: {self.config.model_size}")
        print(f"VAD: {'启用' if self.config.enable_vad else '禁用'}")
        print(f"对抗性防御: {'启用' if self.config.enable_adversarial_defense else '禁用'}")
        print(f"后处理纠错: {'启用(仅防御)' if self.config.enable_postprocessing else '禁用'}")
        print(f"拼音纠错: {'启用' if self.config.enable_phonetic_correction and PINYIN_AVAILABLE else '禁用'}")
        print(f"强制字典匹配: {'启用' if self.config.force_dict_match else '禁用'}")

        self.vad = None
        if self.config.enable_vad and VAD_AVAILABLE:
            self.vad = VADProcessor(aggressiveness=self.config.vad_aggressiveness)

        print("\n📡 初始化ASR引擎...")
        self.engine = create_asr_engine(
            model_size=self.config.model_size,
            device="cpu",
            compute_type="int8",
            language=self.config.language
        )

        self.confidence_analyzer = ConfidenceAnalyzer(low_conf_threshold=0.5)
        self.phonetic_corrector = PhoneticCorrector() if PINYIN_AVAILABLE else None

        # 常见正确指令列表（字典）
        self.valid_commands = [
            "上一首歌", "下一首歌", "播放音乐", "暂停播放",
            "关闭导航", "导航到北京", "打开空调", "关闭空调",
            "打开蓝牙", "关闭蓝牙", "打开车灯", "关闭车灯",
            "打开车窗", "关闭车窗", "打开车门", "关闭车门",
            "减小音量", "增大音量", "调低温度", "调高温度",
        ]

        print("✅ 初始化完成")

    def compute_risk(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """风险评估 - 后处理只在防御时生效"""
        timings = {}

        # 1. 预处理
        pre_start = time.time()
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        timings['preprocess_ms'] = (time.time() - pre_start) * 1000

        # 2. VAD处理
        if self.vad:
            vad_start = time.time()
            audio = self.vad.process(audio)
            timings['vad_ms'] = (time.time() - vad_start) * 1000
        else:
            timings['vad_ms'] = 0

        # 3. 防御处理
        if self.config.enable_adversarial_defense:
            defense_start = time.time()
            noise = np.random.normal(0, self.config.defense_noise_std, audio.shape)
            audio = audio + noise
            audio = np.clip(audio, -0.99, 0.99)
            timings['defense_ms'] = (time.time() - defense_start) * 1000
        else:
            timings['defense_ms'] = 0

        # 4. ASR转录
        asr_start = time.time()
        asr_result = self.engine.transcribe(audio, sample_rate=16000)
        timings['asr_ms'] = (time.time() - asr_start) * 1000

        # 5. 后处理纠错 - 只在防御开启时生效
        post_start = time.time()
        if self.config.enable_adversarial_defense and self.config.enable_postprocessing:
            corrected_text = self._postprocess(asr_result.text)
        else:
            corrected_text = asr_result.text
        timings['post_ms'] = (time.time() - post_start) * 1000

        # 6. 置信度
        confidence = self.confidence_analyzer.analyze(asr_result)

        # 7. 风险计算
        risk = 1.0 - confidence.confidence_score
        risk = max(0.0, min(1.0, risk))

        timings['total_ms'] = sum(timings.values())

        if risk < 0.3:
            risk_level = "低风险 ✅"
            decision = "接受"
        elif risk < 0.7:
            risk_level = "中风险 ⚠️"
            decision = "人工确认"
        else:
            risk_level = "高风险 ❌"
            decision = "拒绝"

        return {
            'text': corrected_text,
            'original_text': asr_result.text,
            'confidence': confidence.confidence_score,
            'risk_score': risk,
            'risk_level': risk_level,
            'decision': decision,
            'timings': timings
        }

    def _postprocess(self, text: str) -> str:
        """后处理纠错 - 完整版（强制字典匹配）"""

        print(f"\n🔧 [后处理-防御] 原始: '{text}'")

        # ==================== 1. 特定错误纠正 ====================
                corrections = {
            "一首歌": "上一首歌",
            "下一首歌": "上一首歌",
            "但停播放": "暂停播放",
            "倒地温度": "调低温度",
            "倒地溫度": "调低温度",
            "关闭到旁": "关闭导航",
            "关闭到航": "关闭导航",
            "关闭到行": "关闭导航",
            "关闭狼呀": "关闭蓝牙",
            "关闭狼牙": "关闭蓝牙",
            "关闭空桥": "关闭空调",
            "关闭车灯": "关闭车窗",
            "关闭车窗": "关闭车门",
            "咱们打一面": "增大音量",
            "哇哇音乐": "播放音乐",
            "哇噪音乐": "播放音乐",
            "增大一面": "增大音量",
            "大一首歌": "下一首歌",
            "大蜜首歌": "上一首歌",
            "好 航道北京": "导航到北京",
            "好,航到北京": "导航到北京",
            "好,行到北京": "导航到北京",
            "好,高温度": "调高温度",
            "好导航到北京": "导航到北京",
            "好航到北京": "导航到北京",
            "好行到北京": "导航到北京",
            "好高温度": "调高温度",
            "好高溫度": "调高温度",
            "完毕同条": "关闭空调",
            "完畢同條": "关闭空调",
            "害臭门": "打开车门",
            "导致温度": "调低温度",
            "導致溫度": "调低温度",
            "導航": "导航",
            "小鈴鐺": "减小音量",
            "小铃铛": "减小音量",
            "开空桥": "打开空调",
            "开车动": "打开车灯",
            "开车灯": "打开车灯",
            "开车门": "打开车门",
            "我太燃养": "打开蓝牙",
            "我太燃養": "打开蓝牙",
            "我太蓝牙": "打开蓝牙",
            "我开蓝牙": "打开蓝牙",
            "我放音乐": "播放音乐",
            "我放音樂": "播放音乐",
            "打开车灯": "关闭车灯",
            "打開": "打开",
            "把开蓝牙": "打开蓝牙",
            "把开车撞": "打开车窗",
            "把開車撞": "打开车窗",
            "按停播放": "暂停播放",
            "按停鍋放": "暂停播放",
            "按停锅放": "暂停播放",
            "播放": "播放",
            "暫停": "暂停",
            "暴力手割": "下一首歌",
            "海拳闷": "打开车门",
            "溫度": "温度",
            "灯大营亮": "增大音量",
            "炸一套車": "下一首歌",
            "炸一套车": "下一首歌",
            "点效音量": "减小音量",
            "燈大營亮": "增大音量",
            "玩地蓝牙": "关闭蓝牙",
            "玩地藍牙": "关闭蓝牙",
            "玩地車方": "关闭车窗",
            "玩地车方": "关闭车窗",
            "空調": "空调",
            "航道北京": "导航到北京",
            "藍牙": "蓝牙",
            "融效音量": "减小音量",
            "車燈": "车灯",
            "車窗": "车窗",
            "車門": "车门",
            "開空橋": "打开空调",
            "開車動": "打开车灯",
            "開車燈": "打开车灯",
            "開車門": "打开车门",
            "關閉": "关闭",
            "關閉到旁": "关闭导航",
            "關閉到航": "关闭导航",
            "關閉到行": "关闭导航",
            "關閉空橋": "关闭空调",
            "闹地温度": "调低温度",
            "障大一面": "增大音量",
            "音樂": "音乐",
            "音量": "音量",
            "鬧地溫度": "调低温度",
        }

        for wrong, correct in corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                print(f"   ✓ 替换: '{wrong}' -> '{correct}'")

        # ==================== 2. 繁简转换 ====================
        text = zhconv.convert(text, 'zh-cn')

        # ==================== 3. 标点符号清理 ====================
        text = re.sub(r'[，,。！？；：""''《》【】（）]', '', text)
        text = re.sub(r'\s+', '', text)

        # ==================== 4. 拼音纠错 + 强制字典匹配 ====================
        if self.config.enable_phonetic_correction and self.phonetic_corrector:
            # 先尝试整体匹配
            if self.config.force_dict_match:
                best_match, sim = self.phonetic_corrector.find_best_match(text, self.valid_commands, threshold=0.4)
                if best_match and sim > 0.4:
                    print(f"   ✓ 强制匹配: '{text}' -> '{best_match}' (相似度: {sim:.3f})")
                    text = best_match
                else:
                    # 按词分割匹配
                    words = text.split()
                    corrected_words = []
                    for word in words:
                        best_match, sim = self.phonetic_corrector.find_best_match(word, self.valid_commands,
                                                                                  threshold=0.5)
                        if best_match:
                            corrected_words.append(best_match)
                            print(f"   ✓ 词匹配: '{word}' -> '{best_match}' (相似度: {sim:.3f})")
                        else:
                            corrected_words.append(word)
                    text = ''.join(corrected_words)
            else:
                # 不强制匹配，只做拼音纠错
                words = text.split()
                corrected_words = []
                for word in words:
                    if word in self.valid_commands:
                        corrected_words.append(word)
                    else:
                        best_match, sim = self.phonetic_corrector.find_best_match(word, self.valid_commands,
                                                                                  threshold=0.5)
                        if best_match:
                            corrected_words.append(best_match)
                            print(f"   ✓ 拼音纠错: '{word}' -> '{best_match}' (相似度: {sim:.3f})")
                        else:
                            corrected_words.append(word)
                text = ''.join(corrected_words)

        # ==================== 5. 最终验证：如果不在字典中，强制匹配最相似的 ====================
        if self.config.force_dict_match and text not in self.valid_commands:
            best_match, sim = self.phonetic_corrector.find_best_match(text, self.valid_commands, threshold=0.3)
            if best_match:
                print(f"   ✓ 最终强制匹配: '{text}' -> '{best_match}' (相似度: {sim:.3f})")
                text = best_match

        print(f"   📝 结果: '{text}'")

        return text

    def cleanup(self):
        """清理资源"""
        self.engine.cleanup()


OptimizedASRRiskModel = EnhancedASRRiskModel
OptimizedConfig = EnhancedConfig


def test_model():
    """测试模型"""
    import librosa

    config = EnhancedConfig(
        model_size="tiny",
        enable_vad=False,
        enable_adversarial_defense=True,
        enable_postprocessing=True,
        enable_phonetic_correction=True,
        force_dict_match=True
    )

    model = EnhancedASRRiskModel(config)

    try:
        audio, sr = librosa.load("test.wav", sr=16000, mono=True)
        result = model.compute_risk(audio, sr)
        print(f"\n识别: {result['text']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"耗时: {result['timings']['total_ms']:.1f}ms")
    finally:
        model.cleanup()


if __name__ == "__main__":
    test_model()
