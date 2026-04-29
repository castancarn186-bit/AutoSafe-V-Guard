# asr_risk_model.py
"""
增强版ASR风险模型 - 按词匹配优先版 + 强制指令集匹配
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
from audio_preprocessor import AudioPreprocessor

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

# 数据集加载（可选，需要安装 datasets）
try:
    from datasets import load_dataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("⚠️ 数据集加载未安装，跳过 MAC-SLU 扩充")


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
    defense_strength: str = "medium"
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
        best_match = None
        best_sim = threshold
        for cand in candidates:
            sim = PhoneticCorrector.similarity(text, cand)
            if sim > best_sim:
                best_sim = sim
                best_match = cand
        return best_match, best_sim


class EnhancedASRRiskModel:
    """增强版ASR风险模型 - 按词匹配优先 + 强制指令集匹配"""

    def __init__(self, config: Optional[EnhancedConfig] = None):
        self.config = config or EnhancedConfig()

        print("\n" + "=" * 70)
        print("🚀 增强版ASR风险模型")
        print("=" * 70)
        print(f"模型: {self.config.model_size}")
        print(f"VAD: {'启用' if self.config.enable_vad else '禁用'}")
        print(f"对抗性防御: {'启用' if self.config.enable_adversarial_defense else '禁用'}")
        print(f"防御强度: {self.config.defense_strength}")
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

        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            enable_adversarial_defense=self.config.enable_adversarial_defense,
            defense_strength=self.config.defense_strength
        )

        # ==================== 有效指令列表（强制匹配目标） ====================
        self.valid_commands = [
            # ==================== 唤醒与通用 ====================
            "你好小V", "退出", "取消", "返回", "帮助", "确认", "是的", "不是",
            "今天天气怎么样", "明天天气", "现在几点了", "今天星期几",

            # ==================== 车窗控制 ====================
            "打开车窗", "关闭车窗", "打开天窗", "关闭天窗",
            "打开左前车窗", "关闭左前车窗", "打开右前车窗", "关闭右前车窗",
            "打开左后车窗", "关闭左后车窗", "打开右后车窗", "关闭右后车窗",
            "车窗留缝",

            # ==================== 车门控制 ====================
            "打开车门", "关闭车门", "打开后备箱", "关闭后备箱", "打开引擎盖", "关闭引擎盖",
            "打开左前门", "关闭左前门", "打开右前门", "关闭右前门",
            "打开左后门", "关闭左后门", "打开右后门", "关闭右后门",

            # ==================== 车灯控制 ====================
            "打开车灯", "关闭车灯", "打开近光灯", "关闭近光灯",
            "打开远光灯", "关闭远光灯", "打开雾灯", "关闭雾灯",
            "打开示廓灯", "关闭示廓灯", "打开双闪", "关闭双闪",
            "打开日间行车灯", "关闭日间行车灯", "打开氛围灯", "关闭氛围灯",

            # ==================== 空调控制 ====================
            "打开空调", "关闭空调", "调高温度", "调低温度",
            "温度调到16度", "温度调到18度", "温度调到20度", "温度调到22度",
            "温度调到24度", "温度调到26度", "温度调到28度",
            "打开制冷", "关闭制冷", "打开制热", "关闭制热",
            "打开内循环", "关闭内循环", "打开外循环", "关闭外循环",
            "打开前除雾", "关闭前除雾", "打开后除雾", "关闭后除雾",
            "风量大一点", "风量小一点", "打开A/C", "关闭A/C", "自动空调",

            # ==================== 座椅控制 ====================
            "座椅加热", "关闭座椅加热", "座椅通风", "关闭座椅通风",
            "座椅按摩", "关闭座椅按摩", "座椅前移", "座椅后移",
            "座椅靠背放倒", "座椅靠背调直", "记忆座椅1", "记忆座椅2", "记忆座椅3",

            # ==================== 后视镜控制 ====================
            "折叠后视镜", "展开后视镜", "调节左后视镜", "调节右后视镜",

            # ==================== 多媒体控制 ====================
            "播放音乐", "暂停播放", "停止播放", "继续播放",
            "上一首", "上一首歌", "下一首", "下一首歌",
            "重复播放", "随机播放", "打开歌词", "关闭歌词",
            "打开收音机", "关闭收音机", "切换电台", "上一个电台", "下一个电台",

            # ==================== 音量控制 ====================
            "增大音量", "减小音量", "音量加", "音量减", "静音", "取消静音",
            "音量调到10", "音量调到20", "音量调到30", "音量最大", "音量最小",

            # ==================== 蓝牙控制 ====================
            "打开蓝牙", "关闭蓝牙", "连接蓝牙", "断开蓝牙",

            # ==================== 导航控制 ====================
            "打开导航", "关闭导航", "导航到家", "导航到公司",
            "导航到加油站", "导航到充电站", "导航到停车场", "导航到最近的地铁站",
            "导航到北京", "导航到上海", "导航到广州", "导航到深圳",
            "开始导航", "停止导航", "查看全程", "查看路况",
            "躲避拥堵", "重新规划", "放大地图", "缩小地图", "2D视图", "3D视图",
            "导航音量调大", "导航音量调小",

            # ==================== 电话通讯 ====================
            "接听电话", "挂断电话", "拒接电话", "拨打电话", "重拨",
            "静音通话", "取消静音", "切换听筒", "切换免提",

            # ==================== 驾驶辅助 ====================
            "打开巡航", "关闭巡航", "巡航加速", "巡航减速",
            "车道保持开启", "车道保持关闭", "自动泊车", "退出泊车",
            "打开360影像", "关闭360影像",

            # ==================== 车辆设置 ====================
            "驾驶模式运动", "驾驶模式经济", "驾驶模式舒适",
            "能量回收高", "能量回收中", "能量回收低",
            "打开ESP", "关闭ESP", "打开陡坡缓降", "关闭陡坡缓降",
            "查看胎压", "查看电量", "查看续航",

            # ==================== 行车记录仪 ====================
            "打开行车记录仪", "关闭行车记录仪", "拍照", "录像", "紧急录制",

            # ==================== 雨刮控制 ====================
            "打开雨刮", "关闭雨刮", "雨刮一档", "雨刮二档", "雨刮三档", "雨刮自动",
        ]

        # ==================== 从 MAC-SLU 数据集扩充 ====================
        self._expand_from_mac_slu()

        print(f"✅ 初始化完成，有效指令数: {len(self.valid_commands)}")

    def _expand_from_mac_slu(self, max_samples=3000):
        """从 MAC-SLU 数据集扩充有效指令集"""
        if not DATASET_AVAILABLE:
            print("   ⚠️ datasets 库未安装，跳过 MAC-SLU 扩充")
            print("   安装命令: pip install datasets")
            return

        print("\n📡 从 MAC-SLU 数据集扩充指令集...")

        try:
            dataset = load_dataset("Gatsby1984/MAC_SLU", split="train")
            print(f"   加载成功，共 {len(dataset)} 条样本")

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            # 车载意图关键词
            car_keywords = [
                "navigation", "poi", "route", "traffic", "map",
                "audio", "music", "radio", "volume",
                "air_conditioner", "temperature", "ac", "climate",
                "window", "door", "trunk", "light", "seat",
                "call", "message", "phone",
                "vehicle", "setting", "battery", "tire",
            ]

            new_commands = []

            for item in dataset:
                text = item.get("text", "")
                intent = item.get("intent", "")

                if not text:
                    continue

                # 清洗
                text = re.sub(r'[，,。！？；：""''《》【】（）]', '', text)
                text = re.sub(r'\s+', '', text)

                if len(text) < 2 or len(text) > 25:
                    continue

                # 过滤车载相关
                is_car = False
                for kw in car_keywords:
                    if kw in intent.lower():
                        is_car = True
                        break

                if not is_car:
                    continue

                if text not in self.valid_commands and text not in new_commands:
                    new_commands.append(text)

            # 合并到 valid_commands
            old_count = len(self.valid_commands)
            self.valid_commands.extend(new_commands)
            self.valid_commands = list(set(self.valid_commands))
            new_count = len(self.valid_commands)

            print(f"   ✅ 从数据集新增 {len(new_commands)} 条指令")
            print(f"   📊 有效指令总数: {old_count} → {new_count} (+{new_count - old_count})")

        except Exception as e:
            print(f"   ⚠️ MAC-SLU 加载失败: {e}")
            print("   可手动安装 datasets 后重试，或跳过")

    def compute_risk(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
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
            audio = self.audio_preprocessor.prepare_for_asr(audio, sample_rate=16000, apply_defense=True)
            timings['defense_ms'] = (time.time() - defense_start) * 1000
        else:
            timings['defense_ms'] = 0

        # 4. ASR转录
        asr_start = time.time()
        asr_result = self.engine.transcribe(audio, sample_rate=16000)
        timings['asr_ms'] = (time.time() - asr_start) * 1000

        if not asr_result.success:
            return {
                'text': '', 'original_text': '', 'confidence': 0.0,
                'risk_score': 1.0, 'risk_level': '高风险 ❌', 'decision': '拒绝',
                'timings': timings
            }

        # 5. 后处理纠错
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
        """后处理纠错 - 按词匹配优先 + 强制指令集匹配"""

        print(f"\n🔧 [后处理-防御] 原始: '{text}'")

        # ==================== 1. 繁简转换 ====================
        text = zhconv.convert(text, 'zh-cn')
        print(f"   ✓ 繁简转换: '{text}'")

        # ==================== 2. 标点符号清理 ====================
        text = re.sub(r'[，,。！？；：""''《》【】（）]', '', text)
        text = re.sub(r'\s+', '', text)

        # ==================== 3. 动词纠错 ====================
        verb_corrections = {
            "把开": "打开", "我开": "打开", "玩开": "打开", "拨开": "打开", "大开": "打开",
            "关地": "关闭", "完毕": "关闭", "玩地": "关闭",
            "好高": "调高", "超高": "调高", "太高": "调高",
            "倒地": "调低", "导致": "调低",
            "小": "减小", "融效": "减小", "点效": "减小",
            "灯大": "增大", "障大": "增大", "真大": "增大",
            "我放": "播放", "哇放": "播放", "拨放": "播放",
            "但停": "暂停", "按停": "暂停",
            "导肮": "导航", "好行": "导航", "好航": "导航", "好像": "导航","打开导航": "导航",
            "炸一": "下一", "一套": "一首",
            "海拳": "打开", "这么": "增大",
        }

        for wrong, correct in verb_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                print(f"   ✓ 动词纠错: '{wrong}' -> '{correct}'")

        # ==================== 4. 名词纠错 ====================
        noun_corrections = {
            "柱门": "车门", "车创": "车窗", "车双": "车窗", "车装": "车窗",
            "车吧": "车门", "车动": "车灯", "车们": "车门",
            "空桥": "空调", "空橋": "空调", "通告": "空调",
            "狼牙": "蓝牙", "狼呀": "蓝牙", "兰牙": "蓝牙", "地藍": "蓝牙",
            "岛航": "导航", "到旁": "导航",
            "音了": "音乐", "大音乐": "音量",
            "音亮": "音量", "营亮": "音量", "一面": "音量",
            "温渡": "温度", "寶寄溫固": "温度",
            "天创": "天窗", "车方": "车窗", "车撞": "车窗",
            "后车厢": "后备箱", "后箱": "后备箱",
            "雨刷": "雨刮", "吐吐": "车灯", "車燈": "车灯", "車動": "车灯",
            "一套車": "一首歌", "我太燃养": "蓝牙", "我太燃養": "蓝牙",
            "冰冰痛": "车门", "王毕端": "车门", "海拳闷": "车门",
            "一吋燈": "车灯", "一吋灯": "车灯",
            "皮疙瘩": "车门", "航道北京": "导航到北京", "航到北京": "导航到北京",
        }

        for wrong, correct in noun_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                print(f"   ✓ 名词纠错: '{wrong}' -> '{correct}'")

        # ==================== 5. 按词匹配 ====================
        verbs = [
            "打开", "关闭", "开启", "关掉", "合上", "收起", "展开", "升起", "降下",
            "调高", "调低", "增大", "减小", "增加", "减少", "提高", "降低", "调大", "调小",
            "播放", "暂停", "停止", "继续", "开始",
            "导航", "查找", "搜索", "规划", "定位",
            "接听", "挂断", "拒绝", "拨号", "拍照", "录像", "切换", "设置",
            "上一", "下一",
        ]

        nouns = [
            "车窗", "左前车窗", "右前车窗", "左后车窗", "右后车窗", "天窗",
            "车门", "左前门", "右前门", "左后门", "右后门", "后备箱", "引擎盖",
            "车灯", "近光灯", "远光灯", "雾灯", "示廓灯", "双闪", "日间行车灯", "氛围灯",
            "空调", "温度", "制冷", "制热", "内循环", "外循环", "前除雾", "后除雾", "风量", "AC",
            "座椅", "座椅加热", "座椅通风", "座椅按摩", "座椅位置",
            "音乐", "收音机", "电台", "蓝牙", "歌词", "列表", "收藏", "音量",
            "导航", "地图", "路线", "路况", "目的地",
            "电话", "通讯录", "联系人",
            "雨刮", "行车记录仪", "充电口", "油箱盖", "胎压", "电量", "续航",
            "一首歌",
        ]

        # 去重并按长度排序
        verbs = list(set(verbs))
        nouns = list(set(nouns))
        verbs.sort(key=len, reverse=True)
        nouns.sort(key=len, reverse=True)

        found_verb = None
        found_noun = None

        for v in verbs:
            if v in text:
                found_verb = v
                break

        for n in nouns:
            if n in text:
                found_noun = n
                break

        if found_verb and found_noun:
            if found_verb in found_noun:
                result = found_noun
            else:
                result = f"{found_verb}{found_noun}"
            print(f"   ✓ 按词匹配: '{found_verb}' + '{found_noun}' -> '{result}'")
            text = result

        # ==================== 6. 拼音纠错（按词匹配） ====================
        if self.config.enable_phonetic_correction and self.phonetic_corrector:
            words = re.findall(r'[\u4e00-\u9fa5]+', text)
            corrected_words = []
            for word in words:
                if word in self.valid_commands:
                    corrected_words.append(word)
                else:
                    best_match, sim = self.phonetic_corrector.find_best_match(word, self.valid_commands, threshold=0.5)
                    if best_match:
                        print(f"   ✓ 拼音纠错(词): '{word}' -> '{best_match}' (相似度: {sim:.3f})")
                        corrected_words.append(best_match)
                    else:
                        corrected_words.append(word)
            text = ''.join(corrected_words)

        print(f"   📝 结果: '{text}'")

        return text

    def cleanup(self):
        self.engine.cleanup()


OptimizedASRRiskModel = EnhancedASRRiskModel
OptimizedConfig = EnhancedConfig


def test_model():
    import librosa

    config = EnhancedConfig(
        model_size="tiny",
        enable_vad=False,
        enable_adversarial_defense=True,
        defense_strength="medium",
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
