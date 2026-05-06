# asr_risk_model.py
"""
增强版ASR风险模型 - 按词匹配优先版 + 强制指令集匹配
支持动态纠错规则（运行时学习）
"""

import numpy as np
import time
import re
import json
import os
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
    commands_json_path: str = "commands.json"


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


class CommandLoader:
    """指令加载器 - 从JSON文件加载"""

    def __init__(self, json_path: str = "commands.json"):
        self.json_path = json_path
        self._commands = []
        self._load()

    def _load(self):
        """加载JSON文件"""
        if not os.path.exists(self.json_path):
            print(f"⚠️ 指令文件不存在: {self.json_path}")
            print(f"   将使用默认指令集")
            self._load_default_commands()
            return

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_commands = []
            base_commands = data.get("base_commands", {})
            for category, cmd_list in base_commands.items():
                all_commands.extend(cmd_list)

            self._commands = sorted(list(set(all_commands)))
            print(f"✅ 从 {self.json_path} 加载指令集")
            print(f"   📊 总指令数: {len(self._commands)}")

        except Exception as e:
            print(f"⚠️ 加载指令文件失败: {e}")
            self._load_default_commands()

    def _load_default_commands(self):
        """加载默认指令集"""
        default_commands = [
            "你好小V", "退出", "取消", "返回", "帮助", "确认", "是的", "不是",
            "打开车窗", "关闭车窗", "打开天窗", "关闭天窗", "打开车门", "关闭车门",
            "打开空调", "关闭空调", "调高温度", "调低温度", "播放音乐", "暂停播放",
            "上一首", "下一首", "增大音量", "减小音量", "静音", "取消静音",
            "打开导航", "关闭导航", "接听电话", "挂断电话"
        ]
        self._commands = sorted(default_commands)
        print(f"   📊 使用默认指令集: {len(self._commands)} 条")

    def get_all_commands(self) -> List[str]:
        return self._commands.copy()

    def reload(self):
        self._load()

    def __len__(self):
        return len(self._commands)


class EnhancedASRRiskModel:
    """增强版ASR风险模型 - 按词匹配优先 + 强制指令集匹配"""

    def __init__(self, config: Optional[EnhancedConfig] = None):
        self.config = config or EnhancedConfig()

        # ==================== 新增：动态纠错规则（运行时学习） ====================
        self.dynamic_corrections: Dict[str, str] = {}
        self.dynamic_corrections_file = "asr_dynamic_corrections.json"

        print("\n" + "=" * 70)
        print("🚀 增强版ASR风险模型")
        print("=" * 70)
        print(f"模型: {self.config.model_size}")
        print(f"VAD: {'启用' if self.config.enable_vad else '禁用'}")
        print(f"对抗性防御: {'启用' if self.config.enable_adversarial_defense else '禁用'}")
        print(f"防御强度: {self.config.defense_strength}")
        print(f"后处理纠错: {'启用' if self.config.enable_postprocessing else '禁用'}")
        print(f"拼音纠错: {'启用' if self.config.enable_phonetic_correction and PINYIN_AVAILABLE else '禁用'}")
        print(f"强制字典匹配: {'启用' if self.config.force_dict_match else '禁用'}")
        print(f"动态纠错: {'启用' if self.config.enable_postprocessing else '禁用'}")

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

        # ==================== 加载指令集 ====================
        print("\n📚 加载指令集...")
        self.command_loader = CommandLoader(self.config.commands_json_path)
        self.valid_commands = self.command_loader.get_all_commands()
        self.valid_commands.sort(key=len, reverse=True)

        # ==================== 加载动态纠错规则 ====================
        self._load_dynamic_corrections()

        print(f"✅ 初始化完成，有效指令数: {len(self.valid_commands)}")
        print(f"   📚 动态纠错规则: {len(self.dynamic_corrections)} 条")

    # ==================== 新增：动态纠错规则管理 ====================

    def _load_dynamic_corrections(self):
        """加载之前学习的动态纠错规则"""
        if os.path.exists(self.dynamic_corrections_file):
            try:
                with open(self.dynamic_corrections_file, 'r', encoding='utf-8') as f:
                    self.dynamic_corrections = json.load(f)
                print(f"✅ 加载动态纠错规则: {len(self.dynamic_corrections)} 条")
            except Exception as e:
                print(f"⚠️ 加载动态纠错规则失败: {e}")

    def _save_dynamic_corrections(self):
        """保存动态纠错规则"""
        try:
            with open(self.dynamic_corrections_file, 'w', encoding='utf-8') as f:
                json.dump(self.dynamic_corrections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存动态纠错规则失败: {e}")

    def add_correction(self, wrong: str, correct: str) -> bool:
        """
        动态添加纠错规则（运行时学习）

        Args:
            wrong: 错误的识别结果
            correct: 正确的文本

        Returns:
            是否成功添加
        """
        if not wrong or not correct or wrong == correct:
            return False

        # 避免添加重复规则
        if wrong in self.dynamic_corrections:
            if self.dynamic_corrections[wrong] == correct:
                return False
            else:
                print(f"   🔄 更新纠错规则: '{wrong}' -> '{correct}' (原: '{self.dynamic_corrections[wrong]}')")

        self.dynamic_corrections[wrong] = correct
        self._save_dynamic_corrections()
        print(f"   📚 动态添加纠错: '{wrong}' -> '{correct}'")
        return True

    def add_corrections_batch(self, corrections: Dict[str, str]) -> int:
        """批量添加纠错规则"""
        added = 0
        for wrong, correct in corrections.items():
            if self.add_correction(wrong, correct):
                added += 1
        return added

    def get_dynamic_corrections(self) -> Dict[str, str]:
        """获取所有动态纠错规则"""
        return self.dynamic_corrections.copy()

    def remove_correction(self, wrong: str) -> bool:
        """移除纠错规则"""
        if wrong in self.dynamic_corrections:
            del self.dynamic_corrections[wrong]
            self._save_dynamic_corrections()
            return True
        return False

    def clear_dynamic_corrections(self):
        """清空所有动态纠错规则"""
        self.dynamic_corrections.clear()
        self._save_dynamic_corrections()

    # ==================== 核心方法 ====================

    def compute_risk(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        timings = {}

        pre_start = time.time()
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        timings['preprocess_ms'] = (time.time() - pre_start) * 1000

        if self.vad:
            vad_start = time.time()
            audio = self.vad.process(audio)
            timings['vad_ms'] = (time.time() - vad_start) * 1000
        else:
            timings['vad_ms'] = 0

        if self.config.enable_adversarial_defense:
            defense_start = time.time()
            audio = self.audio_preprocessor.prepare_for_asr(audio, sample_rate=16000, apply_defense=True)
            timings['defense_ms'] = (time.time() - defense_start) * 1000
        else:
            timings['defense_ms'] = 0

        asr_start = time.time()
        asr_result = self.engine.transcribe(audio, sample_rate=16000)
        timings['asr_ms'] = (time.time() - asr_start) * 1000

        if not asr_result.success:
            return {
                'text': '', 'original_text': '', 'confidence': 0.0,
                'risk_score': 1.0, 'risk_level': '高风险 ❌', 'decision': '拒绝',
                'timings': timings
            }

        post_start = time.time()
        if self.config.enable_postprocessing:
            corrected_text = self._postprocess(asr_result.text)
        else:
            corrected_text = asr_result.text
        timings['post_ms'] = (time.time() - post_start) * 1000

        confidence = self.confidence_analyzer.analyze(asr_result)
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
        """后处理纠错 - 按词匹配 + 整条指令拼音纠错 + 动态纠错"""

        print(f"\n🔧 [后处理-防御] 原始: '{text}'")

        # ==================== 优先级1: 动态纠错规则（运行时学习） ====================
        for wrong, correct in self.dynamic_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                print(f"   ✓ 动态纠错: '{wrong}' -> '{correct}'")

        # ==================== 1. 繁简转换 ====================
        text = zhconv.convert(text, 'zh-cn')
        print(f"   ✓ 繁简转换: '{text}'")

        # ==================== 2. 标点符号清理 ====================
        text = re.sub(r'[，,。！？；：""''《》【】（）]', '', text)
        text = re.sub(r'\s+', '', text)

        # ==================== 3. 动词纠错 ====================
        verb_corrections = {
            "把开": "打开", "我开": "打开", "玩开": "打开", "拨开": "打开", "大开": "打开",
            "关地": "关闭", "完毕": "关闭", "玩地": "关闭", "完畢": "关闭",
            "好高": "调高", "超高": "调高", "太高": "调高",
            "倒地": "调低", "导致": "调低",
            "小": "减小", "融效": "减小", "点效": "减小", "小鈴鐺": "减小",
            "灯大": "增大", "障大": "增大", "真大": "增大",
            "我放": "播放", "哇放": "播放", "拨放": "播放",
            "但停": "暂停", "按停": "暂停",
            "导肮": "导航", "好行": "导航", "好航": "导航", "好像": "导航",
            "炸一": "下一", "一套": "一首",
            "海拳": "打开", "这么": "增大", "中大一秒": "增大音量",
        }

        for wrong, correct in verb_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                print(f"   ✓ 动词纠错: '{wrong}' -> '{correct}'")

        # ==================== 4. 名词纠错 ====================
        noun_corrections = {
            "柱门": "车门", "车创": "车窗", "车双": "车窗", "车装": "车窗",
            "车吧": "车门", "车动": "车灯", "车们": "车门", "車動": "车灯",
            "空桥": "空调", "空橋": "空调", "通告": "空调",
            "狼牙": "蓝牙", "狼呀": "蓝牙", "兰牙": "蓝牙", "地藍": "蓝牙", "我太燃养": "蓝牙",
            "岛航": "导航", "到旁": "导航",
            "音了": "音乐", "大音乐": "音量",
            "音亮": "音量", "营亮": "音量", "一面": "音量",
            "温渡": "温度", "寶寄溫固": "温度",
            "天创": "天窗", "车方": "车窗", "车撞": "车窗",
            "后车厢": "后备箱", "后箱": "后备箱",
            "雨刷": "雨刮", "吐吐": "车灯", "車燈": "车灯",
            "一套車": "一首歌",
            "冰冰痛": "车门", "王毕端": "车门", "海拳闷": "车门", "皮疙瘩": "车门",
            "一吋燈": "车灯", "一吋灯": "车灯",
            "航道北京": "导航到北京", "航到北京": "导航到北京", "封鴨": "空调",
        }

        for wrong, correct in noun_corrections.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                print(f"   ✓ 名词纠错: '{wrong}' -> '{correct}'")

        # ==================== 5. 按词匹配（动词+名词组合） ====================
        verbs = [
            "打开", "关闭", "开启", "关掉", "合上", "收起", "展开", "升起", "降下",
            "调高", "调低", "增大", "减小", "增加", "减少", "提高", "降低", "调大", "调小",
            "播放", "暂停", "停止", "继续", "开始", "停一下",
            "导航", "查找", "搜索", "规划", "定位", "查询",
            "接听", "挂断", "拒绝", "拨号", "拨打",
            "拍照", "录像", "保存", "录制",
            "切换", "设置", "调节", "调整", "解除", "禁用", "启用", "激活",
            "上一", "下一", "调到", "转到", "切歌",
            "锁", "解锁", "上锁",
            "吹", "不要吹", "往上吹",
            "显示", "隐藏", "清除",
            "启动", "停止", "取消", "退出",
        ]

        nouns = [
            # 车窗天窗
            "车窗", "左前车窗", "右前车窗", "左后车窗", "右后车窗", "天窗", "遮阳板",
            # 车门后备箱
            "车门", "左前门", "右前门", "左后门", "右后门", "后备箱", "行李箱", "引擎盖", "前备箱",
            # 车灯
            "车灯", "近光灯", "远光灯", "雾灯", "示廓灯", "双闪", "日间行车灯", "氛围灯",
            "顶灯", "阅读灯", "车内灯",
            # 空调相关
            "空调", "温度", "制冷", "制热", "内循环", "外循环", "前除雾", "后除雾", "风量",
            "AC", "风扇", "挡风玻璃", "前挡", "后窗", "露营模式", "狗模式", "爱犬模式", "温度保持",
            # 座椅
            "座椅", "座椅加热", "座椅通风", "座椅按摩", "座椅位置",
            "方向盘", "方向盘加热器",
            # 后视镜
            "后视镜", "左后视镜", "右后视镜", "反光镜", "侧视镜",
            # 多媒体
            "音乐", "收音机", "电台", "蓝牙", "歌词", "音量",
            "抖音", "腾讯视频", "优酷视频", "浏览器",
            # 导航地图
            "导航", "地图", "路线", "路况", "目的地", "交通图", "卫星图", "卫星视野",
            "加油站", "充电站", "停车场", "地铁站", "医院",
            "北京", "上海", "广州", "深圳", "成都", "陆家嘴", "东方明珠",
            "特斯拉服务中心", "航海博物馆",
            # 电话
            "电话", "通讯录", "联系人", "信息",
            # 驾驶辅助
            "巡航", "车道保持", "自动泊车", "360影像", "悬架",
            # 车辆设置
            "驾驶模式", "能量回收", "ESP", "陡坡缓降", "胎压", "电量", "续航",
            "屏幕亮度", "手套箱", "杂物箱",
            # 雨刮
            "雨刮", "雨刷", "刮雨器",
            # 充电
            "充电口", "充电端口", "充电盖板",
            # 行车记录仪
            "行车记录仪", "行车记录", "短片", "哨兵模式",
            # 其他
            "车窗锁", "儿童锁", "顶灯",
        ]

        verbs = list(set(verbs))
        nouns = list(set(nouns))
        verbs.sort(key=len, reverse=True)
        nouns.sort(key=len, reverse=True)

        # 记录已经匹配成功的词组，拼音纠错时跳过它们
        matched_parts = []

        found_verb = None
        found_noun = None

        for v in verbs:
            if v in text:
                found_verb = v
                matched_parts.append(v)
                break

        for n in nouns:
            if n in text:
                found_noun = n
                matched_parts.append(n)
                break

        if found_verb and found_noun:
            if found_verb in found_noun:
                result = found_noun
            else:
                result = f"{found_verb}{found_noun}"
            print(f"   ✓ 按词匹配: '{found_verb}' + '{found_noun}' -> '{result}'")
            text = result

        # ==================== 6. 整条指令拼音纠错（跳过已匹配部分） ====================
        if self.config.enable_phonetic_correction and self.phonetic_corrector:
            # 获取完整的指令列表（动词+名词组合和完整指令）
            all_commands = list(set(verbs + nouns + self.valid_commands))
            all_commands.sort(key=len, reverse=True)

            # 检查是否已经匹配到完整指令
            matched_full_command = False

            # 先尝试精确匹配完整指令
            if text in self.valid_commands:
                matched_full_command = True
                print(f"   ✓ 已是完整指令: '{text}'")
            else:
                # 尝试最相似的整体匹配
                best_match, similarity = self.phonetic_corrector.find_best_match(
                    text, all_commands, threshold=0.5
                )

                if best_match and similarity >= 0.6:
                    print(f"   ✓ 整条指令拼音匹配: '{text}' -> '{best_match}' (相似度: {similarity:.3f})")
                    text = best_match
                    matched_full_command = True
                elif best_match and 0.5 <= similarity < 0.6:
                    print(f"   ⚠️ 整条指令拼音相似度过低: '{text}' -> '{best_match}' (相似度: {similarity:.3f})")
                    print(f"   → 保持原样: '{text}'")
                else:
                    print(f"   ✗ 未找到匹配的指令，保持原样: '{text}'")

            # 如果整条指令没有匹配成功，尝试对未匹配的部分进行拼音纠错
            if not matched_full_command:
                # 提取未匹配的词组
                words = re.findall(r'[\u4e00-\u9fa5]+', text)
                corrected_words = []

                for word in words:
                    # 检查这个词是否已经被匹配过
                    if word in matched_parts:
                        print(f"   ⏭️ 跳过已匹配词: '{word}'")
                        corrected_words.append(word)
                        continue

                    # 检查是否已经是正确的动词或名词
                    if word in verbs or word in nouns or word in self.valid_commands:
                        print(f"   ✓ 已是正确词: '{word}'")
                        corrected_words.append(word)
                        matched_parts.append(word)
                        continue

                    # 在动词库中找最相似的
                    best_match, sim = self.phonetic_corrector.find_best_match(word, verbs, threshold=0.5)
                    if best_match and sim >= 0.6:
                        print(f"   ✓ 拼音纠错(动词): '{word}' -> '{best_match}' (相似度: {sim:.3f})")
                        corrected_words.append(best_match)
                        continue

                    # 在名词库中找最相似的
                    best_match, sim = self.phonetic_corrector.find_best_match(word, nouns, threshold=0.5)
                    if best_match and sim >= 0.6:
                        print(f"   ✓ 拼音纠错(名词): '{word}' -> '{best_match}' (相似度: {sim:.3f})")
                        corrected_words.append(best_match)
                        continue

                    # 在完整指令库中找最相似的
                    best_match, sim = self.phonetic_corrector.find_best_match(word, self.valid_commands, threshold=0.5)
                    if best_match and sim >= 0.6:
                        print(f"   ✓ 拼音纠错(指令): '{word}' -> '{best_match}' (相似度: {sim:.3f})")
                        corrected_words.append(best_match)
                        continue

                    # 相似度过低或未找到，保持原样
                    if best_match and sim < 0.6:
                        print(f"   ⚠️ 拼音相似度过低: '{word}' -> '{best_match}' (相似度: {sim:.3f})，保持原样")
                    else:
                        print(f"   ✗ 未找到匹配: '{word}'，保持原样")
                    corrected_words.append(word)

                text = ''.join(corrected_words)

        # ==================== 7. 最终清理 ====================
        text = re.sub(r'[，,。！？；：""''《》【】（）]', '', text)

        print(f"   📝 最终结果: '{text}'")

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
        force_dict_match=True,
        commands_json_path="commands.json"
    )

    model = EnhancedASRRiskModel(config)

    # 测试动态纠错
    print("\n" + "=" * 70)
    print("🧪 测试动态纠错功能")
    print("=" * 70)
    model.add_correction("测试错误", "测试正确")
    print(f"动态纠错规则: {model.get_dynamic_corrections()}")

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
