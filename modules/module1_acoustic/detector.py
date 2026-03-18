"""
modules/module_a_acoustic/detector.py
声学物理层检测器（基于 AASIST 模型）
继承 core.base_module.BaseDetector，实现 setup() 和 detect(ctx)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from core.base_module import BaseDetector
from core.protocol import SystemContext, DetectionResult

# 从同目录的 aasist_model.py 导入 Model 类并重命名为 AASIST
from modules.module1_acoustic.aasist_model import Model as AASIST


class AcousticDetector(BaseDetector):
    """
    使用 AASIST 预训练模型检测声学欺骗攻击（AI合成语音、重放等）。
    """

    def __init__(self, module_id: str = 'A', config=None):
        # 调用父类初始化，module_id 必须提供，这里给默认值 'A' 方便创建
        super().__init__(module_id, config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.mel_transform = None
        # 现在 self.config 已经由父类设置，可以直接使用
        self.expected_length = self.config.get('expected_length', 64600)
        self.thresholds = self.config.get('thresholds', {'PASS': 0.3, 'CONFIRM': 0.6})

    def setup(self):
        """加载 AASIST 预训练模型"""
        # [架构调整] 移除 Mel-Spectrogram 转换器！
        # AASIST 内置了 SincNet 前端，直接处理 Raw Waveform，大大节省了车载平台的 CPU 算力消耗。

        # 2. 定义模型配置（使用官方 AASIST 参数）
        d_args = {
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.5, 0.5, 0.5],
            "temperatures": [2, 2, 100],
            "first_conv": 128,
        }

        # 3. 实例化模型
        self.model = AASIST(d_args)

        # 4. 加载预训练权重
        model_path = self.config.get(
            'model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'aasist.pth')
        )
        if not os.path.exists(model_path):
            self.logger.error(f"AASIST model file not found: {model_path}")
            raise FileNotFoundError(f"AASIST model missing: {model_path}")

        state_dict = torch.load(model_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded. First conv layer weights (first 5 values):")
        print(self.model.conv_time.band_pass[0, :5])  # 打印 sinc 卷积的前几个值
        print("Model parameter count:", sum(p.numel() for p in self.model.parameters()))
        self.logger.info(f"AASIST model loaded from {model_path} to {self.device}")

    def _preprocess(self, audio):
        """
        [安全网关标准预处理] 循环拼接与物理量纲还原
        """
        # 1. 物理量纲还原：
        # 车载麦克风通常采集为 int16（范围 -32768~32767）
        # AASIST 需要 [-1.0, 1.0] 范围的浮点数。固定除以 32768.0 能保留真实的音量动态，
        # 绝对不能用 `audio / max`，那样会把背景环境底噪放大 100 倍！
        audio = np.array(audio, dtype=np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        # 2. 长度对齐：拒绝补零，使用波形循环拼接 (Tile)
        audio_len = len(audio)
        if audio_len < self.expected_length:
            # 循环重复填充，保持声学连贯性
            num_repeats = int(np.ceil(self.expected_length / audio_len))
            audio = np.tile(audio, num_repeats)[:self.expected_length]
        else:
            # 超长则截断
            audio = audio[:self.expected_length]

        # 3. 转为 tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)

        return audio_tensor


    def detect(self, ctx: SystemContext) -> DetectionResult:
        audio = ctx.audio_frame
        if audio is None or len(audio) == 0:
            return DetectionResult(
                module_id='A',
                risk_score=0.0,
                decision="PASS",
                reason="No audio input",
                metadata={}
            )
            # 1. 预处理为 Log-Mel 谱图
        input_tensor = self._preprocess(audio)  # (1,1,80,T)

            # 2. 模型推理
        with torch.no_grad():
                # AASIST 的 forward 返回 (last_hidden, output)，我们只需要 output
            _, output = self.model(input_tensor)  # output shape: (1, 2)
            probs = F.softmax(output, dim=-1)
            # 【架构级修正】: Class 0 = Spoof, Class 1 = Bonafide (真人)
            bonafide_prob = probs[0, 1].item()
            spoof_prob = probs[0, 0].item()

                # 风险分应该等于 spoof (伪造) 的概率
            risk_score = spoof_prob
                # spoof 概率

            # 3. 生成建议
        suggestion = self._get_suggestion(risk_score)

            # 4. 证据
        evidence = {
                "spoof_prob": risk_score,
                "bonafide_prob": probs[0, 1].item(),
                "input_length": len(audio),
                "spectrogram_shape": list(input_tensor.shape)
        }

        return DetectionResult(
                module_id='A',
                risk_score=risk_score,
                decision=suggestion,
                reason="Acoustic spoofing detection with AASIST",
                metadata=evidence
        )

    '''
        try:
            # 1. 预处理为 Log-Mel 谱图
            input_tensor = self._preprocess(audio)  # (1,1,80,T)

            # 2. 模型推理
            with torch.no_grad():
                # AASIST 的 forward 返回 (last_hidden, output)，我们只需要 output
                _, output = self.model(input_tensor)  # output shape: (1, 2)
                probs = F.softmax(output, dim=-1)
                # 【架构级修正】: Class 0 = Spoof, Class 1 = Bonafide (真人)
                bonafide_prob = probs[0, 1].item()
                spoof_prob = probs[0, 0].item()

                # 风险分应该等于 spoof (伪造) 的概率
                risk_score = spoof_prob
                # spoof 概率

            # 3. 生成建议
            suggestion = self._get_suggestion(risk_score)

            # 4. 证据
            evidence = {
                "spoof_prob": risk_score,
                "bonafide_prob": probs[0, 0].item(),
                "input_length": len(audio),
                "spectrogram_shape": list(input_tensor.shape)
            }

            return DetectionResult(
                module_id='A',
                risk_score=risk_score,
                decision=suggestion,
                reason="Acoustic spoofing detection with AASIST",
                metadata=evidence
            )
        '''
    '''
        except Exception as e:
            self.logger.exception("Error in acoustic detection")
            return DetectionResult(
                module_id='A',
                risk_score=0.5,
                decision="PASS",
                reason=f"Acoustic detection error: {str(e)}",
                metadata={}
            )
        '''

    def _get_suggestion(self, risk_score):
        if risk_score < self.thresholds['PASS']:
            return "PASS"
        elif risk_score < self.thresholds['CONFIRM']:
            return "CONFIRM"
        else:
            return "REJECT"
