# run_confidence_distribution.py
"""
运行置信度分布实验 - 使用真实音频
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import librosa

from asr_engine import create_asr_engine, ASRResult
from confidence_analyzer import ConfidenceAnalyzer


class RealConfidenceDistribution:
    """使用真实音频的置信度分布分析"""

    def __init__(self):
        print("置信度分布实验分析（使用真实音频）")
        print("=" * 60)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建ASR引擎（使用tiny模型更快）
        self.engine = create_asr_engine(model_size="base", device="cpu")
        self.analyzer = ConfidenceAnalyzer(low_conf_threshold=0.5)

        self.data_points = []

    def collect_from_test_wav(self):
        """从test.wav收集数据"""
        print("\n1. 从test.wav收集数据...")

        try:
            # 加载音频
            audio, sr = librosa.load("test.wav", sr=16000, mono=True)
            print(f"   加载test.wav: {len(audio) / sr:.1f}秒")

            # 执行ASR
            result = self.engine.transcribe(audio, sample_rate=sr)

            if result.success:
                # 分析置信度
                metrics = self.analyzer.analyze(result)

                # 存储数据
                self.data_points.append({
                    "source": "test.wav",
                    "audio_type": "真实语音",
                    "text": result.text,
                    "confidence_score": metrics.confidence_score,
                    "mean_prob": metrics.mean_linear_prob,
                    "low_conf_ratio": metrics.low_conf_ratio,
                    "token_count": len(result.tokens)
                })

                print(f"   识别结果: '{result.text}'")
                print(f"   置信度: {metrics.confidence_score:.3f}")
                print(f"   平均概率: {metrics.mean_linear_prob:.3f}")
                print(f"   低置信token: {metrics.low_conf_ratio:.1%}")

        except Exception as e:
            print(f"   加载失败: {e}")

    def collect_from_recordings(self):
        """从录制的音频文件收集数据"""
        print("\n2. 从录音文件收集数据...")

        recording_files = [
            "test_recording.wav",
            "my_recording.wav",
            "test.wav"  # 重复但没关系
        ]

        for file in recording_files:
            try:
                audio, sr = librosa.load(file, sr=16000, mono=True)

                result = self.engine.transcribe(audio, sample_rate=sr)

                if result.success and result.text:
                    metrics = self.analyzer.analyze(result)

                    self.data_points.append({
                        "source": file,
                        "audio_type": "录音语音",
                        "text": result.text,
                        "confidence_score": metrics.confidence_score,
                        "mean_prob": metrics.mean_linear_prob,
                        "low_conf_ratio": metrics.low_conf_ratio,
                        "token_count": len(result.tokens)
                    })

                    print(f"   {file}: '{result.text[:20]}...' 置信度={metrics.confidence_score:.3f}")

            except:
                continue

    def create_distribution_plot(self):
        """创建置信度分布图"""
        if not self.data_points:
            print("无数据，无法绘图")
            return

        print("\n3. 生成置信度分布图...")

        # 提取数据
        conf_scores = [d["confidence_score"] for d in self.data_points]
        mean_probs = [d["mean_prob"] for d in self.data_points]
        low_conf_ratios = [d["low_conf_ratio"] for d in self.data_points]

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ASR置信度分布分析（真实语音）', fontsize=16, fontweight='bold')

        # 1. 置信度分数直方图
        axes[0, 0].hist(conf_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值(0.5)')
        axes[0, 0].set_xlabel('置信度分数')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('置信度分数分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 平均概率分布
        axes[0, 1].hist(mean_probs, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值(0.5)')
        axes[0, 1].set_xlabel('平均token概率')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('平均概率分布')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 低置信度token比例
        axes[1, 0].hist(low_conf_ratios, bins=10, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值(0.5)')
        axes[1, 0].set_xlabel('低置信度token比例')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('低置信度比例分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 置信度 vs 平均概率
        axes[1, 1].scatter(mean_probs, conf_scores, alpha=0.7, s=100, c='purple')
        axes[1, 1].set_xlabel('平均token概率')
        axes[1, 1].set_ylabel('置信度分数')
        axes[1, 1].set_title('置信度 vs 平均概率')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加对角线
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        axes[1, 1].legend()

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confidence_distribution_real_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')

        print(f"\n✓ 分布图已保存: {filename}")
        plt.show()

        return fig

    def generate_report(self):
        """生成分析报告"""
        print("\n4. 统计分析报告:")
        print("=" * 60)

        if not self.data_points:
            print("无数据")
            return

        conf_scores = [d["confidence_score"] for d in self.data_points]

        print(f"\n样本总数: {len(self.data_points)}")
        print(f"置信度范围: [{min(conf_scores):.3f}, {max(conf_scores):.3f}]")
        print(f"置信度均值: {np.mean(conf_scores):.3f}")
        print(f"置信度标准差: {np.std(conf_scores):.3f}")
        print(f"高置信度 (>0.7): {sum(1 for c in conf_scores if c > 0.7)} 个")
        print(f"中置信度 (0.3-0.7): {sum(1 for c in conf_scores if 0.3 <= c <= 0.7)} 个")
        print(f"低置信度 (<0.3): {sum(1 for c in conf_scores if c < 0.3)} 个")

        print("\n详细数据:")
        for i, d in enumerate(self.data_points, 1):
            print(f"  {i}. {d['source']}:")
            print(f"     文本: '{d['text'][:30]}{'...' if len(d['text']) > 30 else ''}'")
            print(f"     置信度: {d['confidence_score']:.3f}")
            print(f"     平均概率: {d['mean_prob']:.3f}")

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confidence_report_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "sample_count": len(self.data_points),
                "statistics": {
                    "mean_confidence": float(np.mean(conf_scores)),
                    "std_confidence": float(np.std(conf_scores)),
                    "min_confidence": float(min(conf_scores)),
                    "max_confidence": float(max(conf_scores))
                },
                "samples": self.data_points
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 报告已保存: {filename}")

    def cleanup(self):
        self.engine.cleanup()


def main():
    """主函数"""
    analyzer = RealConfidenceDistribution()

    try:
        # 1. 从test.wav收集
        analyzer.collect_from_test_wav()

        # 2. 从录音文件收集
        analyzer.collect_from_recordings()

        # 3. 生成分布图
        analyzer.create_distribution_plot()

        # 4. 生成报告
        analyzer.generate_report()

    finally:
        analyzer.cleanup()

    print("\n✅ 置信度分布分析完成！")


if __name__ == "__main__":
    main()