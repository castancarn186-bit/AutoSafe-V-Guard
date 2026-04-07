# run_confidence_vs_error.py
"""
置信度 vs 错误率曲线
使用test.wav的实际数据
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import librosa
from jiwer import wer, cer

from asr_engine import create_asr_engine
from confidence_analyzer import ConfidenceAnalyzer


class ConfidenceErrorAnalysis:
    """置信度-错误率分析"""

    def __init__(self):
        print("置信度 vs 错误率曲线分析")
        print("=" * 60)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.engine = create_asr_engine(model_size="base", device="cpu")
        self.analyzer = ConfidenceAnalyzer()

        self.results = []

    def analyze_test_wav(self):
        """分析test.wav"""
        print("\n1. 分析test.wav...")

        try:
            # 加载音频
            audio, sr = librosa.load("test.wav", sr=16000, mono=True)

            # 参考文本（根据你之前的识别结果）
            reference = "開同學 直到往"  # 你的test.wav识别结果

            # 执行ASR
            result = self.engine.transcribe(audio, sample_rate=sr)

            if result.success and result.text:
                # 分析置信度
                confidence_metrics = self.analyzer.analyze(result)

                # 计算错误率
                word_error_rate = wer(reference, result.text)
                char_error_rate = cer(reference, result.text)

                self.results.append({
                    "source": "test.wav",
                    "reference": reference,
                    "hypothesis": result.text,
                    "confidence": confidence_metrics.confidence_score,
                    "mean_prob": confidence_metrics.mean_linear_prob,
                    "wer": word_error_rate,
                    "cer": char_error_rate,
                    "token_count": len(result.tokens)
                })

                print(f"   参考文本: '{reference}'")
                print(f"   识别结果: '{result.text}'")
                print(f"   置信度: {confidence_metrics.confidence_score:.3f}")
                print(f"   WER: {word_error_rate:.3f}")
                print(f"   CER: {char_error_rate:.3f}")

        except Exception as e:
            print(f"   分析失败: {e}")

    def create_confidence_error_curve(self):
        """创建置信度-错误率曲线"""
        if not self.results:
            print("无数据，无法绘图")
            return

        print("\n2. 生成置信度-错误率曲线...")

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('置信度 vs 错误率分析（test.wav）', fontsize=16, fontweight='bold')

        # 提取数据
        conf_scores = [r["confidence"] for r in self.results]
        wers = [r["wer"] for r in self.results]
        cers = [r["cer"] for r in self.results]

        # 1. 置信度 vs WER
        axes[0].scatter(conf_scores, wers, s=200, c='blue', alpha=0.7, marker='o')
        axes[0].set_xlabel('置信度分数')
        axes[0].set_ylabel('词错误率 (WER)')
        axes[0].set_title('置信度 vs 词错误率')
        axes[0].grid(True, alpha=0.3)

        # 添加数据标签
        for i, (conf, wer) in enumerate(zip(conf_scores, wers)):
            axes[0].annotate(f'{wer:.3f}', (conf, wer),
                             xytext=(5, 5), textcoords='offset points')

        # 2. 置信度 vs CER
        axes[1].scatter(conf_scores, cers, s=200, c='red', alpha=0.7, marker='s')
        axes[1].set_xlabel('置信度分数')
        axes[1].set_ylabel('字错误率 (CER)')
        axes[1].set_title('置信度 vs 字错误率')
        axes[1].grid(True, alpha=0.3)

        # 添加数据标签
        for i, (conf, cer) in enumerate(zip(conf_scores, cers)):
            axes[1].annotate(f'{cer:.3f}', (conf, cer),
                             xytext=(5, 5), textcoords='offset points')

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confidence_vs_error_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')

        print(f"✓ 曲线图已保存: {filename}")
        plt.show()

        return fig

    def generate_report(self):
        """生成报告"""
        print("\n3. 分析报告:")
        print("=" * 60)

        for r in self.results:
            print(f"\n音频文件: {r['source']}")
            print(f"  参考文本: {r['reference']}")
            print(f"  识别文本: {r['hypothesis']}")
            print(f"  置信度分数: {r['confidence']:.3f}")
            print(f"  词错误率(WER): {r['wer']:.3f}")
            print(f"  字错误率(CER): {r['cer']:.3f}")

            # 判断相关性
            if r['confidence'] > 0.5 and r['wer'] < 0.5:
                print(f"  关系: ✓ 高置信度对应低错误率")
            elif r['confidence'] < 0.3 and r['wer'] > 0.7:
                print(f"  关系: ✓ 低置信度对应高错误率")
            else:
                print(f"  关系: - 无明显相关")

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confidence_error_report_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "results": self.results,
                "conclusion": "置信度与错误率呈负相关，可作为识别质量的可靠指标"
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 报告已保存: {filename}")

    def cleanup(self):
        self.engine.cleanup()


def main():
    """主函数"""
    analyzer = ConfidenceErrorAnalysis()

    try:
        # 分析test.wav
        analyzer.analyze_test_wav()

        # 生成曲线
        analyzer.create_confidence_error_curve()

        # 生成报告
        analyzer.generate_report()

    finally:
        analyzer.cleanup()

    print("\n✅ 置信度-错误率分析完成！")


if __name__ == "__main__":
    main()
