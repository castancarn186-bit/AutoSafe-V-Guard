# evaluation_report.py
"""
修复版评估报告 - 无beta参数版本
"""

import numpy as np
import librosa
import re

# 导入正确的类
from asr_risk_model import EnhancedASRRiskModel, EnhancedConfig


class SimpleEvaluation:
    """简化版评估"""

    def __init__(self):
        print("ASR风险评估系统 - 简化评估\n" + "=" * 50)

        # 创建优化配置 - 速度优化版
        config = EnhancedConfig(
            model_size="tiny",  # 改为 tiny 提高速度
            enable_vad=False,  # 关闭 VAD 提高速度
            enable_adversarial_defense=True,
            defense_noise_std=0.0001,
            enable_postprocessing=True,
            language="zh"
        )

        # 使用 EnhancedASRRiskModel
        self.risk_model = EnhancedASRRiskModel(config=config)
        self.results = []

    def evaluate_audio(self, audio, sample_rate=16000, label=""):
        """评估音频"""
        print(f"\n评估: {label}")

        # compute_risk 返回字典
        result = self.risk_model.compute_risk(audio, sample_rate=sample_rate)

        self.results.append({
            "label": label,
            "risk_score": result['risk_score'],
            "confidence": result['confidence'],
            "text": result['text'],
            "risk_level": result['risk_level'],
            "decision": result['decision'],
            "latency_ms": result['timings']['total_ms'],
            "defense_ms": result['timings'].get('defense_ms', 0)
        })

        print(f"  识别文本: {result['text'][:30]}...")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  风险分数: {result['risk_score']:.3f} {result['risk_level']}")
        print(f"  决策: {result['decision']}")
        print(f"  防御耗时: {result['timings'].get('defense_ms', 0):.1f}ms")
        print(f"  总耗时: {result['timings']['total_ms']:.1f}ms")

        return result

    def run_all_tests(self):
        """运行所有测试"""
        print("\n[阶段1] 测试音频文件")

        try:
            audio, sr = librosa.load("test.wav", sr=16000, mono=True)
            print(f"  加载音频: {len(audio) / sr:.1f}秒")
            self.evaluate_audio(audio, label="test.wav")
        except FileNotFoundError:
            print("  ⚠️ 未找到test.wav")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")

        print("\n[阶段2] 生成报告")
        self.generate_report()

    def generate_report(self):
        """生成简单报告"""
        if not self.results:
            print("无评估结果")
            return

        print("\n" + "=" * 60)
        print("📊 评估报告总结")
        print("=" * 60)

        for result in self.results:
            print(f"\n📁 {result['label']}:")
            print(f"  文本: {result['text'][:50]}...")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  风险分数: {result['risk_score']:.3f}")
            print(f"  风险等级: {result['risk_level']}")
            print(f"  决策: {result['decision']}")
            print(f"  防御耗时: {result['defense_ms']:.1f}ms")
            print(f"  总耗时: {result['latency_ms']:.1f}ms")

        risk_scores = [r["risk_score"] for r in self.results]
        conf_scores = [r["confidence"] for r in self.results]
        latencies = [r["latency_ms"] for r in self.results]

        if risk_scores:
            print(f"\n📈 总体统计:")
            print(f"  平均风险分数: {np.mean(risk_scores):.3f}")
            print(f"  平均置信度: {np.mean(conf_scores):.3f}")
            print(f"  平均耗时: {np.mean(latencies):.1f}ms")
            realtime_status = '✅ 达标' if np.mean(latencies) < 500 else '❌ 超标'
            print(f"  实时性: {realtime_status}")

    def cleanup(self):
        self.risk_model.cleanup()
        print("\n✅ 资源清理完成")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 ASR风险评估系统 - 对抗性防御验证")
    print("=" * 60)

    evaluator = SimpleEvaluation()

    try:
        evaluator.run_all_tests()

        print("\n" + "=" * 60)
        print("✅ 系统验证完成！")
        print("=" * 60)

        if evaluator.results:
            result = evaluator.results[0]
            print(f"\n📊 当前系统状态:")
            print(f"  • 模型: tiny")
            print(f"  • VAD: 关闭")
            print(f"  • 对抗性防御: 启用")
            print(f"  • 置信度: {result['confidence']:.3f}")
            print(f"  • 实时性: {result['latency_ms']:.1f}ms")

            if result['latency_ms'] < 500:
                print(f"\n✅ 实时性达标！")
            else:
                print(f"\n⚠️ 实时性超标，建议使用 tiny 模型并关闭 VAD")

    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
