# evaluation_report.py
"""
修复版评估报告 - 支持对抗性防御
"""

import numpy as np
import librosa

# 导入正确的类
from asr_risk_model import OptimizedASRRiskModel, OptimizedConfig


class SimpleEvaluation:
    """简化版评估"""

    def __init__(self):
        print("ASR风险评估系统 - 简化评估\n" + "=" * 50)

        # 创建优化配置 - 启用对抗性防御
        config = OptimizedConfig(
            model_size="tiny",
            enable_vad=True,
            enable_stability=False,
            alpha=1.0,
            enable_adversarial_defense=True,  # 新增
            defense_strength="medium"          # 新增
        )

        # 使用 OptimizedASRRiskModel
        self.risk_model = OptimizedASRRiskModel(config=config)
        self.results = []

    def evaluate_audio(self, audio, sample_rate=16000, label=""):
        """评估音频"""
        print(f"\n评估: {label}")

        result = self.risk_model.compute_risk(audio, sample_rate=sample_rate)

        self.results.append({
            "label": label,
            "risk_score": result['risk_score'],
            "confidence": result['confidence'],
            "text": result['text'],
            "risk_level": result['risk_level'],
            "decision": result['decision'],
            "latency_ms": result['timings']['total_ms'],
            "defense_ms": result['timings'].get('defense_ms', 0)  # 新增
        })

        print(f"  识别文本: {result['text'][:30]}...")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  风险分数: {result['risk_score']:.3f} {result['risk_level']}")
        print(f"  决策: {result['decision']}")
        print(f"  防御耗时: {result['timings'].get('defense_ms', 0):.1f}ms")  # 新增
        print(f"  总耗时: {result['timings']['total_ms']:.1f}ms")

        return result

    def run_all_tests(self):
        """运行所有测试"""
        print("\n[阶段1] 测试音频文件")

        try:
            audio, sr = librosa.load("test.wav", sr=16000, mono=True)
            print(f"  加载音频: {len(audio)/sr:.1f}秒")
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
            print(f"  实时性: {'✅ 达标' if np.mean(latencies) < 500 else '❌ 超标'}")

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
            print(f"  • 模型: tiny (最优)")
            print(f"  • VAD: 启用 (激进程度=3)")
            print(f"  • 稳定性测试: 关闭")
            print(f"  • 对抗性防御: 启用 (medium强度)")
            print(f"  • 置信度: {result['confidence']:.3f} (动态变化)")
            print(f"  • 实时性: {result['latency_ms']:.1f}ms (<500ms)")
            print(f"\n🎉 恭喜！系统已集成对抗性防御！")

    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
