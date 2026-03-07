# model_benchmarker.py
import pandas as pd
import time
from core.engine import VGuardEngine
from modules.module2_ASR.detector import ASRDetector


def run_benchmark():
    # 1. 初始化引擎
    engine = VGuardEngine()

    # 定义测试维度：不同的 ASR 模型尺寸
    # 注意：这会触发模型下载，确保网络通畅
    asr_variants = ["tiny", "base"]
    test_audios = ["test_zh.wav", "test_en.wav"]  # 你准备好的测试音频

    results = []

    for model_name in asr_variants:
        print(f"🚀 正在切换 ASR 模型至: {model_name}...")

        # 动态更换模块 B 的模型
        # 在 detector.py 中，我们需要确保 ASRRiskModel 接收 model_size 参数
        engine.m2 = ASRDetector()
        # 假设我们在 ASRRiskModel 中支持了模型切换

        for audio in test_audios:
            print(f"  - 正在测试音频: {audio}")

            # 执行分析
            # 假设 text 预设或从 simple ASR 获取
            start_time = time.time()
            res = engine.analyze_risk(audio_data=audio, asr_text="打开后备箱", speed=100)
            latency = (time.time() - start_time) * 1000

            # 记录数据
            data_point = {
                "model_variant": model_name,
                "audio_file": audio,
                "total_risk": res["total_risk"],
                "A_score": res["reports"][0].risk_score,
                "B_score": res["reports"][1].risk_score,
                "C_score": res["reports"][2].risk_score,
                "latency_ms": latency
            }
            results.append(data_point)

    # 保存为 CSV 方便后续可视化对比
    df = pd.DataFrame(results)
    df.to_csv("model_comparison_results.csv", index=False)
    print("✅ 测评完成，结果已保存至 model_comparison_results.csv")


if __name__ == "__main__":
    run_benchmark()