# embedded_performance_test.py
"""
嵌入式运行测试报告
测试ASR系统在嵌入式平台（如树莓派）上的性能
"""

import numpy as np
import time
import json
import psutil
import platform
from datetime import datetime
from asr_engine import create_asr_engine, create_test_audio
from asr_risk_model import ASRRiskModel, ASRRiskConfig


class EmbeddedPerformanceTest:
    """嵌入式性能测试"""

    def __init__(self):
        print("嵌入式性能测试报告")
        print("=" * 60)

        # 系统信息
        self.system_info = self.get_system_info()
        self.print_system_info()

        # 测试结果存储
        self.test_results = []

    def get_system_info(self):
        """获取系统信息"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 ** 3,
            "test_time": datetime.now().isoformat()
        }

    def print_system_info(self):
        """打印系统信息"""
        print("系统信息:")
        print(f"  平台: {self.system_info['platform']}")
        print(f"  处理器: {self.system_info['processor']}")
        print(f"  Python版本: {self.system_info['python_version']}")
        print(f"  CPU核心数: {self.system_info['cpu_count']}")
        print(f"  内存: {self.system_info['memory_total_gb']:.1f} GB")
        print(f"  测试时间: {self.system_info['test_time']}")

    def test_memory_usage(self):
        """测试内存使用"""
        print("\n1. 内存使用测试:")

        # 测量初始内存
        initial_memory = psutil.Process().memory_info().rss / 1024 ** 2  # MB

        # 创建模型
        engine = create_asr_engine(model_size="tiny")

        # 测量加载后内存
        after_load_memory = psutil.Process().memory_info().rss / 1024 ** 2

        # 测试推理内存
        audio = create_test_audio(duration=3.0, audio_type="speech_like")
        result = engine.transcribe(audio, sample_rate=16000)

        after_inference_memory = psutil.Process().memory_info().rss / 1024 ** 2

        # 清理
        engine.cleanup()

        memory_results = {
            "initial_memory_mb": round(initial_memory, 2),
            "after_load_memory_mb": round(after_load_memory, 2),
            "after_inference_memory_mb": round(after_inference_memory, 2),
            "model_load_increase_mb": round(after_load_memory - initial_memory, 2),
            "inference_increase_mb": round(after_inference_memory - after_load_memory, 2)
        }

        print(f"  初始内存: {memory_results['initial_memory_mb']} MB")
        print(f"  模型加载后: {memory_results['after_load_memory_mb']} MB")
        print(f"  推理后: {memory_results['after_inference_memory_mb']} MB")
        print(f"  模型加载增加: {memory_results['model_load_increase_mb']} MB")
        print(f"  推理增加: {memory_results['inference_increase_mb']} MB")

        return memory_results

    def test_latency(self, num_tests=10):
        """测试延迟"""
        print(f"\n2. 延迟测试 (运行{num_tests}次):")

        # 创建模型
        engine = create_asr_engine(model_size="tiny")
        risk_model = ASRRiskModel(asr_engine=engine)

        latencies = []
        confidence_scores = []
        stability_scores = []

        for i in range(num_tests):
            # 生成测试音频
            audio = create_test_audio(
                duration=2.0 + np.random.uniform(-0.5, 0.5),  # 1.5-2.5秒随机长度
                audio_type="speech_like"
            )

            # 测量总延迟
            start_time = time.time()

            # 执行完整风险评估
            metrics = risk_model.compute_risk(audio, sample_rate=16000)

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 转换为毫秒

            latencies.append(latency)
            confidence_scores.append(metrics.confidence_score)
            stability_scores.append(metrics.stability_score)

            if (i + 1) % 5 == 0:
                print(f"  第{i + 1}次: {latency:.1f}ms")

        # 统计
        latency_results = {
            "test_count": num_tests,
            "avg_latency_ms": round(np.mean(latencies), 1),
            "min_latency_ms": round(np.min(latencies), 1),
            "max_latency_ms": round(np.max(latencies), 1),
            "std_latency_ms": round(np.std(latencies), 1),
            "avg_confidence": round(np.mean(confidence_scores), 3),
            "avg_stability": round(np.mean(stability_scores), 3)
        }

        print(f"\n  延迟统计:")
        print(f"    平均延迟: {latency_results['avg_latency_ms']}ms")
        print(f"    最小延迟: {latency_results['min_latency_ms']}ms")
        print(f"    最大延迟: {latency_results['max_latency_ms']}ms")
        print(f"    标准差: {latency_results['std_latency_ms']}ms")
        print(f"    平均置信度: {latency_results['avg_confidence']}")
        print(f"    平均稳定性: {latency_results['avg_stability']}")

        # 检查是否满足实时性要求（项目要求 <500ms）
        real_time_ok = latency_results['avg_latency_ms'] < 500
        print(f"\n  实时性检查: {'✓ 通过 (<500ms)' if real_time_ok else '✗ 未通过'}")

        # 清理
        risk_model.cleanup()

        return latency_results

    def test_different_audio_lengths(self):
        """测试不同音频长度的延迟"""
        print("\n3. 不同音频长度延迟测试:")

        audio_lengths = [1.0, 2.0, 3.0, 5.0, 10.0]  # 秒
        results = []

        engine = create_asr_engine(model_size="tiny")
        risk_model = ASRRiskModel(asr_engine=engine)

        for length in audio_lengths:
            audio = create_test_audio(duration=length, audio_type="speech_like")

            # 测量延迟
            start_time = time.time()
            metrics = risk_model.compute_risk(audio, sample_rate=16000)
            end_time = time.time()

            latency = (end_time - start_time) * 1000

            results.append({
                "audio_length_sec": length,
                "latency_ms": round(latency, 1),
                "rtf": round(latency / (length * 1000), 3),  # 实时因子
                "confidence": round(metrics.confidence_score, 3),
                "stability": round(metrics.stability_score, 3)
            })

            print(f"  {length}s音频: {latency:.1f}ms (RTF: {latency / (length * 1000):.3f})")

        risk_model.cleanup()
        return results

    def test_different_models(self):
        """测试不同模型大小的性能"""
        print("\n4. 不同模型性能测试:")

        models = [
            ("tiny", "39M参数"),
            ("base", "74M参数"),
            ("small", "244M参数")
        ]

        results = []

        for model_size, description in models:
            try:
                print(f"\n  测试模型: {model_size} ({description})")

                # 创建模型
                engine = create_asr_engine(model_size=model_size)

                # 测试内存
                memory_before = psutil.Process().memory_info().rss / 1024 ** 2

                # 测试延迟
                audio = create_test_audio(duration=2.0, audio_type="speech_like")

                start_time = time.time()
                result = engine.transcribe(audio, sample_rate=16000)
                end_time = time.time()

                memory_after = psutil.Process().memory_info().rss / 1024 ** 2
                latency = (end_time - start_time) * 1000

                model_result = {
                    "model_size": model_size,
                    "description": description,
                    "memory_usage_mb": round(memory_after - memory_before, 1),
                    "latency_ms": round(latency, 1),
                    "transcription": result.text[:20] + "..." if result.text else "无识别",
                    "success": result.success
                }

                results.append(model_result)

                print(f"    内存使用: {model_result['memory_usage_mb']} MB")
                print(f"    延迟: {model_result['latency_ms']}ms")
                print(f"    识别结果: '{model_result['transcription']}'")

                engine.cleanup()

            except Exception as e:
                print(f"    模型 {model_size} 测试失败: {e}")
                results.append({
                    "model_size": model_size,
                    "error": str(e)
                })

        return results

    def generate_report(self, all_results):
        """生成完整测试报告"""
        print("\n" + "=" * 60)
        print("嵌入式运行测试报告 - 总结")
        print("=" * 60)

        report = {
            "system_info": self.system_info,
            "test_results": all_results,
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(all_results),
                "recommendation": ""
            }
        }

        # 分析结果
        if 'latency' in all_results:
            latency = all_results['latency']
            if latency['avg_latency_ms'] < 500:
                report['summary']['real_time_performance'] = "合格 (<500ms)"
                report['summary']['recommendation'] += "实时性满足要求。"
            else:
                report['summary']['real_time_performance'] = "不合格 (≥500ms)"
                report['summary']['recommendation'] += "建议优化模型或使用更小模型。"

        if 'models' in all_results:
            best_model = min(
                [m for m in all_results['models'] if 'latency_ms' in m],
                key=lambda x: x['latency_ms']
            )
            report['summary']['recommended_model'] = best_model['model_size']
            report['summary']['recommendation'] += f" 推荐使用 {best_model['model_size']} 模型。"

        # 打印总结
        print("\n测试总结:")
        for key, value in report['summary'].items():
            if key != 'recommendation':
                print(f"  {key}: {value}")

        print(f"\n建议: {report['summary']['recommendation']}")

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embedded_performance_report_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n完整报告已保存: {filename}")

        return report


def main():
    """主测试函数"""
    print("开始嵌入式性能测试...")

    tester = EmbeddedPerformanceTest()

    # 运行所有测试
    all_results = {}

    # 1. 内存测试
    memory_results = tester.test_memory_usage()
    all_results['memory'] = memory_results

    # 2. 延迟测试
    latency_results = tester.test_latency(num_tests=5)
    all_results['latency'] = latency_results

    # 3. 不同音频长度测试
    length_results = tester.test_different_audio_lengths()
    all_results['audio_lengths'] = length_results

    # 4. 不同模型测试
    model_results = tester.test_different_models()
    all_results['models'] = model_results

    # 生成报告
    report = tester.generate_report(all_results)

    print("\n" + "=" * 60)
    print("✅ 嵌入式运行测试完成！")


if __name__ == "__main__":
    # 检查依赖
    try:
        import psutil

        main()
    except ImportError:
        print("需要安装psutil: pip install psutil")