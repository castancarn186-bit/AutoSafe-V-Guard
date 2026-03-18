import csv
import json
import time
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.module3_semantic.detector import SemanticDetector
from core.protocol import SystemContext  # 新增导入

def load_test_cases_from_csv(csv_path):
    """从 CSV 加载测试用例，返回列表"""
    cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            context = {
                'speed': float(row['speed']),
                'gear': row['gear'],
                'weather': row['weather'],
                'has_pedestrians': row['has_pedestrians'].lower() == 'true',
            }
            cases.append({
                'text': row['text'],
                'context': context,
                'expected_decision': row['expected_decision']
            })
    return cases

def load_test_cases_from_jsonl(jsonl_path, score_thresholds=(0.3, 0.7)):
    """从合成数据集加载，将分数映射为决策"""
    cases = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            score = item['ground_truth_score']
            if score < score_thresholds[0]:
                expected = 'PASS'
            elif score < score_thresholds[1]:
                expected = 'REVIEW'
            else:
                expected = 'BLOCK'
            cases.append({
                'text': item['text'],
                'context': item['context'],
                'expected_decision': expected,
                'expected_score': score
            })
    return cases

def test_module(detector, cases):
    """批量运行测试，统计结果"""
    total = len(cases)
    correct = 0
    results = []
    latencies = []

    for i, case in enumerate(cases):
        text = case['text']
        context = case['context']
        expected = case['expected_decision']

        # 构造 SystemContext 对象
        ctx = SystemContext(
            audio_frame=None,                          # 必须提供，设为 None
            asr_text=text,                              # ASR 文本
            speed=context.get('speed', 0.0),             # 车速
            weather=context.get('weather', 'sunny'),     # 天气
            has_pedestrians=context.get('has_pedestrians', False),  # 是否有行人
            is_night=False                               # 默认非夜间
        )

        start = time.perf_counter()
        try:
            # 重要：调用 run 方法，而不是直接调用 detect
            result = detector.run(ctx)
        except Exception as e:
            print(f"用例 {i} 崩溃: {e}")
            result = None
        elapsed = (time.perf_counter() - start) * 1000

        if result:
            actual = result.decision
            is_correct = (actual == expected)
            if is_correct:
                correct += 1
            latencies.append(elapsed)

            results.append({
                'index': i,
                'text': text,
                'context': context,
                'expected': expected,
                'actual': actual,
                'score': result.risk_score,
                'reason': result.reason,
                'latency_ms': elapsed,
                'correct': is_correct
            })
        else:
            results.append({
                'index': i,
                'text': text,
                'expected': expected,
                'actual': 'ERROR',
                'correct': False
            })

        if (i+1) % 10 == 0:
            print(f"已处理 {i+1}/{total}")

    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return results, accuracy, avg_latency

def analyze_results(results):
    """简单分析：输出准确率、混淆矩阵、错误案例"""
    decisions = ['PASS', 'REVIEW', 'BLOCK']
    cm = {exp: {act: 0 for act in decisions} for exp in decisions}
    errors = []

    for r in results:
        if r['correct']:
            cm[r['expected']][r['actual']] += 1
        else:
            if r['actual'] != 'ERROR':
                cm[r['expected']][r['actual']] += 1
            errors.append(r)

    print("\n========== 测试结果统计 ==========")
    print(f"总用例数: {len(results)}")
    print(f"准确率: {sum(r['correct'] for r in results)/len(results):.2%}")
    if 'latency_ms' in results[0]:
        avg_lat = sum(r['latency_ms'] for r in results if 'latency_ms' in r) / len(results)
        print(f"平均延迟: {avg_lat:.2f} ms")

    print("\n混淆矩阵 (预期 vs 实际):")
    print("预期\\实际\t" + "\t".join(decisions))
    for exp in decisions:
        row = [str(cm[exp][act]) for act in decisions]
        print(f"{exp}\t\t" + "\t".join(row))

    if errors:
        print("\n错误案例（前10个）:")
        for e in errors[:10]:
            print(f"  {e['index']}: 预期={e['expected']}, 实际={e['actual']}, 文本=\"{e['text']}\"")
    else:
        print("\n🎉 所有用例通过！")

if __name__ == "__main__":
    print("正在初始化 SemanticDetector...")
    detector = SemanticDetector()
    from core.base_module import BaseDetector

    print("BaseDetector 来源模块:", BaseDetector.__module__)
    print("BaseDetector 所在文件:", sys.modules[BaseDetector.__module__].__file__)
    print("BaseDetector 是否有 run 方法:", hasattr(BaseDetector, 'run'))
    print("SemanticDetector 是否有 run 方法:", hasattr(detector, 'run'))
    csv_path = PROJECT_ROOT / "test_cases.csv"
    if csv_path.exists():
        cases = load_test_cases_from_csv(csv_path)
    else:
        jsonl_path = PROJECT_ROOT / "semantic_safety_test.jsonl"
        if jsonl_path.exists():
            cases = load_test_cases_from_jsonl(jsonl_path)
        else:
            print("找不到测试数据，请先创建测试用例。")
            sys.exit(1)

    print(f"加载了 {len(cases)} 条测试用例。")

    results, accuracy, avg_latency = test_module(detector, cases)

    analyze_results(results)

    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n详细结果已保存到 test_results.json")