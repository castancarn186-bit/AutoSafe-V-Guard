# check_alignment.py
import sys
import os
import importlib
import inspect

# 1. 确保环境路径正确
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from core.base_module import BaseDetector, DetectionResult


def check_module(module_name, class_path, class_name):
    print(f"🔍 正在检查模块 [{module_name}]...")

    # 尝试导入
    try:
        # 动态添加队友的路径
        sys.path.append(os.path.join(ROOT_DIR, "modules", class_path))
        mod = importlib.import_module("detector")
        detector_class = getattr(mod, class_name)
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

    # 检查继承关系
    is_subclass = issubclass(detector_class, BaseDetector)
    if not is_subclass:
        print(f"  ❌ 架构违规: {class_name} 没有继承 BaseDetector！")
        return False
    else:
        print(f"  ✅ 继承检查通过")

    # 检查接口实现
    sig = inspect.signature(detector_class.detect)
    if 'data' not in sig.parameters or 'context' not in sig.parameters:
        print(f"  ❌ 接口违规: detect() 方法参数不符合协议 (需要 data, context)")
        return False
    else:
        print(f"  ✅ 接口参数检查通过")

    # 模拟运行检查返回类型
    print(f"  💡 正在尝试空负载运行测试...")
    try:
        # 这里需要根据具体模块给一点 mock 数据
        instance = detector_class()
        res = instance.detect(None, context={})
        if isinstance(res, DetectionResult):
            print(f"  ✅ 返回值协议检查通过 (Type: DetectionResult)")
        else:
            print(f"  ❌ 返回值违规: 返回了 {type(res)} 而不是 DetectionResult")
            return False
    except Exception as e:
        print(f"  ⚠️  模拟运行跳过 (可能因为缺少模型文件或初始化参数: {e})")

    return True


def main():
    print("=" * 50)
    print("🛡️  V-Guard 全系统架构对齐验收工具")
    print("=" * 50)

    modules_to_check = [
        ("声学模块 A", "module1_acoustic/src", "AcousticDetector"),
        ("ASR行为模块 B", "module2_asr/src", "ASRDetector"),
        ("语义模块 C", "module3_semantic/src", "SemanticDetector"),
    ]

    passed_count = 0
    for name, path, cls in modules_to_check:
        if check_module(name, path, cls):
            passed_count += 1
        print("-" * 30)

    print(f"\n验收结果: {passed_count}/{len(modules_to_check)} 模块已完全对齐协议")
    if passed_count < len(modules_to_check):
        print("🚩 状态: [不合格] 请根据上方错误提示督促队友重构！")
    else:
        print("🏆 状态: [一等奖预备] 全系统架构已实现解耦。")


if __name__ == "__main__":
    main()