import json
import csv
from pathlib import Path

def extract_block_errors(json_path, output_csv):
    """
    从 test_results.json 中提取预期 BLOCK 但实际为 REVIEW 或 PASS 的案例，
    并保存为 CSV 文件。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 过滤条件
    error_cases = [
        case for case in results
        if case.get('expected') == 'BLOCK' and case.get('actual') in ('REVIEW', 'PASS')
    ]

    print(f"找到 {len(error_cases)} 条预期 BLOCK 但实际未拦截的案例。")

    if not error_cases:
        print("没有符合条件的案例。")
        return

    # 准备 CSV 字段：除了常规字段，还将 context 中的子字段展开
    # 先收集所有可能的 context 键名（确保每一行都有这些列）
    context_keys = set()
    for case in error_cases:
        ctx = case.get('context', {})
        if isinstance(ctx, dict):
            context_keys.update(ctx.keys())

    # 基础字段
    base_fields = ['index', 'text', 'expected', 'actual', 'score', 'reason', 'latency_ms']
    # 加上 context 字段
    context_fields = sorted(context_keys)  # 排序保持一致性
    fieldnames = base_fields + context_fields

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in error_cases:
            row = {}
            # 基础字段
            for field in base_fields:
                row[field] = case.get(field, '')
            # context 字段
            ctx = case.get('context', {})
            if isinstance(ctx, dict):
                for key in context_fields:
                    row[key] = ctx.get(key, '')
            else:
                # 如果 context 不是字典（比如 None 或字符串），则留空
                for key in context_fields:
                    row[key] = ''

            writer.writerow(row)

    print(f"已保存到: {output_csv}")
    print("你可以用 Excel 或任何表格软件打开分析。")

if __name__ == "__main__":
    # 默认路径：当前目录下的 test_results.json
    json_path = Path(__file__).parent / "test_results.json"
    output_csv = Path(__file__).parent / "C_block_errors.csv"

    if not json_path.exists():
        print(f"错误：找不到 {json_path}，请确保文件存在。")
    else:
        extract_block_errors(json_path, output_csv)