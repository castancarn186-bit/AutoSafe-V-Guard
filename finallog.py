# finallog.py
import os


def dump_project():
    # 1. 锁定当前运行位置
    root_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"--- [V-Guard Sync Tool] ---")
    print(f"[Step 1] 当前定位目录: {root_dir}")

    # 2. 检查关键文件夹是否存在
    target_dirs = ['core', 'ui', 'modules', 'hardware', 'data']
    found_any = False
    for d in target_dirs:
        if os.path.exists(os.path.join(root_dir, d)):
            found_any = True
            print(f"[Check] 找到目录: {d} √")
        else:
            print(f"[Check] 未找到目录: {d} x")

    if not found_any:
        print("\n[Error] 找不到核心文件夹！请确保此脚本放在 AutoSafe-V-Guard 根目录下（和 ctest_3.py 同级）。")
        return

    # 3. 开始扫描
    summary = []
    file_count = 0

    for d in target_dirs:
        dir_path = os.path.join(root_dir, d)
        if not os.path.exists(dir_path): continue

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                # 排除编译缓存和无关文件
                if file.endswith(('.py', '.json')) and '__pycache__' not in root:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_dir)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            summary.append(f"\n{'=' * 30}\nFILE: {rel_path}\n{'=' * 30}\n{content}")
                            file_count += 1
                    except Exception as e:
                        print(f"[Skip] 无法读取 {rel_path}: {e}")

    # 4. 强制写入根目录
    output_path = os.path.join(root_dir, "project_snapshot.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"V-GUARD PROJECT FULL SNAPSHOT\nCOUNT: {file_count} files\n")
        f.write("\n".join(summary))

    print(f"\n[Success] 快照生成成功！共捕捉到 {file_count} 个文件。")
    print(f"[File] 路径: {output_path}")


if __name__ == "__main__":
    dump_project()