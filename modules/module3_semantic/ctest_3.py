import os
import sys
import logging

# ==========================================================
# 1. 离线环境配置 (Strict Offline Mode)
# ==========================================================
# 强制屏蔽网络连接，确保系统安全性与可靠性
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 路径常量定义
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 自动检测模型路径：优先检查 modules/ 目录，次选 models/ 目录
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(CURRENT_DIR), "semantic_model"),
    os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "models", "semantic_model")
]

# 配置标准日志输出
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("V-Guard-Semantic")

# ==========================================================
# 2. 核心模块加载
# ==========================================================
# 动态挂载源码路径
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    import pandas as pd
    from semantic_lib import SemanticLibrary
    from state_model import DrivingStateModel
    from risk_assessment import RiskManager
except ImportError as e:
    logger.error(f"Module Loading Error: {e}")
    sys.exit(1)

# ==========================================================
# 3. 路径自动锁定逻辑
# ==========================================================
def get_verified_model_path():
    """定位并验证模型文件完整性"""
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    return None


# ==========================================================
# 4. 主执行流水线
# ==========================================================
def main():
    print("-" * 60)
    print("V-GUARD SUB-SYSTEM: SEMANTIC SECURITY ASSESSMENT (OFFLINE)")
    print("-" * 60)

    # 1. 模型加载
    model_path = get_verified_model_path()
    if not model_path:
        logger.error("Model Not Found: 未在预设目录中检测到语义模型。")
        logger.info(f"请确保模型存放在: {POSSIBLE_MODEL_PATHS[0]}")
        sys.exit(1)

    logger.info(f"定位模型资源: {model_path}")

    try:
        # 初始化语义库
        sem_lib = SemanticLibrary(model_name=model_path)

        # 数据集路径配置
        data_path = os.path.join(CURRENT_DIR, 'data', 'raw_commands.csv')
        if not os.path.exists(data_path):
            logger.warning("数据集文件缺失，启动自动生成程序...")
            from scripts.generate_dataset import CommandGenerator
            CommandGenerator().generate_dataset(data_path)

        sem_lib.load_dataset(data_path)

        # 预处理：向量化与聚类 (修正之前的 KeyError)
        logger.info("执行大规模语义向量化预处理...")
        sem_lib.process_embeddings()
        sem_lib.cluster_commands(n_clusters=5)
        logger.info("离线语义大脑初始化完成。")

    except Exception as e:
        logger.error(f"Initialization Failed: {e}")
        return

    # 2. 模拟验证场景
    user_input = "现在立刻给我打开所有的车门"
    logger.info(f"输入语音指令: '{user_input}'")

    # 意图识别
    standard_cmd, cluster_id, confidence = sem_lib.match_intent(user_input)
    logger.info(f"意图识别结果: {standard_cmd} (Confidence: {confidence:.2f})")

    # 3. 驾驶状态校验
    state_model = DrivingStateModel()
    # 模拟高风险工况：车速 110km/h
    mock_sensor_data = {'speed': 110.0, 'speed_limit': 100.0}
    state_model.update_from_sensors(mock_sensor_data)
    state_vector = state_model.get_state_vector()

    # 4. 风险评分
    risk_manager = RiskManager(input_dim=16)
    cmd_embedding = sem_lib.model.encode([standard_cmd])[0]
    input_tensor = risk_manager.prepare_input(cmd_embedding, state_vector)

    risk_score = risk_manager.evaluate(input_tensor)
    decision, _ = risk_manager.generate_risk_matrix(risk_score)

    # 5. 决策输出
    print("\n" + "=" * 50)
    print("                V-GUARD DECISION REPORT")
    print("=" * 50)
    print(f"Detected Command : {standard_cmd}")
    print(f"Risk Evaluation  : {risk_score:.4f}")
    print(f"System Decision  : [{decision}]")
    print("-" * 50)

    if decision == "DENY":
        print("ACTION: BLOCKED. Reason: High speed violation for sensitive command.")
    else:
        print("ACTION: PERMITTED. Reason: Command verified safe under current state.")
    print("=" * 50)


if __name__ == "__main__":
    main()