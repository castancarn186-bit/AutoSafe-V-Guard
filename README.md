# AutoSafe-V-Guard


```text
V-Guard/
├── core/                   # 系统中枢 
│   ├── engine.py           # 风险融合引擎 (计算 A+B+C 的最终分)
│   ├── state.py            # 全局共享状态 (存放风险分、决策、车辆时速)
│   ├── protocol.py         # 统一数据协议 (定义 RiskReport 格式)
│   └── simulator.py        # 模拟器 (模拟高速、障碍物等物理环境)
├── modules/                # 具体算法
│   ├── module_a_acoustic/  # 声学层 (物理注入检测)
│   ├── module_b_asr/       # 行为层 (ASR 特征分析)
│   └── module_c_semantic/  # 语义层 (意图-状态冲突推演)
│       ├── configs/        # 存放 policy.json (冲突矩阵)
│       └── reasoning.py    # 深度推演逻辑
├── ui/                     # 可视化看板
│   └── app.py              # iOS 金属质感看板代码
├── hardware/               # 物理反馈 (树莓派 GPIO 灯光/继电器)
│   └── gpio_ctrl.py
├── data/                   # 影子模式记录 (自主学习的数据集)
│   └── unknown_intents.log
├── main.py                 # 唯一启动入口 (整合 UI 与后台任务)
└── README.md               # 项目说明

```