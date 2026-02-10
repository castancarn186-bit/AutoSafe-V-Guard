# AutoSafe-V-Guard


```text
V-Guard/
├── main.py                 # 唯一启动点，负责调度 UI 线程与算法引擎线程
├── requirements.txt        # 项目依赖库，用于一键安装环境
├── assets/                 # 存放视觉素材，App打包时包含
│   ├── icon.png            # App桌面图标
│   ├── logo.png            # 启动画面Logo
│   └── fonts/              # 字体
├── core/                   # 系统内核
│   ├── protocol.py         # 统一数据契约（RiskReport, SystemContext）
│   ├── base_module.py      # 模块基类（BaseDetector），强制规范行为
│   ├── state.py            # 全局共享状态机（State Hub），连接UI与后台
│   ├── engine.py           # 风险融合引擎（Fusion Engine），负责加权决策
│   └── simulator.py        # 上帝模式模拟器，用于演示攻击场景
├── modules/                # 具体防御模块
│   ├── module_a_acoustic/  # 声学物理层（继承 BaseDetector）
│   │   └── detector.py
│   ├── module_b_asr/       # ASR 行为层（继承 BaseDetector）
│   │   └── analyzer.py
│   └── module_c_semantic/  # 语义冲突层（继承 BaseDetector）
│       ├── configs/        # 存放policy.json（车载安全约束矩阵）
│       └── reasoning.py    # 执行意图 vs 状态的校验逻辑
├── ui/                     # 可视化界面
│   ├── app.py              # Flet 响应式界面代码（支持 App/Web）
│   └── components/         # 存放 UI 小组件
├── hardware/               # 物理反馈
│   └── gpio_ctrl.py        # 树莓派GPIO控制，驱动LED灯带和继电器
├── data/                   # 影子模式日志
│   └── defense_logs.db     # 使用 SQLite 存储拦截记录
└── scripts/                # 打包与部署脚本
    ├── build_exe.bat       # Windows打包脚本
    └── deploy_pi.sh        # 树莓派部署脚本

```