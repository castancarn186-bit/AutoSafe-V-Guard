# AutoSafe-V-Guard


```text
V-Guard/
├── core/
│   ├── engine.py           # 核心调度与风险融合算法 (Brain)
│   ├── decision.py         # 决策逻辑（放行/二次确认/拦截）
│   └── protocol.py         # 定义统一的数据结构 (Data Schema)
├── modules/
│   ├── __init__.py         # 定义 BaseGuardModule 基类
│   ├── module_a/           # 第一部分: 声学物理层
│   ├── module_b/           # 第二部分: ASR 行为安全
│   └── module_c/           # 第三部分: 语义与车辆状态
├── ui/                     # 第四部分: App 界面 (Flet/Flutter)
├── hardware/               # 第四部分·: 树莓派外设控制 (LED/Buzzer)
├── main_app.py             # 界面模式启动入口
├── main_embedded.py        # 嵌入式无头模式启动入口
└── requirements.txt        # 环境依赖

```