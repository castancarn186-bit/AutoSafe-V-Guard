# c:\Users\Leo\Desktop\gemini\data_pipeline\synthesizer.py
import json
import random
import uuid
from pathlib import Path

def generate_hybrid_dataset(count=10000):
    # 保存路径：根目录下
    output_path = Path(__file__).resolve().parent.parent / "semantic_safety_dataset_10000.jsonl"
    print(f"[*] 启动 V-Guard 增强型数据集生成器...")
    print(f"[*] 目标规模: {count} 条 | 包含对冲逻辑与环境变量")

    # 意图模版库 (涵盖安全、警告、危险三类意图)
    templates = [
        {"text": "帮我打开车门", "intent": "open_door", "type": "physical"},
        {"text": "我现在想下车", "intent": "open_door", "type": "physical"},
        {"text": "把窗户全部降下来", "intent": "open_window", "type": "physical"},
        {"text": "帮我开一点点窗缝透气", "intent": "open_window_slight", "type": "physical"},
        {"text": "调高空调温度", "intent": "set_ac", "type": "safe"},
        {"text": "我想听首欢快的歌", "intent": "media", "type": "safe"},
        {"text": "打开后排娱乐屏幕", "intent": "entertainment", "type": "media"},
        {"text": "帮我开启自动紧急制动", "intent": "emergency", "type": "safety_system"}
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(count):
            tpl = random.choice(templates)
            
            # 随机生成环境变量
            speed = random.uniform(0, 150)
            # 增加“静止状态”的权重，用于训练模型分辨静止与行驶的巨大差异
            if random.random() < 0.3: speed = 0.0 
            
            weather = random.choice(["sunny", "rainy", "snowy", "hail"])
            gear = "D" if speed > 5 else random.choice(["P", "R", "N"])
            has_peds = random.choice([True, False]) # 周边是否有行人
            
            # --- 专家打分逻辑 (神经网络的学习目标) ---
            risk_score = 0.0
            reason = "SAFE: 环境安全，操作合理。"

            # 逻辑 1: 物理门锁限制 (极其严格)
            if tpl["intent"] == "open_door":
                if speed > 5:
                    risk_score = 1.0
                    reason = f"FATAL: 车辆行驶中({speed:.1f}km/h)严禁操作车门。"
                elif has_peds:
                    risk_score = 0.8
                    reason = "DANGER: 车辆静止但周围有行人经过，开门存在碰撞风险。"
                elif gear != "P":
                    risk_score = 0.4
                    reason = "WARNING: 建议先挂入P档再开启车门。"

            # 逻辑 2: 车窗逻辑 (受速度与天气双重影响)
            elif tpl["intent"] == "open_window":
                if speed > 80:
                    risk_score = 0.9
                    reason = "DANGER: 高速行驶禁止大幅开启车窗。"
                elif weather in ["rainy", "hail"]:
                    risk_score = 0.7
                    reason = "WARNING: 检测到降雨/冰雹，开启车窗可能损坏内饰。"
            
            # 逻辑 3: 娱乐系统 (防分心逻辑)
            elif tpl["intent"] == "entertainment":
                if speed > 10:
                    risk_score = 0.5
                    reason = "WARNING: 行驶中开启视频可能分散驾驶员注意力。"

            # 逻辑 4: 安全系统 (永远是安全的)
            elif tpl["type"] == "safety_system":
                risk_score = 0.0
                reason = "SAFE: 安全相关系统允许任何时刻操作。"

            # 构造数据条目
            item = {
                "id": str(uuid.uuid4()),
                "text": tpl["text"],
                "intent_label": tpl["intent"],
                "context": {
                    "speed": round(speed, 1),
                    "gear": gear,
                    "weather": weather,
                    "has_pedestrians": has_peds
                },
                "ground_truth_score": risk_score,
                "reason": reason
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            if (i + 1) % 2000 == 0:
                print(f"[>] 已完成 {i + 1} 条...")

    print(f"【成功】增强版数据集已保存至: {output_path}")

if __name__ == "__main__":
    generate_hybrid_dataset(10000)