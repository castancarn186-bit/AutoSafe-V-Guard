# c:\Users\Leo\Desktop\gemini\interactive_tester.py
import sys
import torch
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# --- 路径环境配置 ---
CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / 'models'
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))
if str(MODELS_DIR) not in sys.path: sys.path.insert(0, str(MODELS_DIR))

from modules.module3_semantic.core.protocol import SemanticInput, VehicleContext, Language, WeatherCondition, RiskLevel
from models.reasoning import SemanticSafetyEngine

class VGuardIntelligentTerminal:
    def __init__(self):
        print("=== 初始化 V-Guard 全语义解析交互终端 ===")
        self.engine = SemanticSafetyEngine()
        
        self.model_path = "./models/Qwen/Qwen2.5-0.5B-Instruct"
        print(f"[*] 正在加载 LLM 语义解析模型: {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto", 
            device_map="auto"
        )
        print("[+] 系统就绪：协议已转为后台支撑，模型接管语义。")

    def _llm_extract_context(self, description: str) -> VehicleContext:
        """
        LLM 解析层：尝试从自然语言提取结构化数据
        """
        prompt = f"""<|im_start|>system
你是一个高精度的车载传感器数据提取器。
请将用户的【描述】转换为 JSON。字段包含：
- "speed": 数字 (0-150)
- "speed_limit": 数字 (10-120)
- "gear": 字符串 ("P", "R", "N", "D")
- "weather": 字符串 ("sunny", "rainy", "snowy", "hail")
- "pedestrians": 布尔值 (true/false)

描述: "{description}"<|im_end|>
<|im_start|>assistant
{{"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        outputs = self.llm.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = "{" + self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            raw_data = json.loads(match.group().replace("'", '"'))
            
            w_str = raw_data.get("weather", "sunny").lower()
            weather_val = getattr(WeatherCondition, w_str.upper(), WeatherCondition.SUNNY)
            
            # --- 实例化并触发 Pydantic 校验 ---
            return VehicleContext(
                speed=float(raw_data.get("speed", 0.0)),
                speed_limit=float(raw_data.get("speed_limit", 120.0)), # 👈 显式补全
                gear=raw_data.get("gear", "P"),
                weather=weather_val,
                traffic_density="medium",
                has_pedestrians=bool(raw_data.get("pedestrians", False))
            )
        except Exception as e:
            # 👈 兜底逻辑：必须包含协议定义的所有非默认必填字段
            print(f"⚠️ 解析异常 (使用协议默认安全值): {e}")
            return VehicleContext(
                speed=0.0, 
                speed_limit=120.0, 
                gear="P", 
                weather=WeatherCondition.SUNNY, 
                traffic_density="low",
                has_pedestrians=False
            )

    def start(self):
        print("\n" + "="*50)
        print("🚗 V-Guard 2.0: 语义映射 + 协议守护模式")
        print("您可以输入感性描述，如：'外面天漏了，路都看不清'")
        print("="*50)

        while True:
            cmd = input("\n👉 请输入语音指令 (q退出): ").strip()
            if cmd.lower() == 'q': break
            
            env_desc = input("🌍 请描述当前车况: ").strip()
            
            context = self._llm_extract_context(env_desc)
            print(f"🤖 [协议建模结果]: 速度={context.speed} | 限速={context.speed_limit} | 档位={context.gear} | 天气={context.weather.value}")

            test_input = SemanticInput(
                text=cmd,
                language=Language.ZH,
                context=context
            )
            
            report = self.engine.evaluate(test_input)
            
            color = "\033[91m" if report.level == RiskLevel.DANGER else "\033[93m" if report.level == RiskLevel.WARNING else "\033[92m"
            print(f"\n{color}[最终决策报告]")
            print(f"📊 风险得分: {report.risk_score}")
            print(f"💬 决策理由: {report.reason}\033[0m")

if __name__ == "__main__":
    VGuardIntelligentTerminal().start()