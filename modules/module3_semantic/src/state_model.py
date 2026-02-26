import json
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # 告诉程序在后台绘图，不要弹出那个白色的预览窗口
class DrivingStateModel:
    def __init__(self):
        # 定义状态空间
        self.state = {
            'speed': 0.0,              # 当前车速 km/h
            'speed_limit': 0.0,        # 道路限速 km/h
            'auto_mode': False,        # 是否无人驾驶
            'weather': 'Sunny',        # 天气
            'surroundings': {          # 周围感知 (模拟传感器输入)
                'front_dist': 100.0,   # 前车距离 m
                'lane_keeping': True,  # 是否压线
                'pedestrian_near': False # 附近是否有行人
            },
            'vehicle_health': 'Normal' # 车辆健康状态
        }

    def update_from_sensors(self, sensor_data_dict=None):
        """
        模拟从摄像头/雷达读取数据。
        实际项目中，这里会接入 CAN 总线或 ROS 话题。
        """
        if sensor_data_dict:
            self.state.update(sensor_data_dict)
        else:
            # 模拟随机生成一个驾驶场景用于测试
            self.state['speed'] = random.uniform(0, 120)
            self.state['speed_limit'] = random.choice([40, 60, 80, 120])
            self.state['auto_mode'] = random.choice([True, False])
            self.state['surroundings']['front_dist'] = random.uniform(5, 200)
            self.state['surroundings']['pedestrian_near'] = random.random() < 0.1
        return self.state

    def get_state_vector(self):
        """将状态转换为模型可输入的数值向量"""
        # 归一化处理示例
        vector = [
            self.state['speed'] / 150.0,
            self.state['speed_limit'] / 150.0,
            1.0 if self.state['auto_mode'] else 0.0,
            self.state['surroundings']['front_dist'] / 200.0,
            1.0 if self.state['surroundings']['pedestrian_near'] else 0.0,
            1.0 if self.state['surroundings']['lane_keeping'] else 0.0
        ]
        return np.array(vector, dtype=np.float32)

    def visualize_state(self, save_path='ui/assets/m3_state.png'):
        """可视化当前驾驶状态"""
        # 【中文字体设置】
        import matplotlib.font_manager as fm
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        labels = ['当前车速', '道路限速', '自动驾驶', '前车距离', '附近行人', '车道保持']
        data = self.get_state_vector()
        
        # 创建中文状态说明
        state_descriptions = [
            f"{self.state['speed']} km/h",
            f"{self.state['speed_limit']} km/h",
            "开启" if self.state['auto_mode'] else "关闭",
            f"{self.state['surroundings']['front_dist']:.1f} m",
            "有" if self.state['surroundings']['pedestrian_near'] else "无",
            "正常" if self.state['surroundings']['lane_keeping'] else "异常"
        ]
        
        plt.figure(figsize=(10, 7))
        bars = plt.bar(labels, data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'], 
                      edgecolor='black', linewidth=1.5)
        
        plt.title("当前驾驶状态向量 (归一化)", fontsize=16, fontweight='bold', pad=20)
        plt.ylim(0, 1.2)
        plt.xlabel("状态指标", fontsize=12)
        plt.ylabel("归一化数值 (0-1)", fontsize=12)
        
        # 在柱子上添加数值和实际状态
        for i, (v, desc) in enumerate(zip(data, state_descriptions)):
            plt.text(i, v + 0.05, f"{v:.2f}\n({desc})", ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 添加网格
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 旋转 x 轴标签
        plt.xticks(rotation=15, fontsize=10)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 驾驶状态图已保存至：{save_path}")
        plt.close()
        
        # 打印详细 JSON
        print("📋 当前状态详细信息:")
        print(f"   车速：{self.state['speed']} km/h")
        print(f"   限速：{self.state['speed_limit']} km/h")
        print(f"   自动驾驶模式：{'开启' if self.state['auto_mode'] else '关闭'}")
        print(f"   前车距离：{self.state['surroundings']['front_dist']:.1f} m")
        print(f"   附近行人：{'有' if self.state['surroundings']['pedestrian_near'] else '无'}")
        print(f"   车道保持：{'正常' if self.state['surroundings']['lane_keeping'] else '异常'}")

