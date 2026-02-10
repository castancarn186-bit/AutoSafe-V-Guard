#树莓派外设驱动	根据决策控制红色 LED（拦截）或绿色 LED（通过）。
import platform
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VGuard.Hardware")


class GPIOController:
    """
    硬件抽象层：自动检测环境并切换【真实硬件】或【模拟模式】
    """

    def __init__(self):
        self.system = platform.system()
        self.is_pi = False

        # 尝试检测是否为树莓派环境
        try:
            if self.system == "Linux":
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.is_pi = True
                self._setup_real_gpio()
            else:
                self._setup_mock_gpio()
        except ImportError:
            self._setup_mock_gpio()

    def _setup_real_gpio(self):
        """配置树莓派真实引脚"""
        self.LED_GREEN = 17  # 安全指示灯
        self.LED_RED = 27  # 拦截指示灯
        self.RELAY_PIN = 22  # 继电器（模拟切断物理麦克风）

        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setwarnings(False)
        self.GPIO.setup([self.LED_GREEN, self.LED_RED, self.RELAY_PIN], self.GPIO.OUT)
        self.GPIO.output(self.LED_GREEN, self.GPIO.LOW)
        self.GPIO.output(self.LED_RED, self.GPIO.LOW)
        self.GPIO.output(self.RELAY_PIN, self.GPIO.HIGH)  # 默认闭合(放行)
        logger.info("🟢 硬件模块：检测到 Raspberry Pi，真实 GPIO 已就绪")

    def _setup_mock_gpio(self):
        """配置模拟模式"""
        self.is_pi = False
        logger.info(f"💻 硬件模块：当前环境为 {self.system}，进入【虚拟硬件模拟模式】")

    def set_status(self, decision: str):
        """
        根据决策结果控制硬件逻辑
        decision: 'PASS', 'WARN', 'BLOCK'
        """
        if decision == "PASS":
            self._execute(green=True, red=False, relay=True)
        elif decision == "WARN":
            self._execute_blink(color="AMBER")
        elif decision == "BLOCK":
            self._execute(green=False, red=True, relay=False)

    def _execute(self, green: bool, red: bool, relay: bool):
        """执行具体的电平动作"""
        if self.is_pi:
            self.GPIO.output(self.LED_GREEN, self.GPIO.HIGH if green else self.GPIO.LOW)
            self.GPIO.output(self.LED_RED, self.GPIO.HIGH if red else self.GPIO.LOW)
            self.GPIO.output(self.RELAY_PIN, self.GPIO.HIGH if relay else self.GPIO.LOW)
        else:
            # 模拟控制台输出，用于演示
            status = f"灯光: {'[绿灯]' if green else '[-]'}{'[红灯]' if red else '[-]'} | 物理链路: {'[导通]' if relay else '[拦截断开]'}"
            logger.info(f"模拟硬件动作 >> {status}")

    def _execute_blink(self, color: str):
        """模拟/执行警告闪烁逻辑"""
        logger.info(f"模拟硬件动作 >> [警告模式] {color} 闪烁中...")
        if self.is_pi:
            # 真实闪烁逻辑（此处可根据需要实现线程闪烁）
            pass

    def cleanup(self):
        """释放资源"""
        if self.is_pi:
            self.GPIO.cleanup()
            logger.info("硬件资源已释放")


# 单例模式，方便全项目调用
vguard_hw = GPIOController()