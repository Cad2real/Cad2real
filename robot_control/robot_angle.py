import time
import sys
import os
from typing import List
import pyperclip

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 导入RobotController
from robot_control import RobotController

def get_and_print_joint_values(ip:str):
    """
    连接机器人，读取当前的关节值（角度制），并以不带引号的列表格式打印。
    """
    robot = None
    try:
        # 1. 实例化 RobotController
        robot = RobotController(ip=ip)

        # 2. 连接机器人
        if not robot.connect():
            print("连接机器人失败，请检查网络和IP设置。")
            return

        # 3. 使能机器人
        if not robot.enable():
            print("使能机器人失败。")
            robot.disconnect()
            return
        
        # 等待反馈线程启动并更新数据
        time.sleep(2)

        # 4. 获取并打印当前的关节值
        print("\n正在读取当前机械臂的关节值...")
        current_state = robot.get_current_state()
        
        joint_values = current_state.get('q_actual', None)

        if joint_values is not None and len(joint_values) == 6:
            formatted_values_str = [f"{val:.3f}" for val in joint_values]
            
            final_values = [float(val_str) for val_str in formatted_values_str]
            
            print(f"{final_values}")
            pyperclip.copy(str(final_values)) 
            
        else:
            print(f"无法获取有效的关节值。请确保机器人已连接并使能。")
            print(f"当前获取到的值为: {joint_values}")

    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        # 5. 断开连接
        if robot:
            # robot.disconnect()
            print("程序结束。")

if __name__ == "__main__":
    # get_and_print_joint_values(ip="192.168.5.1")

    get_and_print_joint_values(ip="192.168.5.2")