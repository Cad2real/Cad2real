import posix
import threading
from time import sleep
import sys
import os
import re
import time
from turtle import pos, position
from typing import List, Optional

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'dobot_control', 'robot'))

from control.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, DobotApiFeedBack, MyType, alarmAlarmJsonFile


class RobotController:
    def __init__(self, ip: str = "192.168.5.2"):
        self.ip = ip
        self.dashboard_port = 29999
        self.move_port = 30003
        self.feed_port = 30004
        
        # API对象
        self.dashboard: Optional[DobotApiDashboard] = None
        self.move: Optional[DobotApiMove] = None
        self.feed: Optional[DobotApi] = None
        self.feed_four: Optional[DobotApiFeedBack] = None
        
        # 状态变量
        self.current_actual = [-1]
        self.algorithm_queue = -1
        self.enable_status_robot = -1
        self.robot_error_state = False
        self.robot_mode = 0
        
        # ACT模型设计的状态存储
        self.joint_state = {
            'q_actual': [0.0] * 6,
            'qd_actual': [0.0] * 6,
            'i_actual': [0.0] * 6,
        }
        
        self.global_lock_value = threading.Lock()
        
        # 连接状态
        self.is_connected = False
        self.is_enabled = False
        self.feed_thread = None
        self.error_thread = None
        self.running = False
    
    def connect(self) -> bool:
        """连接机器人"""
        try:
            print("正在建立机器人连接...")
            self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            self.move = DobotApiMove(self.ip, self.move_port)
            self.feed = DobotApi(self.ip, self.feed_port)
            self.feed_four = DobotApiFeedBack(self.ip, self.feed_port)
            
            self.is_connected = True
            print("机器人连接成功!")
            self.dashboard.SpeedFactor(80)
            return True
            
        except Exception as e:
            print(f"机器人连接失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开机器人连接"""
        # if self.is_enabled:
        #     self.disable()
        
        self.running = False
        
        # 等待线程结束
        if self.feed_thread and self.feed_thread.is_alive():
            self.feed_thread.join(timeout=2.0)
        if self.error_thread and self.error_thread.is_alive():
            self.error_thread.join(timeout=2.0)
        
        self.is_connected = False
        print("机器人连接已断开")
    
    def enable(self) -> bool:
        """使能机器人"""
        if not self.is_connected:
            print("请先连接机器人")
            return False
        
        if self.is_enabled:
            return True
        
        try:
            # 清除错误
            self.dashboard.ClearError()
            sleep(0.1)
            
            # 启动反馈线程
            self.running = True
            self.feed_thread = threading.Thread(target=self._get_feed, daemon=True)
            self.feed_thread.start()
            
            # 启动错误处理线程
            self.error_thread = threading.Thread(target=self._clear_robot_error, daemon=True)
            self.error_thread.start()
            
            print("开始使能...")
            self.dashboard.EnableRobot()
            sleep(1.0)  # 等待使能完成
            
            self.is_enabled = True
            print("机器人使能完成!")
            return True
            
        except Exception as e:
            print(f"机器人使能失败: {e}")
            return False
    
    def disable(self):
        """去使能机器人"""
        if not self.is_enabled:
            return
        
        try:
            self.dashboard.DisableRobot()
            self.is_enabled = False
            self.running = False
            print("机器人已去使能")
        except Exception as e:
            print(f"机器人去使能失败: {e}")
    
    def move_to_joint_position(self, joint_list: List[float], wait: bool = True) -> bool:
        """移动到关节位置"""
        if not self.is_enabled:
            print("机器人未使能，无法移动")
            return False
        
        if len(joint_list) != 6:
            print("关节位置必须包含6个值")
            return False
        
        try:
            self.move.JointMovJ(joint_list[0], joint_list[1], joint_list[2],
                              joint_list[3], joint_list[4], joint_list[5])
            
        except Exception as e:
            print(f"移动失败: {e}")
            return False
    
    def move_to_cartesian_position(self, pose_list: List[float], wait: bool = True) -> bool:
        """移动到笛卡尔位置"""
        if not self.is_enabled:
            print("机器人未使能，无法移动")
            return False
        
        if len(pose_list) != 6:
            print("笛卡尔位置必须包含6个值 [x, y, z, rx, ry, rz]")
            return False
        
        try:
            self.move.MovL(pose_list[0], pose_list[1], pose_list[2],
                          pose_list[3], pose_list[4], pose_list[5])
            
        except Exception as e:
            print(f"移动失败: {e}")
            return False

    def move_linear_relative_user(self, axis: str, distance: float, user_index: int, speed: int = None, acceleration: int = None) -> bool:
        """
        沿指定的用户坐标系的轴向进行直线相对运动。
        该功能基于 `RelMovLUser` 指令实现。

        :param axis: 要移动的轴 ('x', 'y', 'z', 'rx', 'ry', 'rz')。
        :param distance: 移动的距离 (mm) 或旋转的角度 (度)。
        :param user_index: 要使用的用户坐标系索引。
        :param speed: 可选的运动速度比例 (1-100)。
        :param acceleration: 可选的运动加速度比例 (1-100)。
        :return: 指令是否成功发送。
        """
        if not self.is_enabled:
            print("机器人未使能，无法移动")
            return False

        valid_axes = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        axis_lower = axis.lower()
        if axis_lower not in valid_axes:
            print(f"无效的轴: {axis}. 请从 {valid_axes} 中选择。")
            return False

        # 初始化所有轴的偏移量为0
        offsets = {ax: 0.0 for ax in valid_axes}
        # 设置指定轴的偏移量
        offsets[axis_lower] = distance

        try:
            print(f"正在沿用户坐标系 {user_index} 的 {axis} 轴相对移动 {distance}...")
            self.move.RelMovLUser(
                offsets['x'], offsets['y'], offsets['z'],
                offsets['rx'], offsets['ry'], offsets['rz'],
                user_index,
                speed,
                acceleration
            )
            return True
        except Exception as e:
            print(f"沿用户坐标系相对移动失败: {e}")
            return False
    
    def is_moving(self) -> bool:
        """检查机器人是否在运动"""
        with self.global_lock_value:
            return self.algorithm_queue > 0
    
    def wait_for_motion_complete(self, timeout: float = 30.0):
        """等待运动完成"""
        import time
        start_time = time.time()
        
        while self.is_moving():
            if time.time() - start_time > timeout:
                print("等待运动完成超时")
                break
            sleep(0.1)
    
    def _get_feed(self):
        while self.running:
            try:
                with self.global_lock_value:
                    feed_info = self.feed_four.feedBackData()
                    if hex((feed_info['test_value'][0])) == '0x123456789abcdef':
                        # 刷新基础属性
                        self.robot_mode = feed_info['robot_mode'][0]
                        self.algorithm_queue = feed_info['run_queued_cmd'][0]
                        self.enable_status_robot = feed_info['enable_status'][0]
                        self.robot_error_state = feed_info['error_status'][0]
                        
                        # 从反馈数据中解析出关节位置、速度和电流，并存入新变量
                        self.joint_state['q_actual'] = feed_info["q_actual"][0]
                        self.joint_state['qd_actual'] = feed_info["qd_actual"][0]
                        self.joint_state['i_actual'] = feed_info["i_actual"][0]

                sleep(0.001) # 保持高频以获取实时数据
            except Exception as e:
                sleep(0.1)
    
    def get_current_state(self) -> dict:
        """
        获取最新的机器人关节状态 (qpos, qvel, effort)。
        """
        with self.global_lock_value:
            return self.joint_state.copy()

    def get_robot_mode(self) -> int:
        """
        获取机器人当前模式。
        用于判断机器人是否空闲（模式5），以控制任务序列。
        """
        with self.global_lock_value:
            return self.robot_mode
    
    def servo_j(self, joint_list: List[float]):
        """
        伺服控制模式，用于平滑的轨迹跟踪。
        """
        if not self.is_enabled:
            return False
        
        try:
            self.move.ServoJ(joint_list[0], joint_list[1], joint_list[2],
                           joint_list[3], joint_list[4], joint_list[5])
            return True
        except Exception as e:
            return False

    def _clear_robot_error(self):
        """清除机器人错误（线程函数）"""
        try:
            data_controller, data_servo = alarmAlarmJsonFile()
        except:
            data_controller, data_servo = [], []
        
        while self.running:
            try:
                with self.global_lock_value:
                    if self.robot_error_state:
                        numbers = re.findall(r'-?\d+', self.dashboard.GetErrorID())
                        numbers = [int(num) for num in numbers]
                        
                        if numbers[0] == 0 and len(numbers) > 1:
                            for i in numbers[1:]:
                                alarm_state = False
                                if i == -2:
                                    print(f"机器告警 - 机器碰撞: {i}")
                                    alarm_state = True
                                
                                if alarm_state:
                                    continue
                                
                                # 查找控制器错误
                                for item in data_controller:
                                    if i == item["id"]:
                                        print(f"机器告警 - Controller errorid {i}: {item['zh_CN']['description']}")
                                        alarm_state = True
                                        break
                                
                                if alarm_state:
                                    continue
                                
                                # 查找伺服错误
                                for item in data_servo:
                                    if i == item["id"]:
                                        print(f"机器告警 - Servo errorid {i}: {item['zh_CN']['description']}")
                                        break
                            
                            # 自动清除错误并继续
                            print("自动清除错误并继续运行...")
                            self.dashboard.ClearError()
                            sleep(0.01)
                            self.dashboard.Continue()
                    else:
                        if (int(self.enable_status_robot) == 1 and 
                            int(self.algorithm_queue) == 0):
                            self.dashboard.Continue()
                
                sleep(5)
            except Exception as e:
                print(f"错误处理线程异常: {e}")
                sleep(1)


def main(ip, test_position):
    """主函数 - 独立运行模式"""
    robot = RobotController(ip=ip)
    
    try:
        # 连接和初始化
        if not robot.connect():
            return
        
        if not robot.enable():
            return
        for pos in test_position:
            # test_position = [47.082, -7.203, 134.137, -307.471, 77.134, 97.035]
            print("移动到测试位置...")
            robot.move_to_joint_position(pos)
            

            time.sleep(1)
            # 等待运动完成 (通过检查机器人模式是否变为空闲)
            # while robot.get_robot_mode() != 5:
            #     state = robot.get_current_state()
            #     print(f"Moving... Current J1: {state['q_actual'][0]:.2f}", end='\r')
            #     sleep(0.1)

        # print("\n到达测试位置!")
        # sleep(2)
        
        # --- 延坐标轴运动 ---
        # 假设用户坐标系 0 (基坐标系) 已被设置或为默认
        # user_coordinate_index = 0

        # robot.move_linear_relative_user(axis='z', distance=-100, user_index=user_coordinate_index, speed=0.1)
        # # robot.move_linear_relative_user(axis='y', distance=-150, user_index=user_coordinate_index, speed=1)
        # while robot.get_robot_mode() != 5:
        #     sleep(0.1)
        # print(f"\n完成!")
        # sleep(2)
        
    except KeyboardInterrupt:
        print("\n用户中断操作...")
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        robot.disconnect()


if __name__ == '__main__':
    position =[
        # [-15.72, -15.487, -35.481, 9.633, 54.229, 30.058],
        # [-15.723, -7.24, -65.115, 31.017, 54.231, 30.062]
        [0,0,0,0,0,0]


        # [25.213, 7.183, 57.852, 28.697, -105.135, -148.978],
        # [25.205, 3.648, 75.168, 14.912, -105.137, -148.994],
        # [17.975, -2.085, 81.197, 12.645, -105.495, -156.482],
        # [4.774, -9.36, 88.933, 8.525, -105.501, -170.184],
        # [4.781, -6.831, 75.348, 19.585, -105.5, -170.17]


        # [39.961, 5.6, 58.411, 41.313, -91.789, -79.595],
    ]

    position=[
    # [-32.269, -3.672, -30.636, -12.694, 50.009, 33.402],
    # [-29.148, -6.799, -33.192, -5.902, 49.969, 31.725],
    # [-26.027, -9.926, -35.749, 0.891, 49.928, 30.048],
    # [-22.907, -13.052, -38.305, 7.683, 49.888, 28.370],
    # [-19.786, -16.179, -40.862, 14.476, 49.847, 26.693],
    # [-16.665, -19.306, -43.418, 21.268, 49.807, 25.016],
    [-13.544, -22.433, -45.975, 28.061, 49.766, 23.339],
]

    main(ip="192.168.5.1", test_position=position)
    # main(ip="192.168.5.2",test_position=position)