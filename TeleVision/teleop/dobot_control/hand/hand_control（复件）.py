import subprocess
import sys
import time
from typing import Union
import numpy as np
import os

class hand_control:
    def __init__(self, password: Union[int,str]):
        self.password = str(password)
        self.service_is_start = False
        
        # 设置ROS环境变量 - 使用绝对路径
        os.environ['ROS_MASTER_URI'] = 'http://localhost:11311'
        os.environ['ROS_ROOT'] = '/opt/ros/noetic/share/ros'
        os.environ['ROS_PACKAGE_PATH'] = '/home/b600/wl/hand_control/src:/opt/ros/noetic/share'
        os.environ['LD_LIBRARY_PATH'] = '/opt/ros/noetic/lib'
        
        # 添加catkin工作空间的Python路径
        catkin_python_path = '/home/b600/wl/hand_control/devel/lib/python3/dist-packages'
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] += os.pathsep + catkin_python_path
        else:
            os.environ['PYTHONPATH'] = catkin_python_path
        
        # 添加ROS的Python路径
        ros_python_path = '/opt/ros/noetic/lib/python3/dist-packages'
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] += os.pathsep + ros_python_path
        else:
            os.environ['PYTHONPATH'] = ros_python_path

    def service_start(self):
        print("Try to start the hand_service")
        start_command = f'echo {self.password} | sudo -S systemctl start inspire_hand_start.service'
        check_command = f'echo {self.password} | sudo -S systemctl is-active inspire_hand_start.service'
        try:
            subprocess.run(start_command, shell=True, stdout=subprocess.PIPE, text=True)
            time.sleep(1)
            if subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip() == 'active':
                self.service_is_start = True
                print('Start service of hand')
        except subprocess.CalledProcessError as e:
            print('error:\n')
            print(e)
            sys.exit(1)
    
    def service_stop(self):
        print("Stop the hand_service")
        stop_command = f'echo {self.password} | sudo -S systemctl stop inspire_hand_start.service'
        check_command = f'echo {self.password} | sudo -S systemctl is-active inspire_hand_start.service'
        try:
            subprocess.run(stop_command, shell=True, stdout=subprocess.PIPE, text=True)
            time.sleep(1)
            if subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip() == 'inactive':
                self.service_is_start = False
                print('Close hand service')
        except subprocess.CalledProcessError as e:
            print('error:\n')
            print(e)
            sys.exit(1)
    
    def move2bag(self):
        if self.service_is_start:
            cmd = np.array([
                [1000,1000,1000,1000,1000,1000],
                [0,0,0,0,0,0]
            ])
            self.angle_move(cmd)
        else:
            print("Service don't start")
            sys.exit(0)

    def move2zero(self):
        if self.service_is_start:
            cmd = np.array([
                [1000,1000,1000,1000,1000,1000]
            ])
            self.angle_move(cmd)
        else:
            print("Service don't start")
            sys.exit(0)

    def angle_move(self, angles: np.ndarray):
        if not self.service_is_start:
            print("Service don't start")
            sys.exit(0)
            
        # 等待服务可用
        for i in range(5):  # 增加等待时间到20秒
            try:
                # 使用绝对路径调用rosservice
                result = subprocess.run(
                    ["/opt/ros/noetic/bin/rosservice", "list"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ  # 传递设置的环境变量
                )
                if "/inspire_hand/set_angle" in result.stdout:
                    break
                else:
                    print(f"Waiting for /inspire_hand/set_angle to be available... ({i+1}/20)")
                    time.sleep(1)
            except Exception as e:
                print(f"Error checking service: {str(e)}")
                time.sleep(1)
        else:
            print("Timeout: /inspire_hand/set_angle not available.")
            sys.exit(1)

        # 调用服务
        for angle in angles:
            try:
                # 创建参数列表：每个位置值作为单独的参数
                angle_args = [str(int(item)) for item in angle]  # 转换为整数再转字符串确保格式正确
                
                # 使用绝对路径调用rosservice
                cmd = [
                    "/opt/ros/noetic/bin/rosservice",
                    "call",
                    "/inspire_hand/set_angle"
                ] + angle_args  # 将位置参数添加到命令列表
                
                print(f"Calling command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ  # 传递设置的环境变量
                )
                
                if result.returncode == 0:
                    print(f'hand_angle: {angle_args}')
                    print(f'Result: {result.stdout}')
                else:
                    print(f'Error calling service: {result.stderr}')
                    print(f'Exit code: {result.returncode}')
                    
            except Exception as e:
                print(f"Error calling service: {str(e)}")
                sys.exit(1)