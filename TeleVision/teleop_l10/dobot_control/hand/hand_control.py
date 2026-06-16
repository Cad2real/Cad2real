import subprocess
import sys
import time
from typing import Union
import numpy as np
import os
import threading
from queue import Queue, Empty

class hand_control:
    def __init__(self, password: Union[int,str]):
        self.password = str(password)
        self.service_is_start = False
        self.service_available = False
        self.command_queue = Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
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
            time.sleep(3)  # 增加等待时间确保服务完全启动
            if subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip() == 'active':
                self.service_is_start = True
                print('Start service of hand')
                
                # 检查ROS服务可用性
                self._wait_for_service()
                
                # 启动工作线程
                self._start_worker_thread()
            else:
                print("Failed to start hand service")
        except subprocess.CalledProcessError as e:
            print('error:\n')
            print(e)
            sys.exit(1)
    
    def _wait_for_service(self):
        """等待ROS服务可用"""
        print("Waiting for ROS service to be available...")
        for i in range(15):  # 增加等待时间到15秒
            try:
                result = subprocess.run(
                    ["/opt/ros/noetic/bin/rosservice", "list"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ,
                    timeout=5  # 增加超时时间
                )
                if "/inspire_hand/set_angle" in result.stdout:
                    self.service_available = True
                    print("ROS service is available!")
                    return
                else:
                    print(f"Waiting for /inspire_hand/set_angle... ({i+1}/15)")
                    time.sleep(1)
            except Exception as e:
                print(f"Error checking service: {str(e)}")
                time.sleep(1)
        
        if not self.service_available:
            print("Timeout: /inspire_hand/set_angle not available.")
            sys.exit(1)
    
    def _start_worker_thread(self):
        """启动工作线程处理指令队列"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            print("Hand control worker thread started")
    
    def _worker_loop(self):
        """工作线程主循环"""
        while not self.stop_event.is_set():
            try:
                # 非阻塞获取命令，稍微增加超时时间
                angle = self.command_queue.get(timeout=0.05)
                
                # 执行命令
                self._execute_angle_command(angle)
                
                # 标记任务完成
                self.command_queue.task_done()
                
            except Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")
    
    def _execute_angle_command(self, angle):
        """执行单个角度命令 - 修复超时问题"""
        try:
            # 创建参数列表
            angle_args = [str(int(item)) for item in angle]
            
            # 使用绝对路径调用rosservice
            cmd = [
                "/opt/ros/noetic/bin/rosservice",
                "call",
                "/inspire_hand/set_angle"
            ] + angle_args
            
            # 增加超时时间，确保命令能够完成执行
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ,
                timeout=1.0  # 增加超时时间到1秒
            )
            
            if result.returncode != 0:
                print(f'Hand command error: {result.stderr}')
            else:
                # 成功执行时可以打印调试信息（可选）
                pass
                
        except subprocess.TimeoutExpired:
            print("Hand command timeout - ROS service may be slow")
        except Exception as e:
            print(f"Error executing hand command: {str(e)}")
    
    def service_stop(self):
        print("Stop the hand_service")
        
        # 停止工作线程
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.worker_thread.join(timeout=2.0)
            print("Worker thread stopped")
        
        # 停止系统服务
        stop_command = f'echo {self.password} | sudo -S systemctl stop inspire_hand_start.service'
        check_command = f'echo {self.password} | sudo -S systemctl is-active inspire_hand_start.service'
        try:
            subprocess.run(stop_command, shell=True, stdout=subprocess.PIPE, text=True)
            time.sleep(1)
            if subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip() == 'inactive':
                self.service_is_start = False
                self.service_available = False
                print('Close hand service')
        except subprocess.CalledProcessError as e:
            print('error:\n')
            print(e)
    
    def move2bag(self):
        if self.service_is_start:
            cmd = np.array([
                [1000,1000,1000,1000,1000,1000],
                [0,0,0,0,0,0]
            ])
            self.angle_move(cmd)
        else:
            print("Service don't start")

    def move2zero(self):
        if self.service_is_start:
            cmd = np.array([
                [1000,1000,1000,1000,1000,1000]
            ])
            self.angle_move(cmd)
        else:
            print("Service don't start")

    def angle_move(self, angles: np.ndarray):
        """非阻塞角度移动 - 优化版本"""
        if not self.service_is_start or not self.service_available:
            print("Service not ready")
            return
        
        # 减少队列清理频率，避免丢失指令
        queue_size = self.command_queue.qsize()
        if queue_size > 5:  # 只有当队列过大时才清理
            cleared_count = 0
            while not self.command_queue.empty() and cleared_count < queue_size - 2:
                try:
                    self.command_queue.get_nowait()
                    cleared_count += 1
                except Empty:
                    break
        
        # 添加新指令到队列
        for angle in angles:
            try:
                self.command_queue.put_nowait(angle)
            except:
                # 队列满了，等待短暂时间后重试
                time.sleep(0.001)
                try:
                    self.command_queue.put_nowait(angle)
                except:
                    print("Command queue full, skipping command")
    
    def angle_move_immediate(self, angle):
        """立即执行单个角度命令（用于关键动作）"""
        if not self.service_is_start or not self.service_available:
            print("Service not ready")
            return
        
        # 直接执行，不通过队列
        self._execute_angle_command(angle)
    
    def test_connection(self):
        """测试ROS服务连接"""
        if not self.service_is_start or not self.service_available:
            print("Service not ready for testing")
            return False
        
        try:
            # 发送一个测试命令
            test_angle = [500, 500, 500, 500, 500, 500]
            result = subprocess.run(
                ["/opt/ros/noetic/bin/rosservice", "call", "/inspire_hand/set_angle"] + [str(x) for x in test_angle],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ,
                timeout=2.0
            )
            
            if result.returncode == 0:
                print("Hand service connection test: SUCCESS")
                return True
            else:
                print(f"Hand service connection test: FAILED - {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Hand service connection test: ERROR - {e}")
            return False