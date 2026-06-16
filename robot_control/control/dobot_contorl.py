#!/usr/bin/env python3
import threading
from tkinter import N, NO

from debugpy import connect
from control.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, DobotApiFeedBack, MyType, alarmAlarmJsonFile
from time import sleep,time
import numpy as np
import re
from typing import List, Union
import os

connect_dic = {
    "192.168.5.1": "nova2",
    "192.168.5.2": "nova5",
}


class dobot_control:
    def __init__(self, connect_ip: str, save_log:bool=False):
        self.robotMode = None
        self.current_actual = None
        self.algorithm_queue = None
        self.enableStatus_robot = None
        self.robotErrorState = False

        self.joint_state = {
            'q_actual': [0.0] * 6,
            'qd_actual': [0.0] * 6,
            'i_actual': [0.0] * 6,
        }

        self.globalLockValue = threading.Lock()
        self.save_key = save_log
        self.stop_key = False

        self.ip = connect_ip
        self.dashboard:Union[DobotApiDashboard, None] = None
        self.moveControl:Union[DobotApiMove, None] = None
        self.feed:Union[DobotApi, None] = None
        self.feedFour:Union[DobotApiFeedBack, None] = None

        self.feed_thread = threading.Thread(target=self.GetFeed)
        self.feed_thread.daemon = True
        self.error_thread = threading.Thread(target=self.ClearRobotError)
        self.error_thread.daemon = True
        
        self.log = None
        if self.save_key:
            self.save_thread = threading.Thread(target=self.SaveLog)
            self.save_thread.daemon = True
            self.log = "/home/b600/robot/robot_control/log/"+connect_dic[self.ip]+".txt"

        self.ConnectRobot()

    def move(self, pose_list: List[List[float]]):
        for point in pose_list:
            self.moveControl.JointMovJ(point[0], point[1], point[2], point[3], point[4], point[5])
            self.waitArrive(point)

    def SaveLog(self):
        seq = 0   # 模拟递增的 seq
        with open(self.log, "w", encoding="utf-8") as f:
            while True:
                with self.globalLockValue:
                    joint_values = self.joint_state['q_actual']
                    if joint_values is not None and len(joint_values) == 6:
                        # 获取系统时间作为 stamp
                        t = time()
                        secs = int(t)
                        nsecs = int((t - secs) * 1e9)

                        # 格式化关节值
                        formatted_values = [float(f"{val:.8f}") for val in joint_values]

                        # 构造类似 ROS 的 YAML 格式
                        log_line = (
                            f"seq: {seq}\n"
                            f"secs: {secs}\n"
                            f"nsecs: {nsecs}\n"
                            f"position: {formatted_values}\n\n"
                        )

                        f.write(log_line)
                        f.flush()

                        seq += 1
                    else:
                        f.write(f"header:\n  seq: {seq}\n  无效关节值: {joint_values}\n\n")
                        f.flush()

    def DargOrNot(self, val: bool):
        if val:
            self.dashboard.StartDrag()
        else:
            self.dashboard.StopDrag()

    def ConnectRobot(self):
        try:
            dashboardPort = 29999
            movePort = 30003
            feedPort = 30004
            print("正在建立连接...")
            dashboard = DobotApiDashboard(self.ip, dashboardPort)
            move = DobotApiMove(self.ip, movePort)
            feed = DobotApi(self.ip, feedPort)
            feedFour = DobotApiFeedBack(self.ip, feedPort)
            print(">.<连接成功>!<")
            self.dashboard = dashboard
            self.moveControl = move
            self.feed = feed
            self.feedFour = feedFour

            self.feed_thread.start()
            self.error_thread.start()
            if self.save_key:
                self.save_thread.start()
                
            self.enableRobot()
            self.speedSet()
        except Exception as e:
            print(":(连接失败:(")
            raise e

    def waitArrive(self, value):
        while True:
            res = self.dashboard.GetAngle()
            if res:
                res = re.search(r'\{([^}]*)\}', res)
                numbers = [float(num.strip()) for num in res.group(1).split(',')]
                array = np.array(numbers).reshape(1, -1)[0]
                Arrive = True
                for i in range(6):
                    if Arrive:
                        val1 = abs(array[i] - value[i])
                        val2 = abs(array[i] + 180 - value[i])
                        val3 = abs(array[i] - 180 - value[i])
                        val4 = abs(array[i] + 360 - value[i])
                        val5 = abs(array[i] - 360 - value[i])
                        if val1 < 0.01 or val2 < 0.01 or val3 < 0.01 or val4 < 0.01 or val5 < 0.01:
                            Arrive = True
                        else:
                            Arrive = False
            if Arrive:
                sleep(1)
                break
            sleep(1)

    def enableRobot(self):
        print("开始使能...")
        self.dashboard.EnableRobot()
        print("完成使能:)")

    def unableRobot(self):
        print("停止使能...")
        self.dashboard.DisableRobot()
        print("停止使能:)")

    def speedSet(self, CP: int = 100, speedL: int = 25):
        self.dashboard.CP(CP)
        self.dashboard.SpeedL(speedL)

    def GetFeed(self):
        while True:
            with self.globalLockValue:
                feedInfo = self.feedFour.feedBackData()
                if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                    # Refresh Properties
                    self.robotMode = feedInfo['robot_mode'][0]
                    self.current_actual = feedInfo["tool_vector_actual"][0]
                    self.algorithm_queue = feedInfo['run_queued_cmd'][0]
                    self.enableStatus_robot = feedInfo['enable_status'][0]
                    self.robotErrorState = feedInfo['error_status'][0]
                    # 自定义添加所需反馈数据

                    self.joint_state['q_actual'] = feedInfo["q_actual"][0]
                    self.joint_state['qd_actual'] = feedInfo["qd_actual"][0]
                    self.joint_state['i_actual'] = feedInfo["i_actual"][0]
                sleep(0.001)


    def ClearRobotError(self):
        dataController, dataServo = alarmAlarmJsonFile()  # 读取控制器和伺服告警码
        while True:
            self.globalLockValue.acquire()
            if self.robotErrorState:
                numbers = re.findall(r'-?\d+', self.dashboard.GetErrorID())
                numbers = [int(num) for num in numbers]
                if (numbers[0] == 0):
                    if (len(numbers) > 1):
                        for i in numbers[1:]:
                            alarmState = False
                            if i == -2:
                                print("机器告警 机器碰撞 ", i)
                                alarmState = True
                            if alarmState:
                                continue
                            for item in dataController:
                                if i == item["id"]:
                                    print("机器告警 Controller errorid", i,
                                          item["zh_CN"]["description"])
                                    alarmState = True
                                    break
                            if alarmState:
                                continue
                            for item in dataServo:
                                if i == item["id"]:
                                    print("机器告警 Servo errorid", i,
                                          item["zh_CN"]["description"])
                                    break

                        choose = input("输入1, 将清除错误, 机器继续运行: ")
                        if int(choose) == 1:
                            self.dashboard.ClearError()
                            sleep(0.01)
                            self.dashboard.Continue()

            else:
                if int(self.enableStatus_robot) == 1 and int(self.algorithm_queue) == 0:
                    self.dashboard.Continue()
            self.globalLockValue.release()
            sleep(5)
