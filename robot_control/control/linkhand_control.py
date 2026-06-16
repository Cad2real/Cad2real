#!/usr/bin/env python3
import sys,os,time,argparse
from typing import List
target_dir = os.path.abspath('./linker_hand_python_sdk')
sys.path.append(target_dir)
from control.linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
from control.linker_hand_python_sdk.LinkerHand.gui_control import LinkHand_Gui
# from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi

import threading

Type_Dict = {
    "L10": 10
}

def point_check(point:List[float], hand_joint:str):
    if len(point) != Type_Dict[hand_joint]:
        raise False
    else:
        return True


class linkhand_control:
    def __init__(self, hand_type:str, hand_joint:str, can:str, 
                 speed=None, save_log:bool=False,
                 start_gui:bool=False, init_pose:List[float]=None):
        if speed is None:
            speed = [250] * 10
        self.hand_type = hand_type
        self.hand_joint = hand_joint
        self.can = can
        self.speed = speed
        self.save_key = save_log
        self.start_gui = start_gui
        self.init_pose = init_pose
        

        if save_log:
            self.save_thread = threading.Thread(target=self.SaveLog)
            self.save_thread.daemon = True
            self.log = "/home/b600/robot/robot_control/log/"+self.hand_type+".txt"


        print(f"手类型: {self.hand_type}, 型号: {self.hand_joint}即将启动\n")
        if start_gui:
            self.hand = LinkHand_Gui(
                hand_type=self.hand_type,
                can=self.can,
                init_pose=self.init_pose,
            )
        else:
            self.hand = LinkerHandApi(
                hand_joint=self.hand_joint,
                hand_type=self.hand_type,
                can=self.can
                )
            self.hand.set_speed(speed=self.speed)
            if save_log:
                self.save_thread.start()

    def move(self,pose_list:List[List[float]]):
        for pose in pose_list:
            if point_check(pose,self.hand_joint):
                self.hand.finger_move(pose=pose)
                time.sleep(2)
            else:
                raise ValueError("参数输入错误")

    def set_speed(self,speed:List[float]):
        self.speed = speed
        self.hand.set_speed(speed=self.speed)
    
    def SaveLog(self):
        seq = 0   # 模拟递增的 seq
        with open(self.log, "w", encoding="utf-8") as f:
            while True:
                try:
                    joint_values = self.hand.get_state()
                except Exception as e:
                    # 避免线程被异常杀死
                    f.write(
                        f"seq: {seq}\n",
                        f"error: {repr(e)}\n\n"
                    )
                    f.flush()
                    time.sleep(0.1)
                    seq += 1
                    continue

                if joint_values is not None and len(joint_values) == 10:
                    # 获取系统时间作为 stamp
                    t = time.time()
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
                time.sleep(0.01)


