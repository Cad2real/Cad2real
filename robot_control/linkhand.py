#!/usr/bin/env python3
import sys,os,time,argparse
target_dir = os.path.abspath('./control/linker_hand_python_sdk')
sys.path.append(target_dir)
from control.linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
from control.linkhand_control import linkhand_control

def main():
    hand_joint = "L10"
    hand_type = "left"
    # hand_type = "right"
    if hand_type == "left":
        can = "can0"
    elif hand_type == "right":
        can = "can1"

    hand = linkhand_control(hand_joint=hand_joint, hand_type=hand_type, can=can)
    # 设置速度
    # 手指姿态数据
    # "拇指根部", "拇指侧摆","食指根部", "中指根部", "无名根部","小指根部","食指侧摆","无名侧摆","小指侧摆","拇指旋转"
    
    # pose = [188, 0, 218, 222, 224, 238, 0, 88, 108, 170]
    # pose = [200, 0, 196, 188, 0, 0, 0, 255, 255, 235]
    # pose = [255, 0, 255, 255, 255, 255, 101, 88, 108, 170]

#     poses=[
#     [28.84, 20.43, 3.00, 5.00, 6.98, 6.98, 0.00, -0.00, -0.00, 19.48]
# ]


    
    poses=[[220, 0, 220, 167, 0, 0, 0, 0, 0, 240]]
    # poses=[[160, 0, 160, 167, 0, 0, 0, 0, 0, 240]]


    # poses=[[255, 0, 255, 255, 255, 255, 255, 255, 255, 255]]

    # poses=[[210, 0, 180, 255, 0, 0, 0, 0, 0, 240]]

    # poses=[[255, 0, 255, 255, 0, 0, 0, 0, 0, 240]]
    # poses=[[88, 233, 0, 0, 0, 0, 0, 0, 0, 255]]

    poses=[[255, 0, 255, 255, 255, 255, 255, 255, 255, 240]]
    # poses=[[255, 0, 0, 0, 0, 0, 255, 255, 255, 255]]
    # poses=[[55, 255, 0, 0, 0, 0, 255, 255, 255, 255]]
    


    hand.move(pose_list=poses)
    time.sleep(2)


if __name__ == "__main__":
    # python3 linker_hand_fist.py --hand_type left --hand_joint L10 --can=can0
    main()