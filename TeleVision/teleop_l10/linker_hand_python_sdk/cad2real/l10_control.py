#!/usr/bin/env python3
import sys,os,time,argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(target_dir)
from LinkerHand.linker_hand_api import LinkerHandApi

HAND_TYPE = "right"
HAND_JOINT = "L10"
CAN_INTERFACE = "can0"

def main():
    print(f"手类型: {HAND_TYPE}, 关节: {HAND_JOINT}, 接口: {CAN_INTERFACE}")
    
    hand = LinkerHandApi(hand_joint=HAND_JOINT, hand_type=HAND_TYPE, can=CAN_INTERFACE)
    hand.set_speed(speed=[120,250,250,250,250])
    
    poses = [
        [35.0,140.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,30.0],  # 拇指弯曲
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # 张开
        [255.0,70.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0],   # 食指弯曲
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # 张开
        [255.0,70.0,255.0,0.0,255.0,255.0,255.0,255.0,255.0,255.0],   # 中指弯曲
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # 张开
        [255.0,70.0,255.0,255.0,0.0,255.0,255.0,255.0,255.0,255.0],   # 无名指弯曲
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # 张开
        [255.0,70.0,255.0,255.0,255.0,0.0,255.0,255.0,255.0,255.0],   # 小拇指弯曲
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # 张开
    ]
    
    # ✅ 只执行一遍，不循环
    for pose in poses:
        hand.finger_move(pose=pose)
        time.sleep(1)

if __name__ == "__main__":
    main()