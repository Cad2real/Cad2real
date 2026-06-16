#!/usr/bin/env python3
import sys, os, time
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(target_dir)

from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml
from LinkerHand.utils.init_linker_hand import InitLinkerHand
from LinkerHand.utils.color_msg import ColorMsg

class GetState:
    def __init__(self, hand_joint="L10", hand_type="left", position=None):
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.hand = LinkerHandApi(hand_joint=self.hand_joint, hand_type=self.hand_type)
        
        if position is not None:
            self.position = position
            self.set_position()
        else:
            ColorMsg("未提供目标位置参数，仅获取当前位置", "yellow")

        self.get_state()

    def set_position(self):
        if self.hand_joint == "L7":
            if len(self.position) == 5:
                p = self.position + [100, 100]
            else:
                p = self.position
            self.hand.finger_move(pose=p)
        else:
            self.hand.finger_move(pose=self.position)
        time.sleep(0.01)
        ColorMsg(msg=f"Set position: {self.position}", color='green')

    def get_state(self):
        state = self.hand.get_state()
        print("当前关节状态:")
        print(state)
        time.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='获取灵巧手当前关节状态，可选设置目标位置')
    parser.add_argument('--hand_joint', type=str, default='L10', help='手指关节类型，默认是L10')
    parser.add_argument('--hand_type', type=str, default='left', help='手的类型，默认是左手')
    parser.add_argument('--position', nargs='+', type=int, help='目标位置（可选）')

    args = parser.parse_args()
    GetState(hand_joint=args.hand_joint, hand_type=args.hand_type, position=args.position)
