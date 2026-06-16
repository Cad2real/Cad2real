from encodings.punycode import T
import threading
from time import sleep
import time
from control.dobot_contorl import dobot_control
from control.linkhand_control import linkhand_control
from typing import List
import config

class controller:
    def __init__(self, move_pose:List,
                 LeftRobot:dobot_control, RightRobot:dobot_control,
                 LeftHand:linkhand_control, RightHand:linkhand_control,
                 thread_set:bool=False,log_set:bool=False, zero_set:bool=False):
        self.move_pose = move_pose
        self.LeftRobot = LeftRobot
        self.RightRobot = RightRobot
        self.LeftHand = LeftHand
        self.RightHand = RightHand

        self.log_set = log_set
        self.zero_set = zero_set
        self.thread_set = thread_set
        self.left_wait = threading.Event()
        self.right_wait = threading.Event()
        self.left_blocked = False
        self.right_blocked = False
        self.lock = threading.Lock()
        self.LeftPose = []
        self.RightPose = []

        if self.thread_set:
            self.command_divide()
            if self.zero_set:
                self.LeftPose = self.LeftPose[0:4]
                self.LeftPose.append(0)
                self.RightPose = self.RightPose[0:4]
                self.RightPose.append(0)

            self.left_thread = threading.Thread(target=self.LeftThread)
            self.left_thread.start()
            

            self.right_thread = threading.Thread(target=self.RightThread)
            self.right_thread.start()
            
            self.wait_monitor = threading.Thread(target=self.monitor_thread)
            self.wait_monitor.daemon = True
            self.wait_monitor.start()

        else:
            self.pos_move(self.move_pose)
            pass

    def command_divide(self):
        i = 0
        while True:
            index = self.move_pose[i]
            if index == 2 or index == 3:
                self.LeftPose.append(index)
                self.LeftPose.append(self.move_pose[i+1])
                i += 2
            elif index == 1 or index == 4:
                self.RightPose.append(index)
                self.RightPose.append(self.move_pose[i+1])
                i += 2
            elif index == 5:
                if self.move_pose[i+1] == "left":
                    self.LeftPose.append(index)
                    self.LeftPose.append(self.move_pose[i+1])
                elif self.move_pose[i+1] == "right":
                    self.RightPose.append(index)
                    self.RightPose.append(self.move_pose[i+1])
                i += 2
            elif index == -1:
                self.LeftPose.append(index)
                self.RightPose.append(index)
                i += 1
            elif index == 0:
                self.LeftPose.append(index)
                self.RightPose.append(index)
                break

    def pos_move(self,cmd):
        i = 0
        skip_key = False
        while True:
            index = cmd[i]
            if index == -1:
                skip_key = not skip_key
                i += 1
                continue
            
            if skip_key:
                i += 2
                continue


            if index == 1:
                self.RightRobot.move(cmd[i + 1])
                i += 2
            elif index == 2:
                self.LeftRobot.move(cmd[i + 1])
                i += 2
            elif index == 3:
                self.LeftHand.move(cmd[i + 1])
                i += 2
            elif index == 4:
                self.RightHand.move(cmd[i + 1])
                i += 2
            elif index == 5:
                if self.thread_set:
                    if cmd[i + 1] == "left":
                        with self.lock:
                            self.left_blocked = True
                        self.left_wait.wait()
                        self.left_wait.clear()
                    elif cmd[i + 1] == "right":
                        with self.lock:
                            self.right_blocked = True
                        self.right_wait.wait()
                        self.right_wait.clear()
                    i += 2
                else:
                    i += 2
            elif index == 0:
                if self.log_set:
                    time.sleep(60)
                print('\nover!\n')
                break

    def ThreadWait(self):
        self.left_thread.join()
        self.right_thread.join()

    def LeftThread(self):
        self.pos_move(self.LeftPose)

    def RightThread(self):
        self.pos_move(self.RightPose)

    def monitor_thread(self):
        while True:
            with self.lock:
                if self.left_blocked and self.right_blocked:
                    self.left_wait.set()
                    self.right_wait.set()
                    # 重置状态，防止重复设置
                    self.left_blocked = False
                    self.right_blocked = False
            sleep(0.1)
    
    def get_feed(self):
        while True:
            try:
                with self.lock:
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


if __name__ == '__main__':
    start_threading = True
    save_log = False
    # save_log = True

    back2zero = False
    back2zero = True

    command = config.exp_potato

    nova2 = dobot_control(connect_ip="192.168.5.1", save_log=save_log)
    nova5 = dobot_control(connect_ip="192.168.5.2", save_log=save_log)
    Lhand = linkhand_control(hand_type="left", hand_joint="L10", can="can0", save_log=save_log)
    Rhand = linkhand_control(hand_type="right", hand_joint="L10", can="can1", save_log=save_log)

    print('\n\n\n\nstart\n')
    ctl = controller(move_pose=command, thread_set=start_threading, log_set = save_log, zero_set=back2zero,
               LeftRobot=nova5, RightRobot=nova2,
               LeftHand=Lhand, RightHand=Rhand)
    
    if start_threading:
        ctl.ThreadWait()
    