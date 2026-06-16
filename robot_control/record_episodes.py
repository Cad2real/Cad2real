import threading
import time
import cv2
import numpy as np
import argparse

# 导入所有控制器和录制器模块
from camera_control import MultiCameraController
from robot_control import RobotController
from evs_control import EVSController
from dataset_recorder import DataRecorder

# 控制/录制频率 (50Hz)
DT = 0.02

class SequenceRecordingSystem:
    def __init__(self, episode_idx: int, dataset_dir: str = 'act_datasets'):
        print("正在初始化硬件控制器...")
        self.camera = MultiCameraController()
        self.robot = RobotController("192.168.5.1")
        self.evs = EVSController("/dev/ttyUSB0")
        
        self.camera_names = ["RealSense_1", "RealSense_2", "C920"]
        self.recorder = DataRecorder(dataset_dir=dataset_dir, episode_idx=episode_idx, camera_names=self.camera_names)

        self.positions = {
            'initial': [57.241, 15.795, 58.348, 108.733, 59.23, -180.551],
            'target1': [42.307, 39.83, 42.029, 101.732, 44.314, -181.628],
'target2': [42.305, 39.231, 50.427, 93.932, 44.314, -181.628],
'target3': [48.535, 33.448, 36.993, 112.926, 48.247, -181.3],
'target4': [104.723, 44.913, 30.274, 108.336, 134.247, -176.718],
'target5': [104.72, 42.805, 47.139, 93.584, 134.247, -176.718],
        }
        self.sequence = [
            # {'type': 'move', 'target': self.positions['initial'], 'desc': 'Moving to Initial'},
            {'type': 'move', 'target': self.positions['target1'], 'desc': 'Moving to Target 1'},
            {'type': 'move', 'target': self.positions['target2'], 'desc': 'Moving to Target 2'},
            {'type': 'wait', 'duration': 2, 'desc': 'Waiting for suction'},
            {'type': 'suction_on', 'desc': 'Suction ON'},
            {'type': 'wait', 'duration': 2, 'desc': 'Waiting for suction'},
            {'type': 'move', 'target': self.positions['target1'], 'desc': 'Moving to Target 1'},
            {'type': 'move', 'target': self.positions['target3'], 'desc': 'Moving to Target 3'},
            {'type': 'move', 'target': self.positions['target4'], 'desc': 'Moving to Target 4'},
            {'type': 'move', 'target': self.positions['target5'], 'desc': 'Moving to Target 5'},
            {'type': 'wait', 'duration': 2, 'desc': 'Waiting for release'},
            {'type': 'suction_off', 'desc': 'Suction OFF'},
            {'type': 'wait', 'duration': 2, 'desc': 'Waiting for release'},
            {'type': 'move', 'target': self.positions['target4'], 'desc': 'Moving to Target 4'},
            {'type': 'move', 'target': self.positions['initial'], 'desc': 'Returning to Initial'}
        ]

        self.running = False
        self.suction_on = False
        self.is_robot_moving = False
        self.current_step_index = 0

    def init_all_devices(self):
        print("--- 开始初始化相机 ---")
        self.camera.initialize_realsense_camera("RealSense_1", "136522073426")
        self.camera.initialize_realsense_camera("RealSense_2", "136622074052")
        self.camera.initialize_c920_camera("C920", 16, 200) # 请确保ID正确
        self.camera.start_all_cameras()
        time.sleep(2)

        print("--- 开始初始化机械臂 ---")
        if not self.robot.connect(): raise ConnectionError("机械臂连接失败")
        if not self.robot.enable(): raise RuntimeError("机械臂使能失败")

        # --- 新增：设置全局速度为30% ---
        # 您可以将30修改为您需要的任何值 (1-100)
        print("设置机械臂全局速度为 30%")
        self.robot.dashboard.SpeedFactor(30)
        # --- 结束新增 ---

        print("--- 开始初始化吸盘 ---")
        if not self.evs.connect(): raise ConnectionError("吸盘连接失败")
        if not self.evs.enable(): raise RuntimeError("吸盘使能失败")
        
        print("\n所有设备初始化完成！")

    def camera_display_loop(self):
        """
        在后台线程中运行，将所有摄像头画面拼接成2x2网格在一个窗口中显示。
        """
        display_width = 640
        display_height = 360
        
        # --- 关键修改：创建并设置窗口大小 ---
        window_name = "Combined Camera View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 使窗口可由用户调整大小
        cv2.resizeWindow(window_name, 960, 540)          # 设置一个960x540的初始大小
        # --- 结束修改 ---

        while self.running:
            frames = self.camera.get_latest_frames_copy()
            
            rs1_frame = frames.get("RealSense_1")
            rs2_frame = frames.get("RealSense_2")

            if rs1_frame is None:
                rs1_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                cv2.putText(rs1_frame, "RealSense_1 No Signal", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                rs1_frame = cv2.resize(rs1_frame, (display_width, display_height))

            if rs2_frame is None:
                rs2_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                cv2.putText(rs2_frame, "RealSense_2 No Signal", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                rs2_frame = cv2.resize(rs2_frame, (display_width, display_height))
            
            top_row = cv2.hconcat([rs1_frame, rs2_frame])
            
            c920_frame = frames.get("C920")
            
            if c920_frame is None:
                c920_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                cv2.putText(c920_frame, "C920 No Signal", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                c920_frame = cv2.resize(c920_frame, (display_width, display_height))

            step_info = self.sequence[self.current_step_index]['desc'] if self.current_step_index < len(self.sequence) else "Finished"
            step_text = f"Step {self.current_step_index + 1}/{len(self.sequence)}: {step_info}"
            cv2.putText(c920_frame, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            suc_text = f"Suction: {'ON' if self.suction_on else 'OFF'}"
            cv2.putText(c920_frame, suc_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(c920_frame, "Press ESC to Quit", (10, display_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            blank_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            bottom_row = cv2.hconcat([c920_frame, blank_frame])
            
            combined_view = cv2.vconcat([top_row, bottom_row])
            
            cv2.imshow(window_name, combined_view)

            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
                break
    
    def run_and_record_sequence(self):
        print("--- 开始执行任务序列并录制 ---")
        
        while self.robot.get_robot_mode() != 5 and self.running:
            print("等待机器人进入空闲状态 (RobotMode 5)...")
            time.sleep(0.5)

        while self.running and self.current_step_index < len(self.sequence):
            start_time = time.time()
            
            robot_state = self.robot.get_current_state()
            image_frames = self.camera.get_latest_frames_copy()
            self.suction_on = self.evs.is_suction_on()

            qpos = list(robot_state['q_actual']) + [1.0 if self.suction_on else -1.0]
            qvel = list(robot_state['qd_actual']) + [0.0]
            effort = list(robot_state['i_actual']) + [0.0]

            obs = {
                'qpos': np.array(qpos, dtype=np.float64),
                'qvel': np.array(qvel, dtype=np.float64),
                'effort': np.array(effort, dtype=np.float64),
                'images': {k: image_frames.get(k) for k in self.camera_names}
            }
            self.recorder.add_step(obs)

            robot_mode = self.robot.get_robot_mode()
            
            if not self.is_robot_moving and robot_mode == 5:
                step = self.sequence[self.current_step_index]
                print(f"执行步骤: {step['desc']}")
                
                if step['type'] == 'move':
                    self.robot.move_to_joint_position(step['target'], wait=False)
                    self.is_robot_moving = True
                else:
                    if step['type'] == 'suction_on': self.evs.start_suction()
                    elif step['type'] == 'suction_off': self.evs.stop_suction()
                    elif step['type'] == 'wait': time.sleep(step['duration'])
                    self.current_step_index += 1
            
            elif self.is_robot_moving and robot_mode == 5:
                print("移动完成。")
                self.is_robot_moving = False
                self.current_step_index += 1

            time.sleep(max(0, DT - (time.time() - start_time)))

        print("--- 任务序列执行完毕 ---")
        self.running = False

    def run(self):
        try:
            self.init_all_devices()
            self.running = True
            
            display_thread = threading.Thread(target=self.camera_display_loop, daemon=True)
            display_thread.start()
            
            time.sleep(1)
            self.run_and_record_sequence()
            
        except (KeyboardInterrupt, SystemExit):
            print("\n用户中断或程序退出。")
        except Exception as e:
            print(f"\n程序运行中发生严重错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("--- 开始清理和保存 ---")
            self.running = False
            if self.recorder.data_count > 0:
                self.recorder.save_to_hdf5()
            self.shutdown()

    def shutdown(self):
        print("正在关闭所有设备...")
        if self.evs: self.evs.cleanup()
        if self.robot: self.robot.disconnect()
        if self.camera: self.camera.stop_all_cameras()
        cv2.destroyAllWindows()
        print("系统已安全关闭。")

def main():
    parser = argparse.ArgumentParser(description="【最终版】执行固定序列，录制ACT数据集，并拼接显示摄像头。")
    parser.add_argument('--episode_idx', action='store', type=int, help='要录制的片段索引 (例如: 0, 1, 2...)', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='保存数据集的目录', default='act_datasets')
    args = parser.parse_args()
    
    system = SequenceRecordingSystem(episode_idx=args.episode_idx, dataset_dir=args.dataset_dir)
    system.run()

if __name__ == '__main__':
    main()