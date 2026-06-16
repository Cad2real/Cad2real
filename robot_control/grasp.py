import cv2
import threading
import time
import queue
import numpy as np


CONFIG = {
    "USE_MOCK_CAMERA": False,   
    "USE_MOCK_GRIPPER": False,  
    "CAMERA_ID": 2,             
}

try:
    from control.OminiPicker import OminiPicker
    HAS_HARDWARE_LIB = True
except ImportError:
    HAS_HARDWARE_LIB = False
    if not CONFIG["USE_MOCK_GRIPPER"]:
        print("⚠️ 警告: 未找到 OminiPicker 库，强制切换为模拟夹爪模式")
        CONFIG["USE_MOCK_GRIPPER"] = True

class MockCamera:
    """模拟相机：生成带噪点的画面，用于测试图像流程"""
    def __init__(self):
        self.running = True
        # 创建一个黑底图像
        self.base_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def start(self):
        print("[Mock] 模拟相机已启动")
        return self

    def get_frame(self):
        if not self.running:
            return False, None
        
        # 制造一点随机噪点，让你看出画面在刷新
        frame = self.base_frame.copy()
        noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # 在画面上写字
        cv2.putText(frame, "MOCK CAMERA MODE", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return True, frame

    def stop(self):
        self.running = False
        print("[Mock] 模拟相机已停止")

class MockGripper:
    """模拟夹爪：只打印，不动作"""
    def __init__(self):
        pass

    def start(self):
        print("[Mock] 模拟夹爪已就绪 (无硬件连接)")
        return self

    def send_command(self, value):
        # 仅仅打印指令，不发送给硬件
        print(f"👉 [Mock] 假装执行夹爪动作: {value}")

    def stop(self):
        print("[Mock] 模拟夹爪已停止")


class CameraThread:
    def __init__(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头 ID: {camera_id}")
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        print(f"[Real] 真实摄像头 (ID: {int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))}) 已启动")
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            if self.ret:
                return True, self.frame.copy()
            else:
                return False, None

    def stop(self):
        self.running = False
        self.cap.release()
        print("[Real] 真实摄像头已释放")

class GripperWorker:
    def __init__(self):
        self.grasp_queue = queue.Queue()
        self.running = True
        print("[Real] 正在连接真实夹爪...")
        self.picker = OminiPicker(password=3)
        print("[Real] 夹爪连接成功")

    def start(self):
        t = threading.Thread(target=self.process_queue, args=())
        t.daemon = True
        t.start()
        return self

    def process_queue(self):
        while self.running:
            try:
                val = self.grasp_queue.get(timeout=1)
                print(f"[Real] 硬件执行: {val}")
                self.picker(val) 
                self.grasp_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Real] 硬件错误: {e}")

    def send_command(self, value):
        self.grasp_queue.put(value)

    def stop(self):
        self.running = False
        print("[Real] 夹爪线程停止")


def main():
    print("=== 系统初始化 ===")
    
    # --- 初始化夹爪 ---
    if CONFIG["USE_MOCK_GRIPPER"]:
        gripper = MockGripper().start()
    else:
        try:
            gripper = GripperWorker().start()
        except Exception as e:
            print(f"❌ 真实夹爪连接失败: {e}")
            print("   -> 自动切换回模拟夹爪")
            gripper = MockGripper().start()

    # --- 初始化相机 ---
    if CONFIG["USE_MOCK_CAMERA"]:
        cam = MockCamera().start()
    else:
        try:
            cam = CameraThread(camera_id=CONFIG["CAMERA_ID"]).start()
        except Exception as e:
            print(f"❌ 摄像头打开失败: {e}")
            print("   -> 自动切换回模拟相机")
            cam = MockCamera().start()

    print("\n✅ 系统运行中")
    print("⌨️  按 'g' 抓取 | 按 'r' 释放 | 按 'q' 退出")

    while True:
        ret, frame = cam.get_frame()

        if not ret:
            print("无法获取图像帧")
            time.sleep(0.1)
            continue

        # 在屏幕上显示当前模式
        cam_mode = "MOCK" if CONFIG["USE_MOCK_CAMERA"] else "REAL"
        grip_mode = "MOCK" if CONFIG["USE_MOCK_GRIPPER"] else "REAL"
        
        info_text = f"Cam: {cam_mode} | Grip: {grip_mode}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "g: Grasp | r: Release", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Control Panel', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            gripper.send_command(0)
        elif key == ord('d'):
            gripper.send_command(70) 
        elif key == ord('r'):
            gripper.send_command(255)

    cam.stop()
    gripper.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()