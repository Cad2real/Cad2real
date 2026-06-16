from PyQt5.QtWidgets import QMainWindow, QSplitter, QApplication,QMessageBox,QPushButton
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
import yaml, os, sys,time,json
from views.left_view import LeftView
from views.right_view import RightView
from views.wave_form_plot import WaveformPlot
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(target_dir)
from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml
from LinkerHand.utils.color_msg import ColorMsg
'''
LinkerHand图形控制
'''

import serial
import threading
from JUQ_imu_reader import JQFrameParser, SENSOR_LH, SENSOR_RH

# === JUQ 串口参数（TODO：按你实际改）===
JUQ_PORT = "COM10"
JUQ_BAUD = 921600
JUQ_TIMEOUT = 0.05

# === 选左手/右手（GUI 读 yaml 只支持单手，这里按 hand_type 选）===
# 左手: SENSOR_LH, 右手: SENSOR_RH

# === 你现在用的 5 路弯曲信号索引（0-based）===
JUQ_SENSOR_IDX = {
    "thumb":  47-1,  # sensor_256[210-1]
    "index":  44-1,  # sensor_256[213-1]
    "middle": 41-1,  # sensor_256[216-1]
    "ring":   38-1,  # sensor_256[219-1]
    "pinky":  35-1,  # sensor_256[222-1]
    "index_thumb_middle_1": 110-1,  # 147-1
    "index_thumb_middle_2": 78-1,  # 179-1
    "index_thumb_middle_3": 109-1,  # 148-1
    "thumb_roll": 109-1,  # 148-1
}

# 矩侨手套压电传感器弯曲范围（右手）
JUQ_CALIB = {
    "thumb":  {"RAW_MIN":  21, "RAW_MAX": 137},  # TODO
    "index":  {"RAW_MIN":  85, "RAW_MAX": 141},  # TODO
    "middle": {"RAW_MIN":  92, "RAW_MAX": 150},  # TODO
    "ring":   {"RAW_MIN":  92, "RAW_MAX": 165},  # TODO
    "pinky":  {"RAW_MIN":  81, "RAW_MAX": 144},  # TODO
    "thumb_roll": {"RAW_MIN": 0.0, "RAW_MAX": 20.0},
}

# ===== thumb_cmc_yaw 三路信号标定（TODO：实测后改）=====
CALIB_YAW_RAW = {
    "index_thumb_middle_1": {"RAW_MIN": 0.0, "RAW_MAX": 34.0},
    "index_thumb_middle_2": {"RAW_MIN": 0.0, "RAW_MAX": 43.0},
    "index_thumb_middle_3": {"RAW_MIN": 0.0, "RAW_MAX": 5.0},

}

# ===== 权重可调 =====
YAW_WEIGHTS = {
    "index_thumb_middle_1": 0.2,
    "index_thumb_middle_2": 0.8,
    "index_thumb_middle_3": 0.0,
}

def weighted_yaw01(raw1, raw2, raw3) -> float:
    def norm(v, mn, mx):
        if mx <= mn:
            return 0.0
        t = (v - mn) / (mx - mn)
        return _clamp(t, 0.0, 1.0)

    n1 = norm(raw1, CALIB_YAW_RAW["index_thumb_middle_1"]["RAW_MIN"], CALIB_YAW_RAW["index_thumb_middle_1"]["RAW_MAX"])
    n2 = norm(raw2, CALIB_YAW_RAW["index_thumb_middle_2"]["RAW_MIN"], CALIB_YAW_RAW["index_thumb_middle_2"]["RAW_MAX"])
    n3 = norm(raw3, CALIB_YAW_RAW["index_thumb_middle_3"]["RAW_MIN"], CALIB_YAW_RAW["index_thumb_middle_3"]["RAW_MAX"])

    w1 = YAW_WEIGHTS["index_thumb_middle_1"]
    w2 = YAW_WEIGHTS["index_thumb_middle_2"]
    w3 = YAW_WEIGHTS["index_thumb_middle_3"]
    wsum = (w1 + w2 + w3) if (w1 + w2 + w3) > 1e-9 else 1.0
    return (w1*n1 + w2*n2 + w3*n3) / wsum

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def map_raw_to_0_255(raw, raw_min, raw_max, invert=False):
    """JUQ raw -> 0..255（给真实手 L10/L7/L20 等）"""
    if raw_max <= raw_min:
        return 0
    t = (raw - raw_min) / (raw_max - raw_min)
    t = _clamp(t, 0.0, 1.0)
    if invert:
        t = 1.0 - t
    return int(round(255 * t))

class JUQShared:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw = {k: None for k in JUQ_SENSOR_IDX.keys()}

    def update_from_sensor256(self, sensor_256: bytes):
        with self.lock:
            for k, idx in JUQ_SENSOR_IDX.items():
                if 0 <= idx < len(sensor_256):
                    self.raw[k] = sensor_256[idx]

    def snapshot(self):
        with self.lock:
            return dict(self.raw)

def juq_reader_thread(shared: JUQShared, stop_evt: threading.Event, want_type: int):
    ser = serial.Serial(JUQ_PORT, baudrate=JUQ_BAUD, timeout=JUQ_TIMEOUT)
    parser = JQFrameParser(want_sensor_type=want_type)  # 过滤 LH/RH :contentReference[oaicite:1]{index=1}
    try:
        while not stop_evt.is_set():
            chunk = ser.read(256)
            if not chunk:
                continue
            packets = parser.feed(chunk)
            for order, stype, payload in packets:
                assembled = parser.on_packet(order, stype, payload)
                if assembled is None:
                    continue
                sensor_256, _imu_16 = assembled
                shared.update_from_sensor256(sensor_256)
    finally:
        try:
            ser.close()
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self._init_hand_joint()
        self.api = LinkerHandApi(hand_joint=self.hand_joint,hand_type=self.hand_type)
        self.touch_type = -1
        self._init_gui_view()

        self.current_pose = list(self.init_pos)    # 保存所有关节的当前目标值

        self.joint_name_to_idx = {name: i for i, name in enumerate(self.joint_name)}
        print("[INFO] joint_name_to_idx built, len =", len(self.joint_name_to_idx))

                # ===== JUQ -> 真实手 自动控制（不阻塞 GUI）=====
        self.juq_shared = JUQShared()
        self.juq_stop = threading.Event()

        want_type = SENSOR_LH if self.hand_type == "left" else SENSOR_RH
        self.juq_th = threading.Thread(
            target=juq_reader_thread,
            args=(self.juq_shared, self.juq_stop, want_type),
            daemon=True
        )
        self.juq_th.start()

        # 用 QTimer 定频输出到手（比“来一包就发一次”稳定很多，蓝牙也更顺）
        self.juq_timer = QTimer()
        self.juq_timer.timeout.connect(self._update_from_juq)
        self.juq_timer.start(100)  # 30ms ~33Hz；嫌慢可改 20ms

        self.juq_enabled = True

        if self.hand_joint == "L7":
            self.add_button_position = [255] * 7
            self.set_speed = [180,250,250,250,250,250,250]
            self.touch_type = self.api.get_touch_type()
            # if self.touch_type == 2:
            #     self._init_normal_force_plot(num_lines=6) # 法向压力波形图
            # else:
            #     self._init_normal_force_plot() # 法向压力波形图
            #     self._init_approach_inc_plot() # 接近感应波形图
        elif self.hand_joint == "L10":
            self.add_button_position = [255] * 10 # 记录添加按钮的位置
            self.set_speed(speed=[180,250,250,250,250])
            self.touch_type = self.api.get_touch_type()
            # if self.touch_type == 2:
            #     self._init_normal_force_plot(num_lines=6) # 法向压力波形图
            # else:
            #     self._init_normal_force_plot() # 法向压力波形图
            #     self._init_approach_inc_plot() # 接近感应波形图
        elif self.hand_joint == "L20":
            self.add_button_position = [255] * 20 # 记录添加按钮的位置
            self.set_speed(speed=[120,180,180,180,180])
            self._init_normal_force_plot() # 法向压力波形图
            self.touch_type = self.api.get_touch_type()
            if self.touch_type == 2:
                self._init_normal_force_plot(num_lines=6) # 法向压力波形图
            else:
                self._init_normal_force_plot() # 法向压力波形图
                self._init_approach_inc_plot() # 接近感应波形图
        elif self.hand_joint == "L21":
            self.add_button_position = [255] * 25
            self.set_speed(speed=[60,220,220,220,220])
            self._init_normal_force_plot() # 法向压力波形图
            self.touch_type = self.api.get_touch_type()
            if self.touch_type == 2:
                self._init_normal_force_plot(num_lines=6) # 法向压力波形图
            else:
                self._init_normal_force_plot() # 法向压力波形图
                self._init_approach_inc_plot() # 接近感应波形图
        elif self.hand_joint == "L25":
            self.add_button_position = [255] * 30 # 记录添加按钮的位置
            self.set_speed(speed=[60,250,250,250,250])



    def _init_hand_joint(self):
        self.yaml = LoadWriteYaml() # 初始化配置文件
        # 读取配置文件
        self.setting = self.yaml.load_setting_yaml()
        # 判断左手是否配置
        self.left_hand = False
        self.right_hand = False
        if self.setting['LINKER_HAND']['LEFT_HAND']['EXISTS'] == True:
            self.left_hand = True
        elif self.setting['LINKER_HAND']['RIGHT_HAND']['EXISTS'] == True:
            self.right_hand = True
        # gui控制只支持单手，这里进行左右手互斥
        if self.left_hand == True and self.right_hand == True:
            self.left_hand = True
            self.right_hand = False
        if self.left_hand == True:
            print("左手")
            self.hand_exists = True
            self.hand_joint = self.setting['LINKER_HAND']['LEFT_HAND']['JOINT']
            self.hand_type = "left"
        if self.right_hand == True:
            print("右手")
            self.hand_exists = True
            self.hand_joint = self.setting['LINKER_HAND']['RIGHT_HAND']['JOINT']
            self.hand_type = "right"
        
        self.init_pos = [255] * 10
        if self.hand_joint == "L25":
            # L25
            self.init_pos = [96, 255, 255, 255, 255, 150, 114, 151, 189, 255, 180, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
            self.joint_name = ["拇指根部", "食指根部", "中指根部", "无名指根部","小指根部","拇指侧摆","食指侧摆","中指侧摆","无名指侧摆","小指侧摆","拇指横摆","预留","预留","预留","预留","拇指中部","食指中部","中指中部","无名指中部","小指中部","拇指指尖","食指指尖","中指指尖","无名指指尖","小指指尖"]
        elif self.hand_joint == "L21":
            # L21
            self.init_pos = [96, 255, 255, 255, 255, 150, 114, 151, 189, 255, 180, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
            self.joint_name = ["拇指根部", "食指根部", "中指根部", "无名指根部","小指根部","拇指侧摆","食指侧摆","中指侧摆","无名指侧摆","小指侧摆","拇指横摆","预留","预留","预留","预留","拇指中部","预留","预留","预留","预留","拇指指尖","食指指尖","中指指尖","无名指指尖","小指指尖"]
        elif self.hand_joint == "L20":
            self.init_pos = [255,255,255,255,255,255,10,100,180,240,245,255,255,255,255,255,255,255,255,255]
            # L20
            self.joint_name = ["拇指根部", "食指根部", "中指根部", "无名指根部","小指根部","拇指侧摆","食指侧摆","中指侧摆","无名指侧摆","小指侧摆","拇指横摆","预留","预留","预留","预留","拇指尖部","食指末端","中指末端","无名指末端","小指末端"]
        elif self.hand_joint == "L10":
            # L10
            self.init_pos = [255] * 10
            self.joint_name = ["拇指根部", "拇指侧摆","食指根部", "中指根部", "无名指根部","小指根部","食指侧摆","无名指侧摆","小指侧摆","拇指旋转"]
        elif self.hand_joint == "L7":
            # L7
            self.init_pos = [250] * 7
            self.joint_name = ["大拇指弯曲", "大拇指横摆","食指弯曲", "中指弯曲", "无名指弯曲","小拇指弯曲","拇指旋转"]
        
    def _set_pose_if_exists(self, pose, joint_label, value_0_255):
        idx = self.joint_name_to_idx.get(joint_label, None)
        if idx is not None and 0 <= idx < len(pose):
            pose[idx] = int(value_0_255)

    def _update_from_juq(self):
        if not getattr(self, "juq_enabled", True):
            return

        raw = self.juq_shared.snapshot()
        if any(raw[k] is None for k in ("thumb","index","middle","ring","pinky","thumb_roll")):
            return

        # 1) 五路弯曲 -> 0..255（标定你之后填）
        thumb  = map_raw_to_0_255(raw["thumb"],  JUQ_CALIB["thumb"]["RAW_MIN"],  JUQ_CALIB["thumb"]["RAW_MAX"],  invert=True)
        index  = map_raw_to_0_255(raw["index"],  JUQ_CALIB["index"]["RAW_MIN"],  JUQ_CALIB["index"]["RAW_MAX"],  invert=True)
        middle = map_raw_to_0_255(raw["middle"], JUQ_CALIB["middle"]["RAW_MIN"], JUQ_CALIB["middle"]["RAW_MAX"], invert=True)
        ring   = map_raw_to_0_255(raw["ring"],   JUQ_CALIB["ring"]["RAW_MIN"],   JUQ_CALIB["ring"]["RAW_MAX"],   invert=True)
        pinky  = map_raw_to_0_255(raw["pinky"],  JUQ_CALIB["pinky"]["RAW_MIN"],  JUQ_CALIB["pinky"]["RAW_MAX"],  invert=True)
        thumb_roll = map_raw_to_0_255(raw["thumb_roll"],  JUQ_CALIB["thumb_roll"]["RAW_MIN"],  JUQ_CALIB["thumb_roll"]["RAW_MAX"],  invert=True)

        # 1.5) 计算 thumb_cmc_yaw（0~255）
        r1 = raw.get("index_thumb_middle_1", None)
        r2 = raw.get("index_thumb_middle_2", None)
        r3 = raw.get("index_thumb_middle_3", None)

        YAW_INVERT = True  # TODO: 如果方向反了就 True

        thumb_yaw = None
        if (r1 is not None) and (r2 is not None) and (r3 is not None):
            mix01 = weighted_yaw01(r1, r2, r3)     # 0~1
            if YAW_INVERT:
                mix01 = 1.0 - mix01
            thumb_yaw = int(round(255 * mix01))    # 映射到 0~255（真实手用这个标度）


        # 2) 构造 pose（长度由 init_pos 决定：L10=10，L20=20）
        # pose = list(self.init_pos)
        pose = list(self.current_pose)

        if self.hand_joint == "L10":
            # 你已有的 L10 写法（按你 L10 的 joint_name 顺序）
            pose[0] = thumb
            pose[2] = index
            pose[3] = middle
            pose[4] = ring
            pose[5] = pinky
            pose[1] = thumb_roll   # 拇指侧摆

            if thumb_yaw is not None:
                # L10 里这个关节名叫“拇指侧摆”
                self._set_pose_if_exists(pose, "拇指旋转", thumb_yaw)

        elif self.hand_joint == "L20":
            # ✅ L20：用“名字定位”，先只写每根手指的“弯曲主关节”
            # 下面这些 joint_label 必须和你的 yaml 里 joint_name 完全一致
            # ——你把 yaml 里的 L20 joint_name 列表贴我，我可以替你对齐成准确的名字
            self._set_pose_if_exists(pose, "拇指根部", thumb)
            self._set_pose_if_exists(pose, "食指根部", index)
            self._set_pose_if_exists(pose, "中指根部", middle)
            self._set_pose_if_exists(pose, "无名指根部", ring)
            self._set_pose_if_exists(pose, "小指根部", pinky)
            self._set_pose_if_exists(pose, "拇指侧摆", thumb_roll)

            if thumb_yaw is not None:
                self._set_pose_if_exists(pose, "拇指横摆", thumb_yaw)

            # 其它 15 个自由度先保持默认 init_pos（测试阶段最稳）
            # 以后你要加 yaw/侧摆/旋转，再在这里继续 _set_pose_if_exists

        else:
            return  # 其它型号先不动

        # 3) 发给真实手 + 同步滑条显示
        self.current_pose = list(pose)
        self.api.finger_move(pose=pose)  # :contentReference[oaicite:2]{index=2}

        # try:
        #     self.left_view.blockSignals(True)
        #     self.left_view.set_slider_values(values=pose)
        # finally:
        #     self.left_view.blockSignals(False)


    
    # 初始化窗口界面
    def _init_gui_view(self):
        if self.hand_type == "left":
            self.setWindowTitle(f"Linker_Hand:左手- {self.hand_joint} Control - Qt5 with ROS")
        else:
            self.setWindowTitle(f"Linker_Hand:右手- {self.hand_joint} Control - Qt5 with ROS")
        self.setGeometry(100, 100, 600, 800)
        # 创建分割线
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                width:1px;
                background-color: lightgray;
                margin: 15px 20px;
            }
        """)
        # 左侧滑动条界面
        self.left_view = LeftView(joint_name=self.joint_name, init_pos=self.init_pos)
        splitter.addWidget(self.left_view)
        self.left_view.slider_value_changed.connect(self.handle_slider_value_changed)
        # 右侧记录动作界面
        self.right_view = RightView(hand_joint=self.hand_joint, hand_type=self.hand_type)
        splitter.addWidget(self.right_view)
        # 接收到信号槽事件，这里用于记录动作序列更新滑动条数据
        self.right_view.handle_button_click.connect(self.handle_button_click)
        self.right_view.add_button_handle.connect(self.add_button_handle)
        splitter.setSizes([600, 450])

        # ===== 顶部控制栏：急停（停止手套控制）=====
        self.btn_juq_stop = QPushButton("急停：停止手套控制")
        self.btn_juq_stop.setCheckable(True)  # 按下保持状态
        self.btn_juq_stop.setStyleSheet("""
            QPushButton { font-size: 14px; padding: 8px; }
            QPushButton:checked { background-color: #d9534f; color: white; }
        """)
        self.btn_juq_stop.toggled.connect(self._toggle_juq_control)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_juq_stop)
        top_bar.addStretch(1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(top_bar)
        layout.addWidget(splitter)

        self.setCentralWidget(container)

    def _toggle_juq_control(self, checked: bool):
        # checked=True 表示按下急停
        self.juq_enabled = not checked
        if checked:
            self.btn_juq_stop.setText("已急停：手套控制已停止（再按恢复）")
        else:
            self.btn_juq_stop.setText("急停：停止手套控制")


    # 初始化波形图
    def _init_normal_force_plot(self,num_lines=5):
        # return
        # 初始化波形图
        self.normal_force_plot = WaveformPlot(num_lines=num_lines, labels=None,title="法向压力波形图")
        # 设置波形图位置
        self.normal_force_plot.setGeometry(700, 100, 800, 400)
        self.normal_force_plot.show()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_normal_force_plot)
        self.timer.start(50)
    def _init_approach_inc_plot(self):
        return
        # 初始化波形图
        self.approach_inc_plot = WaveformPlot(num_lines=5, labels=None,title="接近感应波形图")
        # 设置波形图位置
        self.approach_inc_plot.setGeometry(700, 600, 800, 400)
        self.approach_inc_plot.show()
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.update_approach_inc_plot)
        self.timer2.start(50)
    # 点击按钮后将动作数值写入yaml文件
    def handle_button_click(self,text):
        all_action = self.yaml.load_action_yaml(hand_type=self.hand_type,hand_joint=self.hand_joint)
        for index,pos in enumerate(all_action):
            if pos['ACTION_NAME'] == text:
                position = pos['POSITION']
                print(type(position))
        #print(f"动作名称:{text}, 动作数值:{action_pos}")
        # if text == self.la:
        #     self.left_view.set_slider_values(values=self.add_button_position)
        ColorMsg(msg=f"动作名称:{text}, 动作数值:{position}", color="green")
        self.api.finger_move(pose=position)
        self.left_view.set_slider_values(values=position)

    #点击添加按钮后将动作数值写入yaml文件
    def add_button_handle(self,text):
        self.add_button_position = self.left_view.get_slider_values()
        self.add_button_text = text
        self.yaml.write_to_yaml(action_name=text, action_pos=self.left_view.get_slider_values(),hand_joint=self.hand_joint,hand_type=self.hand_type)


    # 通过信号机制实时获取滑动条的当前值
    # def handle_slider_value_changed(self, slider_values):
    #     #print("实时获取滑动条的当前值:", slider_values)
    #     slider_values_list = []
    #     for key in slider_values:
    #         slider_values_list.append(slider_values[key])
    #     self.api.finger_move(pose=slider_values_list)
    def handle_slider_value_changed(self, slider_values):
        slider_values_list = []
        for key in slider_values:
            slider_values_list.append(slider_values[key])

        self.current_pose = list(slider_values_list)   # ✅ 记住用户手动调的其它关节
        self.api.finger_move(pose=self.current_pose)

    # 更新滑动条状态
    def update_label(self, index, value):
        self.left_view.labels[index].setText(f"{self.joint_name[index]}: {value}")

    # 更新法向压力波形图
    def update_normal_force_plot(self):
        import random
        #touch_type = self.api.get_touch_type()
        if self.touch_type == 2:
            values = self.api.get_touch()
        else:
            f = self.api.get_force()
            values = f[0]
        if values == None:
            pass
        else:
            self.normal_force_plot.update_data(values)
    # 更新接近感应波形图
    def update_approach_inc_plot(self):
        import random
        #touch_type = self.api.get_touch_type()
        if self.touch_type == 2:
            values = [0] * 5
        else:
            f = self.api.get_force()
            values = f[3]
        self.approach_inc_plot.update_data(values)

    def set_speed(self,speed=[180,250,250,250,250]):
        ColorMsg(msg=f"设置速度:{speed}", color="green")
        self.api.set_speed(speed)
    # 关闭窗口结束程序
    def closeEvent(self, event):
        """关闭窗口时停止线程并释放资源"""
        try:
            self.juq_timer.stop()
        except Exception:
            pass
        try:
            self.juq_stop.set()
        except Exception:
            pass

        # 关闭波形图（如果存在）
        try:
            self.normal_force_plot.close()
        except Exception:
            pass
        try:
            self.approach_inc_plot.close()
        except Exception:
            pass

        event.accept()

    

    

# 主程序运行
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())