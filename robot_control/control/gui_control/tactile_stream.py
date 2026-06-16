# tactile_stream.py
import serial
import threading
import time

SERIAL_PORTS = ['COM3', 'COM4', 'COM5', 'COM6']
BAUDRATE = 115200
TIMEOUT = 0.1

ROWS = 12
COLS = 8
NUM_PRESSURE_POINTS = ROWS * COLS

MAGIC = bytes.fromhex("A5 5A 01 66 00 01")  # 固定同步标记（6字节）
MAGIC_BACK = 2                              # magic 前面还有2字节变化字段
HEADER_LEN = MAGIC_BACK + len(MAGIC)        # 8
PAYLOAD_LEN = 12 * 8                        # 96
FRAME_LEN = HEADER_LEN + PAYLOAD_LEN        # 104


port_to_idx = {port: i for i, port in enumerate(SERIAL_PORTS)}
latest_data = [None] * len(SERIAL_PORTS)
latest_lock = threading.Lock()
_started = False
prev_frame = [None] * len(SERIAL_PORTS)

def _serial_worker(port):
    try:
        ser = serial.Serial(port, BAUDRATE, timeout=TIMEOUT)
        print(f"[{port}] opened")
        ser.reset_input_buffer()
    except Exception:
        print(f"[{port}] cannot open. thread exit.")
        return

    buf = bytearray()

    while True:
        chunk = ser.read(512)
        if chunk:
            buf += chunk
        else:
            time.sleep(0.001)
            continue

        while True:
            pos = buf.find(MAGIC)
            if pos < MAGIC_BACK:
                # 不够定位到帧起点，或没找到 magic
                if len(buf) > 4096:
                    del buf[:-HEADER_LEN]   # 保留少量尾巴用于跨chunk匹配
                break

            start = pos - MAGIC_BACK
            if len(buf) < start + FRAME_LEN:
                # 还没收到完整一帧
                break

            frame = bytes(buf[start:start + FRAME_LEN])
            del buf[:start + FRAME_LEN]

            payload = frame[HEADER_LEN:HEADER_LEN + PAYLOAD_LEN]  # 96 bytes
            pressure = list(payload)

            idx = port_to_idx[port]
            with latest_lock:
                latest_data[idx] = pressure


def start():
    global _started
    if _started:
        return
    _started = True
    for port in SERIAL_PORTS:
        t = threading.Thread(target=_serial_worker, args=(port,), daemon=True)
        t.start()

# def get_latest_vector(port="COM4"):
#     idx = port_to_idx.get(port, 0)
#     with latest_lock:
#         v = latest_data[idx]
#         return None if v is None else list(v)  # copy

# def get_latest_vector_by_index(idx=0):
#     with latest_lock:
#         v = latest_data[idx]
#         return None if v is None else list(v)

def get_latest_all_vectors(n=5):
    with latest_lock:
        snap = latest_data[:n]
        return [None if v is None else list(v) for v in snap]
