import serial
import threading
import time
import struct
from collections import deque

HEADER = b"\xAA\x55\x03\x99"

# packet order -> payload length
PAYLOAD_LEN = {
    0x01: 128,  # first packet: first 128 sensor bytes
    0x02: 144,  # second packet: last 128 sensor bytes + 16 imu bytes
}

SENSOR_TYPE = {
    0x01: "LH",  # Left Hand
    0x02: "RH",
    0x03: "LF",
    0x04: "RF",
    0x05: "FB",  # (some docs use FB/WB, your pdf has WB in highfreq; keep both)
    0x06: "WB",
}

SENSOR_LH = 0x01
SENSOR_RH = 0x02
SENSOR_LF = 0x03
SENSOR_RF = 0x04
SENSOR_FB = 0x05
SENSOR_WB = 0x06


def decode_imu_quat_f32_le(imu16: bytes):
    # 小端 4 个 float32
    q = struct.unpack("<4f", imu16)
    return q  # (q0,q1,q2,q3) 顺序待确认

def dump_imu16(imu_16: bytes):
    b = bytes(imu_16)
    print("imu bytes hex:", b.hex(" "))

    le_i16 = struct.unpack("<8h", b)
    be_i16 = struct.unpack(">8h", b)
    le_f32 = struct.unpack("<4f", b)
    be_f32 = struct.unpack(">4f", b)

    print("LE int16:", le_i16)
    print("BE int16:", be_i16)
    print("LE float32:", le_f32)
    print("BE float32:", be_f32)

class JQFrameParser:
    """
    Parse stream:
      [AA 55 03 99] + [order 1B] + [type 1B] + [payload N bytes]
    Reassemble a full sensor frame when both packets (01 & 02) arrive for same sensor type.
    """
    def __init__(self, want_sensor_type=None):
        self.buf = bytearray()
        self.want_sensor_type = want_sensor_type

        # hold partial packets by sensor_type
        self.part1 = {}  # sensor_type -> bytes(128)
        self.part2 = {}  # sensor_type -> bytes(144)

    def feed(self, data: bytes):
        self.buf.extend(data)
        frames = []

        while True:
            # 1) find header
            idx = self.buf.find(HEADER)
            if idx < 0:
                # keep last few bytes to allow header spanning reads
                if len(self.buf) > len(HEADER) - 1:
                    self.buf = self.buf[-(len(HEADER) - 1):]
                return frames

            # discard bytes before header
            if idx > 0:
                del self.buf[:idx]

            # need at least header + order + type
            if len(self.buf) < len(HEADER) + 2:
                return frames

            order = self.buf[len(HEADER)]
            stype = self.buf[len(HEADER) + 1]

            # validate order
            if order not in PAYLOAD_LEN:
                # false header? shift by 1 and resync
                del self.buf[0:1]
                continue

            payload_len = PAYLOAD_LEN[order]
            full_len = len(HEADER) + 2 + payload_len

            if len(self.buf) < full_len:
                return frames  # wait more bytes

            payload = bytes(self.buf[len(HEADER) + 2: full_len])
            del self.buf[:full_len]

            frames.append((order, stype, payload))

        # unreachable

    def on_packet(self, order, stype, payload):
        # optionally filter only LH
        if self.want_sensor_type is not None and stype != self.want_sensor_type:
            return None

        if order == 0x01:
            self.part1[stype] = payload  # 128 bytes
        elif order == 0x02:
            self.part2[stype] = payload  # 144 bytes

        # if both arrived, assemble one "complete frame"
        if stype in self.part1 and stype in self.part2:
            p1 = self.part1.pop(stype)
            p2 = self.part2.pop(stype)

            sensor_256 = p1 + p2[:128]         # 256 sensor bytes
            imu_16 = p2[128:128+16]            # 16 imu bytes

            return sensor_256, imu_16

        return None


def main():
    port = "/dev/ttyACM0"         # COM9 left   COM10 right
    baud = 921600         # 规格书常用 921600；高频版有线可能到 3,000,000
    timeout = 0.05

    ser = serial.Serial(port, baudrate=baud, timeout=timeout)
    parser = JQFrameParser(want_sensor_type=SENSOR_RH)  # 0x01 = LH

    print("Reading... Ctrl+C to stop")
    try:
        while True:
            chunk = ser.read(256)  # read whatever available (up to 4096 bytes)
            if not chunk:
                continue

            packets = parser.feed(chunk)
            for order, stype, payload in packets:
                assembled = parser.on_packet(order, stype, payload)
                if assembled is None:
                    continue

                sensor_256, imu_16 = assembled

                # --- 你拿到的数据就是这里 ---
                # sensor_256: 256个字节（0~255）代表256个传感点的原始值/状态值
                # imu_16: 16个字节（具体怎么解码要看他们IMU打包格式；文档只说“16个数据”）
                q0, q1, q2, q3 = decode_imu_quat_f32_le(imu_16)
                # dump_imu16(imu_16)

                print(f"[{SENSOR_TYPE.get(stype, hex(stype))}] "
                      f"sensor_len={len(sensor_256)} imu_len={len(imu_16)} "
                      f"sensor_head={list(sensor_256[:60])}")
                # 左手弯曲信号
                print(f"thumb_cmc_pitch={sensor_256[210-1]}")
                print(f"index_mcp_pitch={sensor_256[213-1]}")
                print(f"middle_mcp_pitch={sensor_256[216-1]}")
                print(f"ring_mcp_pitch={sensor_256[219-1]}")
                print(f"pinky_mcp_pitch={sensor_256[222-1]}")
                print(f"index_thumb_middle_1={sensor_256[147-1]}")  # 0 15
                print(f"index_thumb_middle_2={sensor_256[179-1]}")  # 0 30
                print(f"index_thumb_middle_3={sensor_256[148-1]}")  # 0 5

                print(f"ring_dip_1={sensor_256[236-1]}")  # 2 34
                print(f"ring_dip_2={sensor_256[235-1]}")  # 30 48
                print(f"ring_dip_3={sensor_256[234-1]}")  # 21 48

                # 右手弯曲信号
                print(f"thumb_cmc_pitch={sensor_256[47-1]}")
                print(f"index_mcp_pitch={sensor_256[44-1]}")
                print(f"middle_mcp_pitch={sensor_256[41-1]}")
                print(f"ring_mcp_pitch={sensor_256[38-1]}")
                print(f"pinky_mcp_pitch={sensor_256[35-1]}")
                print(f"index_thumb_middle_1={sensor_256[110-1]}")  # 0 15
                print(f"index_thumb_middle_2={sensor_256[78-1]}")  # 0 30
                print(f"index_thumb_middle_3={sensor_256[109-1]}")  # 0 5


                print("quat:", q0, q1, q2, q3)
                

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
