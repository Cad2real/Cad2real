import numpy as np
import time
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'dobot_control', 'hand'))

from dobot_control.hand.hand_control import hand_control

hand = hand_control(password=3)
hand.service_start()

hand.angle_move([[1000,1000,1000,1000,1000,1000]])

time.sleep(1)
hand.service_stop()