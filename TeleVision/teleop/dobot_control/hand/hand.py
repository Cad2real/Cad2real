from hand_control import hand_control
import numpy as np
import time

hand = hand_control(password=3)
hand.service_start()

# hand.move2zero()

hand.angle_move([[500,1000,1000,1000,1000,1000]])

hand.move2zero()

time.sleep(1)
hand.service_stop()