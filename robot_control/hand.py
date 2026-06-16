from control.hand_control import hand_control
import numpy as np
import time

hand = hand_control(password=3)
hand.service_start()

# hand.pos_move([[0,0,0,0,0,0]])
# hand.pos_move([[2000,2000,2000,900,600,1600]])
# 
hand.pos_move([[2000,2000,700,700,700,2000]])

# hand.pos_move([[2000,2000,930,900,810,2000]])

# hand.pos_move([[1000,1000,1000,900,1000,0]])



# hand.move2zero()

# hand.move2bag()

time.sleep(1)
hand.service_stop()