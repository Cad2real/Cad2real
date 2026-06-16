import subprocess
import sys
import time
from typing import Union
import numpy as np

class hand_control:
    def __init__(self, password: Union[int,str]):
        self.password = str(password)
        self.service_is_start = False

    def service_start(self):
        print("Try to start the hand_service")
        start_command = f'echo {self.password} | sudo -S systemctl start inspire_hand_start.service'
        check_command = f'echo {self.password} | sudo -S systemctl is-active inspire_hand_start.service'
        try:
            subprocess.run(start_command, shell=True, stdout=subprocess.PIPE, text=True)
            time.sleep(1)
            if subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip() == 'active':
                self.service_is_start = True
                print('Start service of hand')
        except subprocess.CalledProcessError as e:
            print('error:\n')
            print(e)
            sys.exit(1)
    
    def service_stop(self):
        print("Stop the hand_service")
        stop_command = f'echo {self.password} | sudo -S systemctl stop inspire_hand_start.service'
        check_command = f'echo {self.password} | sudo -S systemctl is-active inspire_hand_start.service'
        try:
            subprocess.run(stop_command, shell=True, stdout=subprocess.PIPE, text=True)
            time.sleep(1)
            if subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip() == 'inactive':
                self.service_is_start = False
                print('Close hand service')
        except subprocess.CalledProcessError as e:
            print('error:\n')
            print(e)
            sys.exit(1)
    
    def move2bag(self):
        if self.service_is_start:
            cmd = np.array([
                [0,0,0,0,0,0],
                [2000,2000,2000,2000,2000,0]
            ])
            self.pos_move(cmd)
        else:
            print("Service don't start")
            sys.exit(0)

    def move2zero(self):
        if self.service_is_start:
            cmd = np.array([
                [0,0,0,0,0,0]
            ])
            self.pos_move(cmd)
        else:
            print("Service don't start")
            sys.exit(0)

    def pos_move(self, position:np.ndarray):
        if self.service_is_start:
            for pos in position:
                pos = ' '.join([str(item) for item in pos])
                cmd = f'rosservice call /inspire_hand/set_pos {pos}'
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    print(f'hand_pos: {pos}')
        else:
            print("Service don't start")
            sys.exit(0)
        
