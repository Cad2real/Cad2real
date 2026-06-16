import subprocess
import sys
import time
from typing import Union
import numpy as np
import os

class OminiPicker:
    def __init__(self, password: Union[int,str], id=1, test:bool=False):
        self.password = str(password)
        self.id = f'{id:02d}'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        canusb_path = os.path.join(base_dir, 'USB-CAN-A', 'canusb')
        self.command = lambda data: f'echo {self.password} | sudo -S {canusb_path} -d /dev/ttyUSB0 -s 1000000 -i {self.id} -j {data} -n 1'
        self.value = lambda Pos,Force,Vel,Acc,Dce: f'00{str(Pos)}{str(Force)}{str(Vel)}{str(Acc)}{str(Dce)}0000'
        self.set_mode()
        if test:            
            print("lamda command is:\n", self.command('POSE_LIST'))
            self.test()

    def __call__(self, pose:Union[int,list,np.ndarray]):
        if type(pose) == int:
            self.Pos = hex(max(0, min(255,pose)))[2:].zfill(2)
            self.move()
        else:
            pose = list(pose)
            for pos in pose:
                self.Pos = hex(max(0, min(255,pos)))[2:].zfill(2)
                self.move()
            
    def set_mode(self, Pos=0, Force=255, Vel=255, Acc=255, Dce=255):
        self.Pos = hex(max(0, min(255,Pos)))[2:].zfill(2)
        self.Force = hex(max(0, min(255,Force)))[2:].zfill(2)
        self.Vel = hex(max(0, min(255,Vel)))[2:].zfill(2)
        self.Acc = hex(max(0, min(255,Acc)))[2:].zfill(2)
        self.Dce = hex(max(0, min(255,Dce)))[2:].zfill(2)

    def test(self):
        pose_list = np.array([255,0,200,50,150,100,0])
        self.__call__(pose_list)
    
    def move(self):
        value = self.value(self.Pos, self.Force, self.Vel, self.Acc, self.Dce)
        command = self.command(value)
        try:
            subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
            time.sleep(1)
        except subprocess.CalledProcessError as e:
            print('error:\n')
            print(e)
            sys.exit(1)
    
        
