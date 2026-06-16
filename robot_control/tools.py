from control.dobot_contorl import dobot_control
# from control.linkhand_control import linkhand_control
import cv2
import numpy as np

if __name__ == '__main__':
    # Lhand_gui = 0
    # Rhand_gui = 0
    # init_pose = [255,0,255,255,255,255,255,255,255,255]

    # Lhand_pose = [[255,255,255,255,255,255,255,255,255,255]]
    # Rhand_pose = [[255,255,255,255,255,255,255,255,255,255]]

    nova2_drag = 0
    nova5_drag = 0

    # 0:nova2  1:nova5
    nova2_nova5 =1
    pose = [[28.006, 21.685, 66.818, 126.066, 77.604, -185.062]]
    # pose = [[-66.604, -5.366, -71.714, 42.856, 116.071, -0.713]]


    # if Lhand_gui and Rhand_gui:
    #     raise ValueError("不可同时开启两个gui")
    # elif Lhand_gui:
    #     Lhand = linkhand_control(
    #             hand_type="left", hand_joint="L10", can="can1",
    #             start_gui=True,
    #             init_pose=init_pose
    #         )
    # elif Rhand_gui:
    #     Rhand = linkhand_control(
    #             hand_type="right", hand_joint="L10", can="can0",
    #             start_gui=False,
    #             init_pose=init_pose
    #         )
    # else:
    #     Lhand = linkhand_control(hand_type="left", hand_joint="L10", can="can1")
    #     Rhand = linkhand_control(hand_type="right", hand_joint="L10", can="can0")
    #     Lhand.move(pose_list=Lhand_pose)
    #     Rhand.move(pose_list=Rhand_pose)


    nova2 = dobot_control(connect_ip="192.168.5.1")
    nova5 = dobot_control(connect_ip="192.168.5.2")

    if nova2_drag and nova5_drag:
        raise ValueError("不可同时拖拽")
    elif nova2_drag:
        nova2.DargOrNot(True)
        dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        while True:
            cv2.imshow('Control Window', dummy_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                print('suibian')
                nova2.DargOrNot(False)
                break
    elif nova5_drag:
        nova5.DargOrNot(True)
        dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        while True:
            cv2.imshow('Control Window', dummy_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                nova5.DargOrNot(False)
                break
    else:
        if nova2_nova5:
            nova5.move(pose_list=pose)
        else:
            nova2.move(pose_list=pose)


