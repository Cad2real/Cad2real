import threading
from control.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile
from time import sleep
import numpy as np
import re

from control.hand_control import hand_control

# 全局变量(当前坐标)
current_joint = [-1]
current_pose = [-1]
algorithm_queue = -1
enableStatus_robot = -1
robotErrorState = False
globalLockValue = threading.Lock()


def ConnectRobot():
    try:
        ip = "192.168.5.1"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        print("正在建立连接...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        print(">.<连接成功>!<")
        return dashboard, move, feed
    except Exception as e:
        print(":(连接失败:(")
        raise e

def GetFeed(feed: DobotApi):
    global current_joint
    global current_pose
    global algorithm_queue
    global enableStatus_robot
    global robotErrorState
    hasRead = 0
    t = 0
    while True:
        data = bytes()
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp
        hasRead = 0
        feedInfo = np.frombuffer(data, dtype=MyType)
        if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
            globalLockValue.acquire()
            # Refresh Properties
            current_joint = feedInfo["q_actual"][0]
            current_pose = feedInfo['tool_vector_actual'][0]
            algorithm_queue = feedInfo['run_queued_cmd'][0]
            enableStatus_robot = feedInfo['enable_status'][0]
            robotErrorState = feedInfo['error_status'][0]
            globalLockValue.release()
                    # print('pose:', current_pose)
        sleep(0.001)



def ClearRobotError(dashboard: DobotApiDashboard):
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()    # 读取控制器和伺服告警码
    while True:
        globalLockValue.acquire()
        if robotErrorState:
            numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
            numbers = [int(num) for num in numbers]
            if (numbers[0] == 0):
                if (len(numbers) > 1):
                    for i in numbers[1:]:
                        alarmState = False
                        if i == -2:
                            print("机器告警 机器碰撞 ", i)
                            alarmState = True
                        if alarmState:
                            continue
                        for item in dataController:
                            if i == item["id"]:
                                print("机器告警 Controller errorid", i,
                                      item["zh_CN"]["description"])
                                alarmState = True
                                break
                        if alarmState:
                            continue
                        for item in dataServo:
                            if i == item["id"]:
                                print("机器告警 Servo errorid", i,
                                      item["zh_CN"]["description"])
                                break

                    choose = input("输入1, 将清除错误, 机器继续运行: ")
                    if int(choose) == 1:
                        dashboard.ClearError()
                        sleep(0.01)
                        dashboard.Continue()

        else:
            if int(enableStatus_robot) == 1 and int(algorithm_queue) == 0:
                dashboard.Continue()
        globalLockValue.release()
        sleep(5)

def waitArrive(key: int, value):
    while True:
        if key == 1:
            res = dashboard.GetAngle() 
        if key == 2:
            res = dashboard.GetPose()
        if (res):
            res = re.search(r'\{([^}]*)\}', res)
            numbers = [float(num.strip()) for num in res.group(1).split(',')]
            array = np.array(numbers).reshape(1,-1)[0]
            Arrive = True
            for i in range(6):
                if Arrive:
                    val1 = abs(array[i]-value[i])
                    val2 = abs(array[i]+180-value[i])
                    val3 = abs(array[i]-180-value[i])
                    val4 = abs(array[i]+360-value[i])
                    val5 = abs(array[i]-360-value[i])
                    if val1<0.01 or val2<0.01 or val3<0.01 or val4<0.01 or val5<0.01:
                        Arrive = True
                    else:
                        Arrive = False
        if Arrive:
            sleep(1)
            break
        sleep(1)


if __name__ == '__main__':
    dashboard, move, feed = ConnectRobot()
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.daemon = True
    feed_thread.start()
    feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
    feed_thread1.daemon = True
    feed_thread1.start()
    print("开始使能...")
    dashboard.EnableRobot()
    print("完成使能:)")

    hand = hand_control(password=3)
    hand.service_start()
    
    
    
    dashboard.CP(100)
    dashboard.SpeedL(25)

    step_list = [1,3,1,3,1,0]
    # step_list = [0]
    step_key = 0

    angle_key = 0
    angle = [
        [
        [0,0,0,0,0,0],
        [325.11081783959963-360, -64.92747813921874, -38.648000270684626, 301.78940226945224, 0.1774670173388721, 71.75809830517609]],

        [[325.1122017251183-360, -73.61059972706292, -51.12660949869831, 332.57976311401785, 0.06962060695746736, 62.19659049958359]],

        [[325.11081783959963-360, -64.92747813921874, -38.648000270684626, 301.78940226945224, 0.1774670173388721, 71.75809830517609]],
    ]

    hand_key = 0
    hand_pose = [
        [[0,0,0,0,0,0],
         [0,0,0,0,0,2000]],

        [[0,0,730,660,410,2000]],

        # [[2000,2000,950,900,600,2000]]

    ]
        

    while True:
        if step_list[step_key] == 1:
            points = angle[angle_key]
            angle_key = angle_key + 1
            for point in points:
                move.JointMovJ(point[0], point[1], point[2],point[3], point[4], point[5])
                waitArrive(1, point)
            step_key = step_key + 1

        elif step_list[step_key] == 3:
            hand.pos_move(hand_pose[hand_key])
            hand_key = hand_key + 1
            step_key = step_key + 1
            # sleep(3)

        elif step_list[step_key] == 0:
            print('\nover!\n')
            break
    # WaitArrive(feed, point)
    
    # sleep(3)
    dashboard.DisableRobot()
    hand.service_stop()

