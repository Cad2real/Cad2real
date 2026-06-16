import threading
from time import sleep
import sys
import os
import re

# Add project root and subdirectories to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'dobot_control', 'robot'))

from dobot_control.robot.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove,DobotApiFeedBack, MyType, alarmAlarmJsonFile

current_actual = [-1]
algorithm_queue = -1
enableStatus_robot = -1
robotErrorState = False
robotMode = 0   
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
        feedFour = DobotApiFeedBack(ip,feedPort)
        print(">.<连接成功>!<")
        return dashboard, move, feed,feedFour
    except Exception as e:
        print(":(连接失败:(")
        raise e


def RunPoint(move: DobotApiMove, joint_list: list):
    move.JointMovJ(joint_list[0], joint_list[1], joint_list[2],
              joint_list[3], joint_list[4], joint_list[5])


def GetFeed(feedFour: DobotApiFeedBack):
    global current_actual
    global algorithm_queue
    global enableStatus_robot
    global robotErrorState
    global robotMode
    # Get robot state
    while True:
        with globalLockValue:
            feedInfo = feedFour.feedBackData()
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                # Refresh properties
                robotMode=feedInfo['robot_mode'][0]
                current_actual = feedInfo["tool_vector_actual"][0]
                algorithm_queue = feedInfo['run_queued_cmd'][0]
                enableStatus_robot = feedInfo['enable_status'][0]
                robotErrorState = feedInfo['error_status'][0]
                # Custom add any additional required feedback data
            sleep(0.001)

def WaitArrive(point_list):
    while True:
        is_arrive = True
        globalLockValue.acquire()
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
            if is_arrive:
                globalLockValue.release()
                return
        globalLockValue.release()
        sleep(0.001)


def ClearRobotError(dashboard: DobotApiDashboard):
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()    # Read controller and servo alarm codes
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


if __name__ == '__main__':
    dashboard, move, feed,feedFour = ConnectRobot()
    feed_thread = threading.Thread(target=GetFeed, args=(feedFour,))
    feed_thread.daemon = True
    feed_thread.start()
    feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
    feed_thread1.daemon = True
    feed_thread1.start()
    dashboard.ClearError()
    print("开始使能...")
    dashboard.EnableRobot()
    print("完成使能:)")
    print("循环执行...")
    point_a = [59.0656, 93.6825, -90.1432, -3.5384, -59.0659, -0.0006]
    point_b = [0, 0, 0, 0, 0, 0]

        
    print("开始执行路径...")
    RunPoint(move, point_b)
    print("任务完成，程序结束。")