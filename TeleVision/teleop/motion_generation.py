import torch
import numpy as np

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class MotionGenerator:
    def __init__(self, robot_config_path="nova_evs.yml"):
        """
        初始化函数修改：
        - 【步骤2】在加载 MotionGenConfig 时，加入了全局优化参数，
          以生成更平滑、更适合遥操作的轨迹。
        """
        # 创建世界环境
        world_config = {
            "cuboid": {
                "table": {
                    "dims": [1, 1, 0.1],
                    "pose": [0, 0, -0.05, 1, 0, 0, 0],
                },
            },
        }

        # 【关键修改】在加载配置时，调整全局优化参数
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config_path,
            world_config,
            interpolation_dt=0.01,
            
            # --- 步骤2：全局优化参数 ---
            
            # 1. 关闭时间优化，使用固定的dt，使轨迹更平滑、可预测
            #    时间最优轨迹（True）可能会导致运动激进，不适合遥操作
            optimize_dt=False,
            
            # 2. 为轨迹优化设置一个固定的时间步长（秒）
            #    较大的dt通常意味着更慢、更平滑的运动
            trajopt_dt=0.1,
            
            # 3. 适当放宽末端位姿的精度要求，给优化器更多空间来平滑轨迹
            #    位置容忍度 (米)
            position_threshold=0.01, # 从默认的 0.005 (5mm) 放宽到 1cm
            
            #    旋转容忍度 (无单位，基于四元数距离)
            rotation_threshold=0.1,  # 从默认的 0.05 放宽
        )
        
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        self.current_state = JointState.from_position(
            torch.zeros(1, 6).cuda(),
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        )

    def plan_motion(self, start_joint_deg, goal_pose_list):
        """
        规划函数：
        - 【步骤1】已集成：在函数内部创建独立的 MotionGenPlanConfig，
          并设置 use_start_state_as_retract=True。
        """
        # 转换输入数据
        start_joint_rad = np.radians(start_joint_deg)
        self.current_state = JointState.from_position(
            torch.tensor(start_joint_rad, dtype=torch.float32).unsqueeze(0).cuda(),
            joint_names=self.current_state.joint_names
        )
        
        # 创建目标位姿
        goal_pose = Pose.from_list(goal_pose_list)
        
        # 【步骤1】在每次调用时创建独立的规划配置
        plan_config = MotionGenPlanConfig(
            max_attempts=3,
            enable_graph=False,
            enable_opt=True,
            use_start_state_as_retract=True,
        )
        
        # 执行规划
        result = self.motion_gen.plan_single(self.current_state, goal_pose, plan_config)
        
        if result.success:
            traj = result.get_interpolated_plan()
            last_joint_rad = traj.position[-1].cpu().numpy().flatten()
            self.current_state = JointState.from_position(
                traj.position[-1].unsqueeze(0).cuda(),
                joint_names=self.current_state.joint_names
            )
            return np.degrees(last_joint_rad).tolist()
        else:
            return None