import torch
import numpy as np

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class MotionGenerator:
    def __init__(self, robot_config_path="template.yml"):
        # Create world environment
        world_config = {
            "cuboid": {
                "table": {
                    "dims": [1, 1, 0.1],
                    "pose": [0, 0, -0.05, 1, 0, 0, 0],
                },
            },
        }

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config_path,
            world_config,
            interpolation_dt=0.01,
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        self.current_state = JointState.from_position(
            torch.zeros(1, 6).cuda(),
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        )
        
        # Common planning configuration
        self.plan_config = MotionGenPlanConfig(
            max_attempts=3,
            enable_graph=False,
            enable_opt=True,
        )

    def plan_motion(self, start_joint_deg, goal_pose_list):
        """Given start joint angles (degrees) and goal pose (7D list), return planned joint angles (degrees)."""
        # Convert input data
        start_joint_rad = np.radians(start_joint_deg)
        self.current_state = JointState.from_position(
            torch.tensor(start_joint_rad, dtype=torch.float32).unsqueeze(0).cuda(),
            joint_names=self.current_state.joint_names
        )
        
        # Create goal pose
        goal_pose = Pose.from_list(goal_pose_list)
        
        # Execute planning
        result = self.motion_gen.plan_single(self.current_state, goal_pose, self.plan_config)
        
        if result.success:
            traj = result.get_interpolated_plan()
            last_joint_rad = traj.position[-1].cpu().numpy().flatten()
            self.current_state = JointState.from_position(
                traj.position[-1].unsqueeze(0).cuda(),
                joint_names=self.current_state.joint_names
            )
            return [round(x, 4) for x in np.degrees(last_joint_rad)]
        else:
            return None