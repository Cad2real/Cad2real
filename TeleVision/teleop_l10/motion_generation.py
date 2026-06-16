import torch
import numpy as np

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class MotionGenerator:
    def __init__(self, robot_config_path="nova_evs.yml"):
        """
        Constructor changes:
        - Step 2: add global optimization parameters when loading MotionGenConfig
          to generate smoother, teleoperation-friendly trajectories.
        """
        # Create world environment
        world_config = {
            "cuboid": {
                "table": {
                    "dims": [1, 1, 0.1],
                    "pose": [0, 0, -0.05, 1, 0, 0, 0],
                },
            },
        }

        # Key change: adjust global optimization parameters when loading config
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config_path,
            world_config,
            interpolation_dt=0.01,
            
            # --- Step 2: global optimization parameters ---
            
            # 1. Disable time optimization and use fixed dt for a smoother,
            #    more predictable trajectory.
            #    Time-optimal trajectories (True) can be aggressive and
            #    are not ideal for teleoperation.
            optimize_dt=False,
            
            # 2. Use a fixed trajectory optimization timestep (seconds).
            #    A larger dt usually yields slower, smoother motion.
            trajopt_dt=0.1,
            
            # 3. Relax end-effector pose accuracy requirements to give the
            #    optimizer more room to smooth the trajectory.
            #    Position tolerance (meters)
            position_threshold=0.01, # Relaxed from default 0.005 (5mm) to 1cm
            
            #    Rotation tolerance (unitless, based on quaternion distance)
            rotation_threshold=0.1,  # Relaxed from default 0.05
        )
        
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        self.current_state = JointState.from_position(
            torch.zeros(1, 6).cuda(),
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        )

    def plan_motion(self, start_joint_deg, goal_pose_list):
        """
        Motion planning function:
        - Step 1: create an independent MotionGenPlanConfig inside the function,
          and set use_start_state_as_retract=True.
        """
        # Convert input data
        start_joint_rad = np.radians(start_joint_deg)
        self.current_state = JointState.from_position(
            torch.tensor(start_joint_rad, dtype=torch.float32).unsqueeze(0).cuda(),
            joint_names=self.current_state.joint_names
        )
        
        # Create goal pose
        goal_pose = Pose.from_list(goal_pose_list)
        
        # Step 1: create a separate plan config for each call
        plan_config = MotionGenPlanConfig(
            max_attempts=3,
            enable_graph=False,
            enable_opt=True,
            use_start_state_as_retract=True,
        )
        
        # Execute planning
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