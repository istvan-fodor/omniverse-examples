# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Standard library imports
import io
import random
from typing import List, Optional

# Third-party imports
import numpy as np
import torch

# Omni core and utilities
import carb
import omni
import omni.graph.core as og
import omni.isaac.core.utils.stage as stage_utils

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid, DynamicCone, DynamicCapsule
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.rotations import euler_to_rot_matrix, quat_to_euler_angles, quat_to_rot_matrix
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.sensor import IMUSensor, LidarRtx

# USD and related
from pxr import Gf, UsdGeom
import usdrt.Sdf


ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
ext_manager.set_extension_enabled_immediate("isaacsim.core.nodes", True)
ext_manager.set_extension_enabled_immediate("omni.anim.curve.core", True)
ext_manager.set_extension_enabled_immediate("omni.anim.curve.bundle", True)
ext_manager.set_extension_enabled_immediate("omni.anim.window.timeline", True)
ext_manager.set_extension_enabled_immediate("omni.kit.window.movie_capture", True)
ext_manager.set_extension_enabled_immediate("isaacsim.sensors.physics", True)

carb.settings.get_settings().set("persistent/app/omniverse/gamepadCameraControl", False)

class Go2FlatTerrainPolicy:

    def __init__(
        self,
        prim_path: str,
        name: str = "go2",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            usd_path {str} -- robot usd filepath in the directory
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot

        """
        self._stage = get_current_stage()
        self._prim_path = prim_path
        prim = get_prim_at_path(self._prim_path)

        assets_root_path = get_assets_root_path()
        if not prim.IsValid():
            prim = define_prim(self._prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")

                asset_path = assets_root_path + "/Isaac/Robots/Unitree/Go2/go2.usd"

                prim.GetReferences().AddReference(asset_path)

        self.robot = Articulation(prim_path=self._prim_path, name=name, position=position, orientation=orientation)

        self._dof_control_modes: List[int] = list()

        # Policy
        file_content = omni.client.read_file(
            "omniverse://localhost/Projects/Go2Walk/go2_flat_terrain_policy.pt"
        )[2]
        file = io.BytesIO(memoryview(file_content).tobytes())

        self._policy = torch.jit.load(file)
        self._base_vel_lin_scale = 1
        self._base_vel_ang_scale = 1
        self._action_scale = 0.2
        self._default_joint_pos = np.array([0.1, -0.1, 0.1, -0.1, 0.9, 0.9, 1.1, 1.1, -1.5, -1.5, -1.5, -1.5])
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        self._decimation = 10

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = self._base_vel_lin_scale * lin_vel_b
        # Base ang vel
        obs[3:6] = self._base_vel_ang_scale * ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9] = self._base_vel_lin_scale * command[0]
        obs[10] = self._base_vel_lin_scale * command[1]
        obs[11] = self._base_vel_ang_scale * command[2]
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:24] = current_joint_pos - self._default_joint_pos
        obs[24:36] = current_joint_vel
        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    def advance(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt {float} -- Timestep update in the world.
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            with torch.no_grad():
                obs = torch.from_numpy(obs).view(1, -1).float()
                self.action = self._policy(obs).detach().view(-1).numpy()
            self._previous_action = self.action.copy()

        action = ArticulationAction(joint_positions=self._default_joint_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize robot the articulation interface, set up drive mode
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes("force")
        self.robot.get_articulation_controller().switch_control_mode("position")
        self.robot._articulation_view.set_gains(np.zeros(12) + 60, np.zeros(12) + 1.5)

    def post_reset(self) -> None:
        """
        Post reset articulation
        """
        self.robot.post_reset()





class SimulationWorld():
    def __init__(self, environment="default"):
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 60)
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
        else:
            carb.log_info(f"Asset root path: {assets_root_path}")

        assets_root_path = get_assets_root_path()
        # if assets_root_path is None:
        #     carb.log_error("Could not find Isaac Sim assets folder")

        self.environment = environment
        if environment == "warehouse":
            prim = get_prim_at_path("/World/Warehouse")

            if not prim.IsValid():
                prim = define_prim("/World/Warehouse", "Xform")
                asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
                prim.GetReferences().AddReference(asset_path)
        elif environment == "jetty":
            prim = get_prim_at_path("/World/Jetty")

            if not prim.IsValid():
                prim = define_prim("/World/Jetty", "Xform")
                asset_path = "omniverse://localhost/Library/jetty_and_gauge2.usd"
                prim.GetReferences().AddReference(asset_path)
        else: # default
            prim = define_prim("/World/Ground", "Xform")
            asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
            prim.GetReferences().AddReference(asset_path)

            self.spawn_random_objects(num_objects=10)

    def spawn_location(self):
        if self.environment == "jetty":
            return np.array([98, -21.5, 12])
        else:
            return np.array([0, 0, 0.7])


    def init_clock(self):
        keys = og.Controller.Keys
        og.Controller.edit(
            {
                "graph_path": "/World/Clock", 
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION
            },
            {
                keys.CREATE_NODES: [
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ]
            },
        )


    def spawn_random_objects(self, num_objects=10):
        """
        Spawns a given number of random geometric objects with rigid body physics in the world.
        """
        for i in range(num_objects):
            # Generate random positions

            x = random.uniform(2, 5) if random.uniform(0,1) > 0.5 else random.uniform(-5,-2) 
            y = random.uniform(-5, 5)
            z = random.uniform(1, 3)

            # Choose a random geometry type
            shape_type = random.choice([DynamicCuboid, DynamicCapsule, DynamicCone])
            prim_path = f"/World/RandomObject_{i}"
            scale = random.uniform(0.5, 1.5)
            self.world.scene.add(
                shape_type(
                    prim_path=prim_path,
                    name=f"RandomObject_{i}",
                    position=np.array([x, y, z]),
                    scale=np.full(3, scale),
                    color=np.random.uniform(0.5, 1.0, 3),
                    mass=scale
                )
            )

            
class Go2Simulation():

    def __init__(self, prim_path, simulation):
        self.image_width = 1920 
        self.image_height = 1080
        self.world = world.world
        self._prim_path=prim_path
        self._absolute_path = f"/World/{self._prim_path}"
       
        # after stage is defined
        self._stage = omni.usd.get_context().get_stage()

        self.first_step = True
        self.reset_needed = False
        
        # spawn robot
        self.go2 = Go2FlatTerrainPolicy(
            prim_path=self._absolute_path,
            name="Go2",
            position=np.array(simulation.spawn_location()),
        )

        self.world.reset()
        
        self._init_twist_subscriber()
        self._init_cameras()
        self._init_odometry()
        #self._init_lidar()
        self._init_imu()
        
        self.world.add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        self.world.reset()

    def _init_imu(self):
        stage = omni.usd.get_context().get_stage()
        xform_path = f"{self._absolute_path}/base/imu_xform"
        xform = UsdGeom.Xform.Define(stage, xform_path)
        xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        
        path = f"{xform_path}/imu"
        
        self.imu = IMUSensor(
            prim_path=path,
            name="imu",
            frequency=60,
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
            linear_acceleration_filter_size = 10,
            angular_velocity_filter_size = 10,
            orientation_filter_size = 10,
        )

        keys = og.Controller.Keys
        graph_path = "/ROS_Imu"
        (imu_path, _, _, _) = og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RosContext", f"isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadIMU", f"isaacsim.sensors.physics.IsaacReadIMU"),
                    ("PublishIMU", f"isaacsim.ros2.bridge.ROS2PublishImu"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ReadIMU.inputs:execIn"),
                    ("RosContext.outputs:context", "PublishIMU.inputs:context"),
                    ("ReadIMU.outputs:execOut", "PublishIMU.inputs:execIn"),
                    ("ReadIMU.outputs:angVel", "PublishIMU.inputs:angularVelocity"),
                    ("ReadIMU.outputs:linAcc","PublishIMU.inputs:linearAcceleration"),
                    ("ReadIMU.outputs:orientation","PublishIMU.inputs:orientation"),
                    ("ReadIMU.outputs:sensorTime","PublishIMU.inputs:timeStamp"),
                ],
                keys.SET_VALUES: [
                    ("ReadIMU.inputs:imuPrim", path),
                    ("ReadIMU.inputs:readGravity" , True),
                    ("PublishIMU.inputs:topicName" , "imu"),
                    ("PublishIMU.inputs:frameId" , "imu"),
                    ("PublishIMU.inputs:publishAngularVelocity" , True),
                    ("PublishIMU.inputs:publishLinearAcceleration" , True),
                    ("PublishIMU.inputs:publishOrientation" , True),
                ]
            },
        )
        self.imu_path = imu_path




    def _init_lidar(self):
        stage = omni.usd.get_context().get_stage()
        xform_path = f"{self._absolute_path}/base/lidar_xform"
        xform = UsdGeom.Xform.Define(stage, xform_path)
        xform.AddTranslateOp().Set(Gf.Vec3f(0.4, 0.0, .1))
        path = f"{xform_path}/lidar"
    
        self.lidar = LidarRtx(path, name="go2_lidar", orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        keys = og.Controller.Keys
        graph_path = "/ROS_Lidar"
        (lidar_path, _, _, _) = og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RosContext", f"isaacsim.ros2.bridge.ROS2Context"),
                    ("CreateRenderProduct", f"isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("PublishPointCloud", f"isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                    ("PublishLaserScan", f"isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                    ("RosContext.outputs:context", "PublishPointCloud.inputs:context"),
                    ("RosContext.outputs:context", "PublishLaserScan.inputs:context"),
                    ("CreateRenderProduct.outputs:execOut", "PublishPointCloud.inputs:execIn"),
                    ("CreateRenderProduct.outputs:execOut", "PublishLaserScan.inputs:execIn"),
                    ("CreateRenderProduct.outputs:renderProductPath","PublishPointCloud.inputs:renderProductPath"),
                    ("CreateRenderProduct.outputs:renderProductPath","PublishLaserScan.inputs:renderProductPath"),
                ],
                keys.SET_VALUES: [
                    ("CreateRenderProduct.inputs:cameraPrim", path),
                    ("PublishPointCloud.inputs:topicName" , "point_cloud"),
                    ("PublishPointCloud.inputs:type" , "point_cloud"),
                    ("PublishPointCloud.inputs:frameId" , "lidar"),
                    ("PublishLaserScan.inputs:topicName" , "scan"),
                    ("PublishLaserScan.inputs:type" , "laser_scan"),    
                    ("PublishLaserScan.inputs:frameId" , "lidar")

                ]
            },
        )
        self.lidar_path = lidar_path


        # RTX sensors are cameras and must be assigned to their own render product
    def _init_odometry(self):
        keys = og.Controller.Keys
        graph_path = "/ROS_Odom"
        (odom_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RosContext", f"isaacsim.ros2.bridge.ROS2Context"),
                    ("ComputeOdometry", f"isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("ReadSimTime", f"isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishOdometry", f"isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    ("PublishRawTF1", f"isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    ("PublishRawTF2", f"isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    ("PublishTF", f"isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishRawTF1.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishRawTF2.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishTF.inputs:execIn"),
                    ("RosContext.outputs:context", "PublishOdometry.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishRawTF1.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishRawTF2.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                    ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                    ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                    ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                    ("ComputeOdometry.outputs:position", "PublishRawTF1.inputs:translation"),
                    ("ComputeOdometry.outputs:orientation", "PublishRawTF1.inputs:rotation"),
                ],
                keys.SET_VALUES: [
                    ("PublishOdometry.inputs:topicName", "/odom"),
                    ("ComputeOdometry.inputs:chassisPrim", self._absolute_path),
                    ("PublishRawTF1.inputs:parentFrameId", "odom"),
                    ("PublishRawTF1.inputs:childFrameId", "base_link"),
                    ("PublishRawTF1.inputs:topicName", "tf"),
                    ("PublishRawTF2.inputs:parentFrameId", "base_link"),
                    ("PublishRawTF2.inputs:childFrameId", "base"),
                    ("PublishRawTF2.inputs:topicName", "tf"),
                    ("PublishTF.inputs:topicName", "tf"),
                    ("PublishTF.inputs:parentPrim", f"{self._absolute_path}/base"),
                    ("PublishTF.inputs:targetPrims", [f"{self._absolute_path}/base/lidar_xform/lidar"
                                                      , f"{self._absolute_path}/base/imu_xform/imu"
                                                      #, f"{self._absolute_path}/base/camera_right"
                                                      , f"{self._absolute_path}/base/camera_left"]),

                ]
            },
        )
        self.odom_graph = odom_graph



    def _init_twist_subscriber(self):
        keys = og.Controller.Keys
        graph_path = "/ROS_Twist"
        (twist_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RosContext", f"isaacsim.ros2.bridge.ROS2Context"),
                    ("ROSSubscribeTwist", f"isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ROSSubscribeTwist.inputs:execIn"),
                    ("RosContext.outputs:context", "ROSSubscribeTwist.inputs:context"),
                ],
                keys.SET_VALUES: [
                    ("ROSSubscribeTwist.inputs:topicName", "/cmd_vel"),
                ]
            },
        )
        self.twist_graph = twist_graph

    def _init_cameras(self):
        keys = og.Controller.Keys

        self.cameras = [
            # 0name, 1offset, 2orientation, 3hori aperture, 4vert aperture, 5projection, 6focal length, 7focus distance
            ("/camera_left", Gf.Vec3d(0.2693, 0.025, 0.067), (90, 0, -90), 21, 16, "perspective", 24, 400),
            #("/camera_right", Gf.Vec3d(0.2693, -0.025, 0.067), (90, 0, -90), 21, 16, "perspective", 24, 400),
        ]

        # add cameras on the base link
        self.camera_graphs = []
        for i in range(len(self.cameras)):
            # add camera prim
            camera = self.cameras[i]
            camera_path = f"{self._absolute_path}/base{camera[0]}" 
            camera_prim = UsdGeom.Camera(self._stage.DefinePrim(camera_path, "Camera"))
            xform_api = UsdGeom.XformCommonAPI(camera_prim)
            xform_api.SetRotate(camera[2], UsdGeom.XformCommonAPI.RotationOrderXYZ)
            xform_api.SetTranslate(camera[1])
            camera_prim.GetHorizontalApertureAttr().Set(camera[3])
            camera_prim.GetVerticalApertureAttr().Set(camera[4])
            camera_prim.GetProjectionAttr().Set(camera[5])
            camera_prim.GetFocalLengthAttr().Set(camera[6])
            camera_prim.GetFocusDistanceAttr().Set(camera[7])


            # Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers
            graph_path = "/ROS_" + camera[0].split("/")[-1]
            (camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                        ("setViewportResolution", "isaacsim.core.nodes.IsaacSetViewportResolution"),
                        ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
                        ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                        ("cameraHelperRgb", f"isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("cameraHelperInfo", f"isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "createViewport.inputs:execIn"),
                        ("createViewport.outputs:execOut", "setViewportResolution.inputs:execIn"),
                        ("createViewport.outputs:viewport", "setViewportResolution.inputs:viewport"),
                        ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                        ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                        ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                        ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                        ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                        ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                        ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                        ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                    ],
                    keys.SET_VALUES: [
                        ("createViewport.inputs:name", "Viewport " + str(i + 1)),
                        ("setViewportResolution.inputs:height", int(self.image_height)),
                        ("setViewportResolution.inputs:width", int(self.image_width)),
                        ("cameraHelperRgb.inputs:frameId", camera[0]),
                        ("cameraHelperRgb.inputs:nodeNamespace", "/go2"),
                        ("cameraHelperRgb.inputs:topicName", "camera_forward" + camera[0] + "/rgb"),
                        ("cameraHelperRgb.inputs:type", "rgb"),
                        ("cameraHelperInfo.inputs:frameId", camera[0]),
                        ("cameraHelperInfo.inputs:nodeNamespace", "/go2"),
                        ("cameraHelperInfo.inputs:topicName", camera[0] + "/camera_info"),
                        ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
                    ],
                },
            )
            self.camera_graphs.append(camera_graph)

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self.go2.initialize()
            self.first_step = False
        elif self.reset_needed:
            self.world.reset(True)
            self.reset_needed = False
            self.first_step = True
        else:
            linear_velocity = og.Controller.attribute("ROSSubscribeTwist.outputs:linearVelocity", graph_id=self.twist_graph).get()
            angular_velocity = og.Controller.attribute("ROSSubscribeTwist.outputs:angularVelocity", graph_id=self.twist_graph).get()
            command = np.array(linear_velocity) + np.array(angular_velocity)
            self.go2.advance(step_size, command)

world = SimulationWorld("warehouse")
sim = Go2Simulation("go2", world)


while simulation_app.is_running():
    world.world.step(render=True)

simulation_app.close()

