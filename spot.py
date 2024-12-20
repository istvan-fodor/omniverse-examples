# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# omniverse
from pxr import Gf, UsdGeom
import usdrt.Sdf
import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy
import random
from omni.isaac.core.objects import DynamicCuboid, DynamicCone, DynamicCapsule
import omni.graph.core as og
from omni.isaac.sensor import IMUSensor, LidarRtx
import numpy as np

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
carb.settings.get_settings().set("persistent/app/omniverse/gamepadCameraControl", False)
is_ros2 = True



class SimulationWorld():
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
        else:
            carb.log_info(f"Asset root path: {assets_root_path}")

        self.is_ros2 = is_ros2
        self.ros_version = "ROS1"
        self.ros_bridge_version = "ros_bridge"
        self.ros_vp_offset = 1
        if self.is_ros2:
            self.ros_version = "ROS2"
            self.ros_bridge_version = "ros2_bridge"

        # Spawn warehouse scene
        prim = define_prim("/World/Ground", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
        prim.GetReferences().AddReference(asset_path)
        self.init_clock()

        # Add random geometric elements
        self.spawn_random_objects(num_objects=10)


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
                    ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
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

            
class SpotSimulation():

    def __init__(self, prim_path, world):
        self.image_width = 640
        self.image_height = 480
        self.world = world
        self._prim_path=prim_path
        self._absolute_path = f"/World/{self._prim_path}"
       
        # after stage is defined
        self._stage = omni.usd.get_context().get_stage()

        self.is_ros2 = is_ros2
        self.first_step = True
        self.reset_needed = False
        
        self.ros_version = "ROS1"
        self.ros_bridge_version = "ros_bridge"
        self.ros_vp_offset = 1
        if self.is_ros2:
            self.ros_version = "ROS2"
            self.ros_bridge_version = "ros2_bridge"
        
        # spawn robot
        self.spot = SpotFlatTerrainPolicy(
            prim_path=self._absolute_path,
            name="Spot",
            position=np.array([0, 0, 0.8]),
        )
        self.world.reset()
        
        self._init_twist_subscriber()
        self._init_cameras()
        self._init_odometry()
        self._init_lidar()
        self._init_imu()
        
        self.world.add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        self.world.reset()

    def _init_imu(self):
        stage = omni.usd.get_context().get_stage()
        xform_path = f"{self._absolute_path}/body/imu_xform"
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
                    ("RosContext", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}Context"),
                    ("ReadIMU", f"omni.isaac.sensor.IsaacReadIMU"),
                    ("PublishIMU", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}PublishImu"),
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
        xform_path = f"{self._absolute_path}/body/lidar_xform"
        xform = UsdGeom.Xform.Define(stage, xform_path)
        xform.AddTranslateOp().Set(Gf.Vec3f(0.4, 0.0, .1))
        path = f"{xform_path}/lidar"
    
        self.lidar = LidarRtx(path, name="spot_lidar", orientation=np.array([1.0, 0.0, 0.0, 0.0]))

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
                    ("RosContext", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}Context"),
                    ("CreateRenderProduct", f"omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                    ("PublishPointCloud", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}RtxLidarHelper"),
                    ("PublishLaserScan", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}RtxLidarHelper"),
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
                    ("RosContext", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}Context"),
                    ("ComputeOdometry", f"omni.isaac.core_nodes.IsaacComputeOdometry"),
                    ("ReadSimTime", f"omni.isaac.core_nodes.IsaacReadSimulationTime"),
                    ("PublishOdometry", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}PublishOdometry"),
                    ("PublishRawTF1", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}PublishRawTransformTree"),
                    ("PublishRawTF2", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}PublishRawTransformTree"),
                    ("PublishTF", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}PublishTransformTree"),
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
                    ("PublishRawTF2.inputs:childFrameId", "body"),
                    ("PublishRawTF2.inputs:topicName", "tf"),
                    ("PublishTF.inputs:topicName", "tf"),
                    ("PublishTF.inputs:parentPrim", f"{self._absolute_path}/body"),
                    ("PublishTF.inputs:targetPrims", [f"{self._absolute_path}/body/lidar_xform/lidar"
                                                      , f"{self._absolute_path}/body/imu_xform/imu"
                                                      , f"{self._absolute_path}/body/camera_right"
                                                      , f"{self._absolute_path}/body/camera_left"]),

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
                    ("RosContext", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}Context"),
                    ("ROSSubscribeTwist", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}SubscribeTwist"),
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
            ("/camera_right", Gf.Vec3d(0.2693, -0.025, 0.067), (90, 0, -90), 21, 16, "perspective", 24, 400),
        ]

        # add cameras on the body link
        self.camera_graphs = []
        for i in range(len(self.cameras)):
            # add camera prim
            camera = self.cameras[i]
            camera_path = f"{self._absolute_path}/body{camera[0]}" 
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
                        ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                        ("setViewportResolution", "omni.isaac.core_nodes.IsaacSetViewportResolution"),
                        ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                        ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                        ("cameraHelperRgb", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}CameraHelper"),
                        ("cameraHelperInfo", f"omni.isaac.{self.ros_bridge_version}.{self.ros_version}CameraHelper"),
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
                        ("createViewport.inputs:name", "Viewport " + str(i + self.ros_vp_offset)),
                        ("setViewportResolution.inputs:height", int(self.image_height)),
                        ("setViewportResolution.inputs:width", int(self.image_width)),
                        ("cameraHelperRgb.inputs:frameId", camera[0]),
                        ("cameraHelperRgb.inputs:nodeNamespace", "/spot"),
                        ("cameraHelperRgb.inputs:topicName", "camera_forward" + camera[0] + "/rgb"),
                        ("cameraHelperRgb.inputs:type", "rgb"),
                        ("cameraHelperInfo.inputs:frameId", camera[0]),
                        ("cameraHelperInfo.inputs:nodeNamespace", "/spot"),
                        ("cameraHelperInfo.inputs:topicName", camera[0] + "/camera_info"),
                        ("cameraHelperInfo.inputs:type", "camera_info"),
                        ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
                    ],
                },
            )
            self.camera_graphs.append(camera_graph)

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self.spot.initialize()
            self.first_step = False
        elif self.reset_needed:
            self.world.reset(True)
            self.reset_needed = False
            self.first_step = True
        else:
            linear_velocity = og.Controller.attribute("ROSSubscribeTwist.outputs:linearVelocity", graph_id=self.twist_graph).get()
            angular_velocity = og.Controller.attribute("ROSSubscribeTwist.outputs:angularVelocity", graph_id=self.twist_graph).get()
            command = np.array(linear_velocity) + np.array(angular_velocity)
            self.spot.advance(step_size, command)

world = SimulationWorld()
sim = SpotSimulation("spot", world.world)


while simulation_app.is_running():
    world.world.step(render=True)

simulation_app.close()

