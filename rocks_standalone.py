from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
import omni.usd
import uuid
save = False
stage = omni.usd.get_context().get_stage()
if not stage:
    stage = Usd.Stage.CreateNew("rocks_with_advanced_materials.usda")
    save = True

import numpy as np
from scipy.spatial import ConvexHull
import random
from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf, UsdPhysics, UsdLux
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.nucleus import get_assets_root_path
import omni.graph.core as og

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)



# Paths to texture files
texture_files = {
    "diffuse": ["/home/ifodor/Downloads/Textures/rock1/rock_face_diff_1k.jpg"],
    "normal": ["/home/ifodor/Downloads/Textures/rock1/rock_face_nor_gl_1k.jpg"],
    "displacement": ["/home/ifodor/Downloads/Textures/rock1/rock_face_disp_1k.jpg"],
}

rock_materials = [
    "/World/ConveyorScene/Looks/Stone_01",
    "/World/ConveyorScene/Looks/Stone_02",
    "/World/ConveyorScene/Looks/Stone_03",
    "/World/ConveyorScene/Looks/Stone_04",
    "/World/ConveyorScene/Looks/Stone_Black_01",
    "/World/ConveyorScene/Looks/Stone_Black_02",
    "/World/ConveyorScene/Looks/Stone_Black_03"
]


class RockSimulation():
    def __init__(self, size_log_mean = -2.5, sigma = 0.6, rocks_per_second = 30, fps = 60, image_width = 640, image_height = 480):
        self.image_width = image_width
        self.image_height = image_height
        self.rocks_per_second = rocks_per_second
        self.size_log_mean = size_log_mean
        self.sigma = sigma
        self.fps = fps
        self.ros_version = "ROS2"
        self.ros_bridge_version = "ros2_bridge"

        self.size_distribution = lambda size: np.random.lognormal(size_log_mean, sigma, size)

        self.render_steps_per_rock = int(fps / rocks_per_second)
        self.render_step = 0
        self.rock_id = 1


        prim = get_prim_at_path("/World/ConveyorScene")

        if not prim.IsValid():
            prim = define_prim("/World/ConveyorScene", "Xform")
            asset_path = "usd/conveyor_demo3.usdc"
            prim.GetReferences().AddReference(asset_path)


        self.dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        UsdLux.LightAPI(self.dome_light).CreateIntensityAttr(1500)
        self.parent = UsdGeom.Scope.Define(stage, "/World/Rocks")
        self.assets_root_path = get_assets_root_path()
        self.drop_zone = get_prim_at_path("/World/ConveyorScene/DropZone")

        self.world = World(set_defaults = True, rendering_dt = 1/ fps)
        self.stage = omni.usd.get_context().get_stage()
        self.world.initialize_physics()
        self.world.add_physics_callback("create_rock", self.generate_rock_task)
        self.world.add_physics_callback("delete_rock", self.delete_rock_task)
        self._init_cameras()


    def create_advanced_material(self, stage, material_path, diffuse, normal, displacement):
        """
        Creates a material with tessellation, bump mapping, and displacement mapping.
        """
        material = UsdShade.Material.Define(stage, material_path)
        
        # Create the shader
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Add a 'surface' output to the shader
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

        # Diffuse map
        diffuse_tex = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTexture")
        diffuse_tex.CreateIdAttr("UsdUVTexture")
        diffuse_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse)
        diffuse_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuse_tex.GetOutput("rgb"))

        # Normal map
        normal_tex = UsdShade.Shader.Define(stage, f"{material_path}/NormalTexture")
        normal_tex.CreateIdAttr("UsdUVTexture")
        normal_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal)
        normal_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal_tex.GetOutput("rgb"))

        # Displacement map
        displacement_tex = UsdShade.Shader.Define(stage, f"{material_path}/DisplacementTexture")
        displacement_tex.CreateIdAttr("UsdUVTexture")
        displacement_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(displacement)
        displacement_tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
        shader.CreateInput("displacement", Sdf.ValueTypeNames.Float).ConnectToSource(displacement_tex.GetOutput("r"))
        shader.CreateInput("displacementScale", Sdf.ValueTypeNames.Float).Set(1)
        # Optional: Set roughness
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)

        # Bind the shader's surface output to the material's surface output
        material.CreateSurfaceOutput().ConnectToSource(shader_output)

        return material

    def apply_physics(self, mesh):

        prim = mesh.GetPrim()

        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        
        prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.05)
        prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.05)
        
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(True)

        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.01)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(0.005)

        prim.CreateAttribute("physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(256)
        prim.CreateAttribute("physxConvexHullCollision:minThickness", Sdf.ValueTypeNames.Float).Set(0.001)

        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision_api.CreateApproximationAttr("convexHull")


    def apply_uv_mapping(self, mesh, vertices, adjusted_faces, uv_scale=1.0):
        """
        Apply planar UV mapping to the mesh, ensuring UVs align with face triplets.
        """
        # Generate planar UV coordinates (scale to control texture repetition)
        uvs = [Gf.Vec2f(v[0] * uv_scale % 1, v[1] * uv_scale % 1) for v in vertices]

        # Flatten UVs to align with the face vertex indices
        flattened_uvs = []
        for face in adjusted_faces:
            for vertex_idx in face:
                # Ensure every vertex in the face contributes a UV coordinate
                flattened_uvs.append(uvs[vertex_idx])

        # Assign UVs to the mesh using `primvars:st`
        st_attr = mesh.GetPrim().CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray)
        st_attr.Set(flattened_uvs)

        # Set interpolation to "faceVarying"
        mesh.GetPrim().CreateAttribute("primvars:st:interpolation", Sdf.ValueTypeNames.Token).Set("faceVarying")


    def generate_rock(self, stage, path, size_distribution=None, translate = None, min_points=100, max_points=200):
        """
        Generates a rock mesh with random transformations and applies an advanced material.
        """
        mean_values = np.random.normal(-2.85, 0.1,3)      # Means for X, Y, Z
        sigma_values = np.random.normal(0.3, 0.1,3)      # Sigmas for X, Y, Z

        # Generate a random number of points
        num_points = random.randint(min_points, max_points)

        # Generate points for each axis using the lognormal distribution.
        # np.random.lognormal expects scalar values for mean and sigma,
        # so we generate each coordinate separately and then stack them.
        points = np.column_stack([
            np.random.lognormal(mean, sigma, num_points)
            for mean, sigma in zip(mean_values, sigma_values)
        ])

        hull = ConvexHull(points)

        vertices = points[hull.vertices]
        faces = hull.simplices

        vertex_map = {original_idx: new_idx for new_idx, original_idx in enumerate(hull.vertices)}
        adjusted_faces = [[vertex_map[idx] for idx in face] for face in faces]

        face_vertex_indices = [idx for face in adjusted_faces for idx in face]
        face_vertex_counts = [len(face) for face in adjusted_faces]

        mesh = UsdGeom.Mesh.Define(stage, path)

        mesh_points = [Gf.Vec3f(*vertex) for vertex in vertices]
        mesh.GetPointsAttr().Set(mesh_points)
        mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)

        points_xy = np.array([[v[0], v[1]] for v in vertices])
        min_xy = points_xy.min(axis=0)
        max_xy = points_xy.max(axis=0)
        range_xy = max_xy - min_xy
        # Avoid division by zero
        range_xy[range_xy == 0] = 1.0

        # Generate UV coordinates per face vertex.
        # Note: Since USD meshes typically expect face-varying UVs, we need one UV for every face vertex (i.e. iterating through adjusted_faces).
        uv_coords = []
        for face in adjusted_faces:
            for idx in face:
                point = vertices[idx]
                u = (point[0] - min_xy[0]) / range_xy[0]
                v = (point[1] - min_xy[1]) / range_xy[1]
                uv_coords.append(Gf.Vec2f(u, v))

        # Create a primvar named "st" for the UV coordinates.
        # Using 'faceVarying' interpolation so that each face-vertex gets its own UV.
        st_primvar = mesh.GetPrim().CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray)
        # st_primvar = mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st_primvar.SetMetadata("interpolation", "faceVarying")
        st_primvar.Set(uv_coords)

        # Apply UV mapping
        self.apply_uv_mapping(mesh, vertices, adjusted_faces, uv_scale=0.1)

        # Enable subdivision for tessellation
        mesh.GetPrim().CreateAttribute("subdivisionScheme", Sdf.ValueTypeNames.Token).Set("bilinear")
        # Enable refinement/tessellation
        mesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
        mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(2)
        
        
        # Bind a random material with advanced features
        
        material_path = random.choice(rock_materials)
        material_prim = UsdShade.Material(get_prim_at_path(material_path))
        UsdShade.MaterialBindingAPI(mesh).Bind(material_prim)

        # Apply random transformations
        xformAPI = UsdGeom.XformCommonAPI(mesh)

        if translate:
            xformAPI.SetTranslate(translate)
        else:
            xformAPI.SetTranslate((random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(2, 5)))
        
        xformAPI.SetRotate((random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)))

        # if size_distribution:
        #     xformAPI.SetScale(tuple(size_distribution(3)))
        # else:
        #     mean = 0.0
        #     sigma = 0.5
        #     xformAPI.SetScale(np.random.lognormal(mean, sigma, 3).tolist())
        self.apply_physics(mesh)

    def generate_rock_task(self, event):
        self.render_step += 1
        if self.render_step % self.render_steps_per_rock == 0:
            print(f"Generating rock {self.rock_id}")
            translate = self.drop_zone.GetPrim().GetAttribute("xformOp:translate").Get()
            self.generate_rock(stage, f"{self.parent.GetPath()}/Rock_{self.rock_id}",
                                translate = translate, size_distribution = self.size_distribution)
            self.rock_id += 1

    def delete_rock_task(self, event):
        rocks = self.stage.GetPrimAtPath("/World/Rocks").GetChildren()
        rocks = [rock for rock in rocks if UsdGeom.Mesh(rock)]
        for rock in rocks:
            if rock.GetPrim().GetAttribute("xformOp:translate").Get()[2] < 0.3:
                rock.GetStage().RemovePrim(rock.GetPath())
                break

    
    def _init_cameras(self):
        keys = og.Controller.Keys
        self.camera_graphs = []
        camera_path = "/World/ConveyorScene/CameraScope/CamerXform/Camera"
        # Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers
        graph_path = "/ROS_camera"
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
                    ("createViewport.inputs:name", "CameraViewport"),
                    ("setViewportResolution.inputs:height", int(self.image_height)),
                    ("setViewportResolution.inputs:width", int(self.image_width)),
                    ("cameraHelperRgb.inputs:frameId", "camera"),
                    ("cameraHelperRgb.inputs:nodeNamespace", "/conveyor"),
                    ("cameraHelperRgb.inputs:topicName", "camera/rgb"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperInfo.inputs:frameId", "camera"),
                    ("cameraHelperInfo.inputs:nodeNamespace", "/conveyor"),
                    ("cameraHelperInfo.inputs:topicName", "/camera_info"),
                    ("cameraHelperInfo.inputs:type", "camera_info"),
                    ("setCamera.inputs:cameraPrim", [Sdf.Path(camera_path)]),
                ],
            },
        )
        self.camera_graphs.append(camera_graph)


rock_sim = RockSimulation()

while simulation_app.is_running():
    rock_sim.world.step(render=True)

simulation_app.close()
