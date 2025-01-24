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



# Paths to texture files
texture_files = {
    "diffuse": ["/home/ifodor/Downloads/Textures/rock1/rock_face_diff_1k.jpg"],
    "normal": ["/home/ifodor/Downloads/Textures/rock1/rock_face_nor_gl_1k.jpg"],
    "displacement": ["/home/ifodor/Downloads/Textures/rock1/rock_face_disp_1k.jpg"],
}


class RockSimulation():
    def __init__(self, size_log_mean = -2.5, sigma = 0.4, rocks_per_second = 30, fps = 60):
        self.rocks_per_second = rocks_per_second
        self.size_log_mean = size_log_mean
        self.sigma = sigma
        self.fps = fps

        self.size_distribution = lambda: np.random.lognormal(size_log_mean, sigma)

        self.render_steps_per_rock = int(fps / rocks_per_second)
        self.render_step = 0
        self.rock_id = 1


        prim = get_prim_at_path("/World/ConveyorScene")

        if not prim.IsValid():
            prim = define_prim("/World/ConveyorScene", "Xform")
            asset_path = "usd/conveyor_demo3.usdc"
            prim.GetReferences().AddReference(asset_path)


        self.dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        UsdLux.LightAPI(self.dome_light).CreateIntensityAttr(3000)
        self.parent = UsdGeom.Scope.Define(stage, "/World/Rocks")
        self.assets_root_path = get_assets_root_path()
        self.drop_zone = get_prim_at_path("/World/ConveyorScene/DropZone")

        self.materials = [
            self.create_advanced_material(stage, self.parent.GetPath().AppendChild("Materials"),
                                    random.choice(texture_files["diffuse"]),
                                    random.choice(texture_files["normal"]),
                                    random.choice(texture_files["displacement"]))
            for i in range(len(texture_files["diffuse"]))
        ]

        self.world = World(set_defaults = True, rendering_dt = 1/ fps)
        self.stage = omni.usd.get_context().get_stage()
        self.world.initialize_physics()
        self.world.add_physics_callback("create_rock", self.generate_rock_task)
        self.world.add_physics_callback("delete_rock", self.delete_rock_task)


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


    def generate_rock(self, stage, path, materials, size_distribution=None, translate = None, min_points=30, max_points=50):
        """
        Generates a rock mesh with random transformations and applies an advanced material.
        """
        num_points = random.randint(min_points, max_points)
        points = np.random.rand(num_points, 3)  # Generate random points in 3D space

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

        # Apply UV mapping
        self.apply_uv_mapping(mesh, vertices, adjusted_faces, uv_scale=0.1)

        # Enable subdivision for tessellation
        mesh.GetPrim().CreateAttribute("subdivisionScheme", Sdf.ValueTypeNames.Token).Set("bilinear")
        # Enable refinement/tessellation
        mesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
        mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(2)
        
        
        # Bind a random material with advanced features
        material = random.choice(materials)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

        # Apply random transformations
        xformAPI = UsdGeom.XformCommonAPI(mesh)

        if translate:
            xformAPI.SetTranslate(translate)
        else:
            xformAPI.SetTranslate((random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(2, 5)))
        
        xformAPI.SetRotate((random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)))

        if size_distribution:
            xformAPI.SetScale((size_distribution(), size_distribution(), size_distribution()))
        else:
            mean = 0.0
            sigma = 0.5
            xformAPI.SetScale(np.random.lognormal(mean, sigma, 3).tolist())
        self.apply_physics(mesh)

    def generate_rock_task(self, event):
        self.render_step += 1
        if self.render_step % self.render_steps_per_rock == 0:
            print(f"Generating rock {self.rock_id}")
            translate = self.drop_zone.GetPrim().GetAttribute("xformOp:translate").Get()
            self.generate_rock(stage, f"{self.parent.GetPath()}/Rock_{self.rock_id}", self.materials
                               , translate = translate, size_distribution = self.size_distribution)
            self.rock_id += 1

    def delete_rock_task(self, event):
        rocks = self.stage.GetPrimAtPath("/World/Rocks").GetChildren()
        rocks = [rock for rock in rocks if UsdGeom.Mesh(rock)]
        for rock in rocks:
            if rock.GetPrim().GetAttribute("xformOp:translate").Get()[2] < 0.3:
                rock.GetStage().RemovePrim(rock.GetPath())
                break




rock_sim = RockSimulation()

while simulation_app.is_running():
    rock_sim.world.step(render=True)

simulation_app.close()
