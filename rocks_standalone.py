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
# Paths to texture filesa
texture_files = {
    "diffuse": ["/home/ifodor/Downloads/Textures/rock1/rock_face_diff_1k.jpg"],
    "normal": ["/home/ifodor/Downloads/Textures/rock1/rock_face_nor_gl_1k.jpg"],
    "displacement": ["/home/ifodor/Downloads/Textures/rock1/rock_face_disp_1k.jpg"],
}

def create_advanced_material(stage, material_path, diffuse, normal, displacement):
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

def apply_physics(mesh):

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


def apply_uv_mapping(mesh, vertices, adjusted_faces, uv_scale=1.0):
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

    # Debugging
    # print("Number of faces:", len(adjusted_faces))
    # print("Expected flattened UV count (faces * 3):", len(adjusted_faces) * 3)
    # print("Actual flattened UV count:", len(flattened_uvs))
    # print("Flattened UVs (face-aligned):", flattened_uvs)

def generate_rock(stage, path, materials, size_distribution=None, translate = None, min_points=30, max_points=50):
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
    apply_uv_mapping(mesh, vertices, adjusted_faces, uv_scale=0.1)

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
    apply_physics(mesh)

# # Ensure the default prim and parent Scope exist
# default_prim = stage.GetDefaultPrim()
# parent = default_prim.GetChild("Rocks")

# # Generate multiple rocks with advanced materials
# gen_id = str(uuid.uuid4())
# for i in range(10):
#     generate_rock(stage, f"{parent.GetPath()}/Rock_{gen_id[0:8]}_{i}", materials)

dome_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
UsdLux.LightAPI(dome_light).CreateIntensityAttr(3000)
parent = UsdGeom.Scope.Define(stage, "/World/Rocks")
assets_root_path = get_assets_root_path()
source = UsdGeom.Cube.Define(stage, "/World/Cube")

ground = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground.GetReferences().AddReference(asset_path)

materials = [
    create_advanced_material(stage, parent.GetPath().AppendChild("Materials"),
                             random.choice(texture_files["diffuse"]),
                             random.choice(texture_files["normal"]),
                             random.choice(texture_files["displacement"]))
    for i in range(len(texture_files["diffuse"]))
]



fps = 60
world = World(set_defaults = True, rendering_dt = 1/ fps)
stage = omni.usd.get_context().get_stage()
world.initialize_physics()


### Simulation parameters
rocks_per_second = 3
size_log_mean = 0
sigma = 0.5
source_location = (0,0,10)
###


source.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(source_location)
render_steps_per_rock = int(fps / rocks_per_second)
render_step = 0
rock_id = 1


def size_distribution():
    return np.random.lognormal(size_log_mean, sigma)


def generate_rock_task(event):
    global render_step
    global rock_id
    render_step += 1
    if render_step % render_steps_per_rock == 0:
        print(f"Generating rock {rock_id}")
        translate = source.GetPrim().GetAttribute("xformOp:translate").Get()
        generate_rock(stage, f"{parent.GetPath()}/Rock_{rock_id}", materials, translate = translate, size_distribution = size_distribution)
        rock_id += 1
        render_step = 0

world.add_render_callback("create_rock", generate_rock_task)

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
