import bpy
from mathutils.bvhtree import BVHTree

object_filepaths = [
    'res/chrystal.obj',
    'res/001570.obj',
    'res/018772.obj',
    'res/036546.obj',
    'res/hanging_lamp.obj',
    'res/teapot.obj']
ground_plane_size = 25
collision_margin = 0.01

# We start with a default scene, which we need to clear first
bpy.ops.object.select_all(action='DESELECT')

for obj in bpy.data.objects:
    obj.select_set(True)
    bpy.ops.object.delete()

# Set up physics simulation
bpy.ops.scene.new()

# We now import each mesh into the scene


for index, path in enumerate(object_filepaths):
    bpy.ops.import_scene.obj(filepath=path)
    obj = bpy.context.selected_objects[0]
    assert(len(bpy.context.selected_objects) == 1)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.objects_add()
    obj.rigid_body.collision_shape = 'MESH'
    obj.rigid_body.collision_margin = collision_margin
    bpy.ops.rigidbody.mass_calculate()

    # Triangulate faces
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create oriented bounding box
    vertices = [v.co for v in obj.data.vertices]


    # Move object
    obj.location.z = 6.0 + 3 * index
    obj.rotation_euler[1] = 4


# Attract objects to the reference object
attractor = bpy.data.objects[0]
constraints = []
for obj in bpy.data.objects[1:]:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.rigidbody.constraint_add(type='GENERIC')
    constraint = obj.constraints["RigidBody Constraint"]
    constraint.object1 = attractor
    constraints.append(constraint)

bpy.context.view_layer.update()
num_frames_to_simulate = 250
bpy.context.scene.frame_end = num_frames_to_simulate
bpy.ops.ptcache.bake_all(bake=True)

# Remove constraints after simulation
for constraint in constraints:
    bpy.context.object.constraints.remove(constraint)

bpy.context.view_layer.update()

# We now create the ground plane
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.mesh.primitive_plane_add(size=ground_plane_size)
ground_plane = bpy.context.active_object
bpy.context.view_layer.objects.active = ground_plane
ground_plane.select_set(True)
bpy.ops.rigidbody.objects_add()
ground_plane.rigid_body.type = 'PASSIVE'
ground_plane.rigid_body.collision_shape = 'MESH'
ground_plane.rigid_body.collision_margin = collision_margin



# debug: save resulting file
bpy.ops.wm.save_as_mainfile(filepath="output.blend")

