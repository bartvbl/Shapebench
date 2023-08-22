import bpy

object_filepaths = ['res/chrystal.obj']
ground_plane_size = 250

# We start with a default scene, which we need to clear first
bpy.ops.object.select_all(action='DESELECT')

for obj in bpy.data.objects:
    obj.select_set(True)
    bpy.ops.object.delete()

# We now import each mesh into the scene
for path in object_filepaths:
    bpy.ops.import_scene.obj(filepath=path)

    # TODO: scale mesh

    # TODO: move mesh
    #object.location.x = ..

# We now create the ground plane
plane_vertices = [(-ground_plane_size, -ground_plane_size, 0),
                  (ground_plane_size, -ground_plane_size, 0),
                  (ground_plane_size, ground_plane_size, 0),
                  (-ground_plane_size, ground_plane_size, 0)]
plane_indices = [(0, 1, 2), (0, 2, 3)]

plane_name = "ground plane"
plane_mesh = bpy.data.meshes.new(plane_name)
plane_mesh.from_pydata(plane_vertices, [], plane_indices)
plane_mesh.update()

plane_object = bpy.data.objects.new(plane_name, plane_mesh)
bpy.context.scene.collection.objects.link(plane_object)



# debug: save resulting file
bpy.ops.wm.save_as_mainfile(filepath="output.blend")

