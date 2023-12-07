import sys

import bpy
from mathutils.bvhtree import BVHTree
import json
import argparse
import sys
import os.path

if __name__ == "__main__":
    print('Creating scene..')

    #parser = argparse.ArgumentParser(
    #    prog='generate_object_pile.py',
    #    description='Sets up a blender scene with a force field attracting objects towards the middle')
    #parser.add_argument('configFile', type=str)
    #arguments = parser.parse_args()
    #print("Reading JSON configuration file:", arguments.configFile)

    with open("/home/bart/git/Shapebench/src/clutter-simulator/sample.json") as inFile:
        config = json.loads(inFile.read())

    # We start with a default scene, which we need to clear first
    bpy.ops.object.select_all(action='DESELECT')

    for obj in bpy.data.objects:
        obj.select_set(True)
        bpy.ops.object.delete()

    # Set up physics simulation
    bpy.ops.scene.new()

    # We now import each mesh into the scene
    for index, inputFile in enumerate(config['inputFiles']):
        meshPath = os.path.join(config['objFileDir'], inputFile['filePath'])
        print("Loading mesh:", meshPath)
        bpy.ops.import_scene.obj(filepath=meshPath)
        obj = bpy.context.selected_objects[0]
        assert(len(bpy.context.selected_objects) == 1)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.objects_add()
        obj.rigid_body.collision_shape = 'MESH'
        obj.rigid_body.collision_margin = config['collissionMargin']
        bpy.ops.rigidbody.mass_calculate()

        # Triangulate faces
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create oriented bounding box
        vertices = [v.co for v in obj.data.vertices]

        # Scale and move object
        obj.location.x = obj.location.x - inputFile['boundingSphereCentre'][0]
        obj.location.y = obj.location.y - inputFile['boundingSphereCentre'][1]
        obj.location.z = obj.location.z + (6.0 + 3 * index) - inputFile['boundingSphereCentre'][2]
        obj.rotation_euler[1] = 4

        scaleFactor = 1.0 / inputFile['boundingSphereRadius']
        bpy.ops.transform.resize(value=(scaleFactor, scaleFactor, scaleFactor))
        bpy.ops.object.transform_apply(scale=True)
        bpy.context.view_layer.update()


    # Remove constraints after simulation
    for constraint in constraints:
        bpy.context.object.constraints.remove(constraint)

    bpy.context.view_layer.update()

    # We now create the ground plane
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_plane_add(size=config['groundPlaneSize'])
    ground_plane = bpy.context.active_object
    bpy.context.view_layer.objects.active = ground_plane
    ground_plane.select_set(True)
    bpy.ops.rigidbody.objects_add()
    ground_plane.rigid_body.type = 'PASSIVE'
    ground_plane.rigid_body.collision_shape = 'MESH'
    ground_plane.rigid_body.collision_margin = config['collissionMargin']

    # Create a physics force field (attractor) on the plane
    #bpy.ops.object.effector_add(type='FORCE', enter_editmode=False, align='WORLD', location=(0, 0, 0))
    #bpy.context.object.field.strength = 50
    #bpy.context.object.field.flow = 'ATTRACT'
    #bpy.context.object.field.falloff_power = 2

    bpy.context.view_layer.update()
    num_frames_to_simulate = 250
    bpy.context.scene.frame_end = num_frames_to_simulate
    bpy.ops.ptcache.bake_all(bake=True)



    # debug: save resulting file
    bpy.ops.wm.save_as_mainfile(filepath=config['outputFile'])

