import bpy
import os
from math import pi
import random
import copy

# scrip assumes an empty blender scene
bpy.context.scene.render.engine = 'CYCLES'


def degree_to_rad(d):
    return pi * d / 180


# if ran on blender path has to be complete, e.g.:
# Users/joaobelo/Desktop/ml_project
dirpath = ""
filename = "lack_updated.obj"
filepath = os.path.join(dirpath, filename)

# import obj file with filepath above
imported_object = bpy.ops.import_scene.obj(filepath=filepath)
obj_object = bpy.context.selected_objects

print(obj_object)

# remove parts of the object (if necessary)
objs = bpy.data.objects
# objs.remove(objs['Body2'], True)
# objs.remove(objs['Body3'], True)
# objs.remove(objs['Body4'], True)
# objs.remove(objs['Body5'], True)
# obj_object = bpy.context.selected_objects

# merge all the obj objects into one if applicable
scene = bpy.context.scene

ctx = bpy.context.copy()
ctx['active_object'] = obj_object[0]
ctx['selected_objects'] = obj_object[1:]
ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obj_object]
bpy.ops.object.join(ctx)

# get the active object again
obj_object = bpy.context.selected_objects[0]

# apply transformations
table_settings = [
    [(0.004, 0.004, 0.004), (0, 0, 0), (-1, -1, 1.6)],
    [(0.004, 0.004, 0.004), (degree_to_rad(180), 0, 0), (-1, 1.1, 0.2)],
]

current_state = 0

obj_object.scale = table_settings[current_state][0]
obj_object.rotation_euler = table_settings[current_state][1]
obj_object.location = table_settings[current_state][2]

objs = []

# different camera settings to render from different starting positions / angles
# possible to iterate but script will take longer
camera_settings = [
    ([0, 4, 3], [degree_to_rad(60), 0, degree_to_rad(180)]),
    ([0, 7.5, 3], [degree_to_rad(75), 0, degree_to_rad(180)]),
    ([0, 5, 3], [degree_to_rad(65), 0, degree_to_rad(180)]),
    ([0, 10, 3], [degree_to_rad(77.5), 0, degree_to_rad(180)]),
    ([0, 4, 1], [degree_to_rad(89), 0, degree_to_rad(180)]),
    ([0, 7.5, 1], [degree_to_rad(89), 0, degree_to_rad(180)]),
    ([0, 5, 1], [degree_to_rad(89), 0, degree_to_rad(180)]),
    ([0, 10, 1], [degree_to_rad(89), 0, degree_to_rad(180)]),
    ([0, 4, 5], [degree_to_rad(45), 0, degree_to_rad(180)]),
    ([0, 7.5, 5], [degree_to_rad(60), 0, degree_to_rad(180)]),
    ([0, 5, 5], [degree_to_rad(50), 0, degree_to_rad(180)]),
    ([0, 10, 5], [degree_to_rad(65), 0, degree_to_rad(180)]),
    ([0, 4, 7.5], [degree_to_rad(30), 0, degree_to_rad(180)]),
    ([0, 7.5, 7.5], [degree_to_rad(50), 0, degree_to_rad(180)]),
    ([0, 5, 7.5], [degree_to_rad(37.5), 0, degree_to_rad(180)]),
    ([0, 10, 7.5], [degree_to_rad(55), 0, degree_to_rad(180)]),
    ([0, 4, 2], [degree_to_rad(70), 0, degree_to_rad(180)]),
    ([0, 7.5, 2], [degree_to_rad(80), 0, degree_to_rad(180)]),
    ([0, 5, 2], [degree_to_rad(75), 0, degree_to_rad(180)]),
    ([0, 10, 2], [degree_to_rad(80), 0, degree_to_rad(180)]),
    ([0, 1, 7.5], [degree_to_rad(10), 0, degree_to_rad(180)]),
]

bpy.ops.object.camera_add(view_align=False, location=[0, 0, 0], rotation=[0, 0, 0])

camera = bpy.context.object
scene.camera = camera
objs.append(camera)

# add three point lighting - might be interesting to randomize lighstning position as well
lamp_settings = [
    (5, 7.5, random.uniform(2, 3)),
    (-5, 7.5, random.uniform(2, 3)),
    (0, -5, random.uniform(2, 3)),
    (0, 0, 0),
    (0, 0, 7.5)
]

for pos in lamp_settings:
    bpy.ops.object.lamp_add(type='POINT', location=pos)
    objs.append(bpy.context.object)

# bpy.ops.object.lamp_add(type='SUN', location=[0, 0, 10])
# objs.append(bpy.context.object)
    
# create an empty object to rotate around during rendering
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
empty_parent = bpy.context.object

for obj in objs:
    obj.parent = empty_parent

out_dir = os.path.join(dirpath, 'data/blender/regular')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.cycles.film_transparent = True
bpy.context.scene.cycles.samples = 300


def randomize_lighting():
    objs[1].data.node_tree.nodes["Emission"].inputs[1].default_value = random.uniform(300, 1000)
    objs[2].data.node_tree.nodes["Emission"].inputs[1].default_value = random.uniform(200, 500)
    objs[3].data.node_tree.nodes["Emission"].inputs[1].default_value = random.uniform(200, 1000)
    objs[4].data.node_tree.nodes["Emission"].inputs[1].default_value = random.uniform(200, 300)
    objs[4].location = (random.uniform(0, 5), random.uniform(0, 5), random.uniform(0, 5))
    objs[5].data.node_tree.nodes["Emission"].inputs[1].default_value = random.uniform(0, 750)


for i, settings in enumerate(camera_settings):
    camera.location = settings[0]
    camera.rotation_euler = settings[1]
    for j in range(31):
        randomize_lighting()
        filename = 'data_camera_' + str(i) + '_pos_' + str(j) + '_state_' + str(current_state)
        bpy.context.scene.render.filepath = os.path.join(out_dir, filename)
        bpy.ops.render.render(write_still=True)
        empty_parent.rotation_euler = (0, 0, degree_to_rad(11.25 * (j + 1)))
