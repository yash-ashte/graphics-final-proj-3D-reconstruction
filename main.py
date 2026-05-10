import argparse
import os

import glfw
from camera import Camera
from extrusion import build_rm
from floorplan_cv import ext_walls
from mesh import Mesh
from minimap import Minimap
from renderer import Renderer
from classifier import classify_floorplan
from collision import wall_hit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floorplan", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    floorplan_path = args.floorplan.strip()
    if floorplan_path and not os.path.exists(floorplan_path):
        print(f"Floorplan path does not exist: {floorplan_path}")
        return
    
    wall_data = ext_walls(floorplan_path)
    classification = classify_floorplan(
        wall_data,
        model_path=None, #Work in progress
        image_path=floorplan_path if floorplan_path else None
    )

    print(f"building type={classification['label']} source={classification['source']}")

    mat = classification["material"]
    wall_spec  = mat["spec_strength"]
    wall_shine = mat["shininess"]
    wall_mesh_data = []
    for wall in wall_data["walls"]:
        w_vertices, w_indices = build_rm([wall], wall_data["bounds"], include_floor=False)
        if w_indices.size > 0:
            wall_mesh_data.append((w_vertices, w_indices))
    floor_vertices, floor_indices = build_rm([], wall_data["bounds"], include_floor=True)
    print(f"walls={len(wall_data['walls'])} bounds={wall_data['bounds']}")

    if not glfw.init():
        return
    primary_monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(primary_monitor)
    win_w, win_h = video_mode.size.width, video_mode.size.height
    window = glfw.create_window(win_w, win_h, "3D Room Reconstruction", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    minimap = Minimap(floorplan_path)
    camera = Camera()
    walls = wall_data["walls"]
    if wall_hit(float(camera.pos[0]), float(camera.pos[2]), walls):
        min_x, max_x, min_z, max_z = wall_data["bounds"]
        camera.pos[0] = (min_x + max_x) * 0.5
        camera.pos[2] = (min_z + max_z) * 0.5
    last_time = glfw.get_time()
    cursor_disabled = True
    tab_latch = False
    last_debug_time = 0.0
    wall_meshes = [Mesh(v, i) for v, i in wall_mesh_data]
    floor_mesh = Mesh(floor_vertices, floor_indices)
    renderer = Renderer()
    renderer.set_scene_textures(classification["label"])
   ## wall_palette = [
     ##   (0.95, 0.35, 0.35),
       ## (0.35, 0.95, 0.35),
       ## (0.35, 0.55, 0.95),
       ## (0.95, 0.85, 0.35),
        ##(0.85, 0.35, 0.95),
       ## (0.35, 0.9, 0.9),
    ##]
    wall_palette = [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ]

    floor_material = {"color": (0.8, 0.8, 0.82), "spec_strength": 0.18, "shininess": 10.0}

    glfw.set_cursor_pos_callback(window, camera.mouse_callback)

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # tab_pressed = glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS
        

        camera.process_input(window, delta_time, walls)

        width, height = glfw.get_framebuffer_size(window)
        aspect = width / max(height, 1)
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(aspect)

        renderer.begin_frame()
        renderer.set_camera(view, projection, camera.pos)

        #walls
        renderer.use_wall()
        for idx, wall_mesh in enumerate(wall_meshes):
            wall_color = wall_palette[idx % len(wall_palette)]
            renderer.set_material(color=wall_color, spec=wall_spec, shininess=wall_shine)
            wall_mesh.draw()

        #floor
        renderer.use_floor()
        renderer.set_material(
            color=floor_material["color"],
            spec=floor_material["spec_strength"],
            shininess=floor_material["shininess"],
        )
        floor_mesh.draw()

        #minimap
        minimap.draw(camera.pos, width, height)

        if current_time - last_debug_time > 1.0:
            last_debug_time = current_time
            cx, cy, cz = camera.pos
            print(f"[debug] pos=({cx:.2f}, {cy:.2f}, {cz:.2f}) yaw={camera.yaw:.1f} pitch={camera.pitch:.1f}")

        glfw.swap_buffers(window)
        glfw.poll_events()

    for wall_mesh in wall_meshes:
        wall_mesh.delete()
    floor_mesh.delete()
    renderer.cleanup()
    minimap.cleanup()
    glfw.terminate()

if __name__ == "__main__":
    main()