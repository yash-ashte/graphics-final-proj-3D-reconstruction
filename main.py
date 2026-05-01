import argparse
import os

import glfw
from camera import Camera
from extrusion import build_room_mesh
from floorplan_cv import extract_wall_segments
from mesh import Mesh
from minimap import Minimap
from renderer import Renderer
from classifier import classify_floorplan

def parse_args():
    parser = argparse.ArgumentParser(description="2D floorplan to 3D room reconstruction MVP.")
    parser.add_argument("--floorplan", type=str, default="", help="Path to floorplan image.")
    return parser.parse_args()


def main():
    args = parse_args()
    floorplan_path = args.floorplan.strip()
    if floorplan_path and not os.path.exists(floorplan_path):
        print(f"[error] Floorplan path does not exist: {floorplan_path}")
        return
    
    wall_data = extract_wall_segments(floorplan_path)

    # after wall_data is extracted
    classification = classify_floorplan(
        wall_data,
        model_path=None,
        image_path=floorplan_path if floorplan_path else None
    )

    print(f"[info] building type={classification['label']} source={classification['source']}")

    mat = classification["material"]
    wall_spec  = mat["spec_strength"]
    wall_shine = mat["shininess"]

    # Precompute scene geometry before opening the window.
    ##wall_data = extract_wall_segments(floorplan_path)
    wall_mesh_data = []
    for wall in wall_data["walls"]:
        w_vertices, w_indices = build_room_mesh([wall], wall_data["bounds"], include_floor=False)
        if w_indices.size > 0:
            wall_mesh_data.append((w_vertices, w_indices))
    floor_vertices, floor_indices = build_room_mesh([], wall_data["bounds"], include_floor=True)
    print(f"[info] demo walls={len(wall_data['walls'])} bounds={wall_data['bounds']}")

    if not glfw.init():
        return

    window = glfw.create_window(720, 720, "3D Room Reconstruction Demo", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    minimap = Minimap(floorplan_path)

    camera = Camera()
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

        tab_pressed = glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS
        if tab_pressed and not tab_latch:
            cursor_disabled = not cursor_disabled
            mode = glfw.CURSOR_DISABLED if cursor_disabled else glfw.CURSOR_NORMAL
            glfw.set_input_mode(window, glfw.CURSOR, mode)
            camera.first_mouse = True
        tab_latch = tab_pressed

        camera.process_input(window, delta_time)

        width, height = glfw.get_framebuffer_size(window)
        aspect = width / max(height, 1)
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(aspect)

        renderer.begin_frame()
        renderer.set_camera(view, projection, camera.pos)

        renderer.use_wall()
        for idx, wall_mesh in enumerate(wall_meshes):
            wall_color = wall_palette[idx % len(wall_palette)]
            renderer.set_material(color=wall_color, spec=wall_spec, shininess=wall_shine)
            wall_mesh.draw()

        # draw floor with floor texture
        renderer.use_floor()
        renderer.set_material(
            color=floor_material["color"],
            spec=floor_material["spec_strength"],
            shininess=floor_material["shininess"],
        )
        floor_mesh.draw()

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