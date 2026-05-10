import glfw
import numpy as np

from collision import move

class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 1.5, 3.0])
        self.front = np.array([0.0, 0.0, -1.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.yaw = -90.0
        self.pitch = 0.0
        self.last_x = 360.0
        self.last_y = 360.0
        self.first_mouse = True
        self.sensitivity = 0.2
        self.max_mouse_delta = 30.0

    @staticmethod
    def normalize(vec):
        norm = np.linalg.norm(vec)
        if norm <= 1e-6:
            return vec
        return vec / norm

    def get_view_matrix(self):
        target = self.pos + self.front
        z_axis = self.normalize(self.pos - target)
        x_axis = self.normalize(np.cross(self.up, z_axis))
        y_axis = np.cross(z_axis, x_axis)
        view = np.eye(4)
        view[0, 0:3] = x_axis
        view[1, 0:3] = y_axis
        view[2, 0:3] = z_axis
        view[0, 3] = -np.dot(x_axis, self.pos)
        view[1, 3] = -np.dot(y_axis, self.pos)
        view[2, 3] = -np.dot(z_axis, self.pos)
        return view

    def get_projection_matrix(self, aspect):
        fov_rad = np.radians(58.0)
        f = 1.0 / np.tan(fov_rad / 2.0)
        near = 0.1
        far = 100.0
        proj = np.zeros((4, 4))
        proj[0, 0] = f / max(aspect, 1e-6)
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    def mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False
        dx = np.clip(xpos - self.last_x, -self.max_mouse_delta, self.max_mouse_delta)
        dy = np.clip(self.last_y - ypos, -self.max_mouse_delta, self.max_mouse_delta)
        xoffset = dx * self.sensitivity
        yoffset = dy * self.sensitivity
        self.last_x, self.last_y = xpos, ypos
        self.yaw = (self.yaw + xoffset) % 360.0
        self.pitch = np.clip(self.pitch + yoffset, -89.0, 89.0)

        direction = np.array([np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)), np.sin(np.radians(self.pitch)), np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))])
        self.front = self.normalize(direction)

    #     yaw = np.radians(self.yaw)
    #  
    def process_input(self, window, delta_time, walls=None):
        speed = 2.5 * delta_time
        yaw = np.radians(self.yaw)
        forward = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        forward = self.normalize(forward)
        right = self.normalize(np.cross(forward, self.up))
        dx = 0.0
        dz = 0.0
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            dx += forward[0] * speed
            dz += forward[2] * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            dx -= forward[0] * speed
            dz -= forward[2] * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            dx -= right[0] * speed
            dz -= right[2] * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            dx += right[0] * speed
            dz += right[2] * speed

        px = float(self.pos[0])
        pz = float(self.pos[2])
        if walls is None:
            self.pos[0] = px + dx
            self.pos[2] = pz + dz
            return

        nx, nz = move(px, pz, dx, dz, walls)
        self.pos[0] = nx
        self.pos[2] = nz