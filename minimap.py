import os

import cv2
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_STREAM_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    GL_VIEWPORT,
    GL_LINEAR,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GLint,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetIntegerv,
    glGetUniformLocation,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes


def _ortho(left, right, bottom, top):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 2] = -1.0
    m[3, 3] = 1.0
    return m.T


VERT = """#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
uniform mat4 ortho;
out vec2 vUV;
void main() {
    gl_Position = ortho * vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
"""

FRAG = """#version 330 core
in vec2 vUV;
uniform sampler2D tex;
uniform vec3 solid;
uniform float dotPass;
out vec4 o;
void main() {
    o = dotPass > 0.5 ? vec4(solid, 1.0) : texture(tex, vUV);
}
"""


def _upload_rgb(tex):
    tex = np.ascontiguousarray(tex, dtype=np.uint8)
    h, w = tex.shape[:2]
    prev = (GLint * 1)()
    glGetIntegerv(GL_UNPACK_ALIGNMENT, prev)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    try:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, tex)
    finally:
        glPixelStorei(GL_UNPACK_ALIGNMENT, int(prev[0]))


class Minimap:
    """Corner picture of your floorplan file + dot at camera (X,Z), same mapping as floorplan_cv."""

    def __init__(self, image_path: str, size_px: int = 180, margin: int = 12):
        self.size_px = int(size_px)
        self.margin = int(margin)
        self.active = False
        self.img_w = 1
        self.img_h = 1
        self.prog = None
        self.tex = None
        self.vao_bg = self.vbo_bg = None
        self.vao_dot = self.vbo_dot = None
        self.loc = {}

        path = image_path.strip() if image_path else ""
        if not path or not os.path.isfile(path):
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return

        self.img_h, self.img_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        flipped = cv2.flip(rgb, 0)

        self.prog = compileProgram(
            compileShader(VERT, GL_VERTEX_SHADER),
            compileShader(FRAG, GL_FRAGMENT_SHADER),
        )
        self.loc = {
            "ortho": glGetUniformLocation(self.prog, "ortho"),
            "tex": glGetUniformLocation(self.prog, "tex"),
            "solid": glGetUniformLocation(self.prog, "solid"),
            "dotPass": glGetUniformLocation(self.prog, "dotPass"),
        }

        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        _upload_rgb(flipped)
        glBindTexture(GL_TEXTURE_2D, 0)

        s = float(self.size_px)
        bg = np.array(
            [0, 0, 0, 0, s, 0, 1, 0, s, s, 1, 1, 0, 0, 0, 0, s, s, 1, 1, 0, s, 0, 1],
            dtype=np.float32,
        )
        self.vao_bg = glGenVertexArrays(1)
        self.vbo_bg = glGenBuffers(1)
        glBindVertexArray(self.vao_bg)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_bg)
        glBufferData(GL_ARRAY_BUFFER, bg.nbytes, bg, GL_STATIC_DRAW)
        st = 4 * bg.itemsize
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, st, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, st, ctypes.c_void_p(2 * bg.itemsize))
        glBindVertexArray(0)

        self.vao_dot = glGenVertexArrays(1)
        self.vbo_dot = glGenBuffers(1)
        glBindVertexArray(self.vao_dot)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_dot)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_STREAM_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

        self.active = True

    def _uv(self, wx, wz):
        # Inverse of floorplan_cv (image flipped for GL the same way as texture upload).
        iw, ih = float(self.img_w), float(self.img_h)
        sx, sz = 10.0 / max(iw, 1.0), 10.0 / max(ih, 1.0)
        u = wx / (iw * sx) + 0.5
        v = wz / (ih * sz) + 0.5
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

    def draw(self, camera_pos, fb_w, fb_h):
        if not self.active:
            return

        wx, wz = float(camera_pos[0]), float(camera_pos[2])
        u, v = self._uv(wx, wz)
        s = float(self.size_px)
        ortho = _ortho(0.0, s, 0.0, s)
        vp_x = max(fb_w - self.margin - self.size_px, 0)
        vp_y = self.margin

        prev = (GLint * 4)()
        glGetIntegerv(GL_VIEWPORT, prev)

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glViewport(vp_x, vp_y, self.size_px, self.size_px)

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.loc["ortho"], 1, GL_FALSE, ortho)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glUniform1i(self.loc["tex"], 0)
        glUniform1f(self.loc["dotPass"], 0.0)
        glBindVertexArray(self.vao_bg)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        r = max(s * 0.02, 2.5)
        cx, cy = u * s, v * s
        dot = np.array(
            [
                cx - r, cy - r, 0, 0,
                cx + r, cy - r, 0, 0,
                cx + r, cy + r, 0, 0,
                cx - r, cy - r, 0, 0,
                cx + r, cy + r, 0, 0,
                cx - r, cy + r, 0, 0,
            ],
            dtype=np.float32,
        )
        glUniform1f(self.loc["dotPass"], 1.0)
        glUniform3f(self.loc["solid"], 1.0, 0.15, 0.1)
        glBindVertexArray(self.vao_dot)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_dot)
        glBufferData(GL_ARRAY_BUFFER, dot.nbytes, dot, GL_STREAM_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glViewport(prev[0], prev[1], prev[2], prev[3])

    def cleanup(self):
        if not self.active:
            return
        glDeleteBuffers(1, [self.vbo_bg])
        glDeleteBuffers(1, [self.vbo_dot])
        glDeleteVertexArrays(1, [self.vao_bg])
        glDeleteVertexArrays(1, [self.vao_dot])
        glDeleteTextures(1, [self.tex])
        glDeleteProgram(self.prog)
