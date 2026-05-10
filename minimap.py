import os
import cv2
import numpy as np
import OpenGL.GL as gl
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
    prev = (gl.GLint * 1)()
    gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT, prev)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    try:
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, tex)
    finally:
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, int(prev[0]))


class Minimap:
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
            compileShader(VERT, gl.GL_VERTEX_SHADER),
            compileShader(FRAG, gl.GL_FRAGMENT_SHADER),
        )
        self.loc = {
            "ortho": gl.glGetUniformLocation(self.prog, "ortho"),
            "tex": gl.glGetUniformLocation(self.prog, "tex"),
            "solid": gl.glGetUniformLocation(self.prog, "solid"),
            "dotPass": gl.glGetUniformLocation(self.prog, "dotPass"),
        }

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        _upload_rgb(flipped)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        s = float(self.size_px)
        bg = np.array(
            [0, 0, 0, 0, s, 0, 1, 0, s, s, 1, 1, 0, 0, 0, 0, s, s, 1, 1, 0, s, 0, 1],
            dtype=np.float32,
        )
        self.vao_bg = gl.glGenVertexArrays(1)
        self.vbo_bg = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao_bg)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_bg)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, bg.nbytes, bg, gl.GL_STATIC_DRAW)
        st = 4 * bg.itemsize
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, st, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, st, ctypes.c_void_p(2 * bg.itemsize))
        gl.glBindVertexArray(0)
        self.vao_dot = gl.glGenVertexArrays(1)
        self.vbo_dot = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao_dot)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_dot)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 6 * 4 * 4, None, gl.GL_STREAM_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))
        gl.glBindVertexArray(0)
        self.active = True

    def _uv(self, wx, wz):
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
        prev = (gl.GLint * 4)()
        gl.glGetIntegerv(gl.GL_VIEWPORT, prev)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glViewport(vp_x, vp_y, self.size_px, self.size_px)
        gl.glUseProgram(self.prog)
        gl.glUniformMatrix4fv(self.loc["ortho"], 1, gl.GL_FALSE, ortho)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glUniform1i(self.loc["tex"], 0)
        gl.glUniform1f(self.loc["dotPass"], 0.0)
        gl.glBindVertexArray(self.vao_bg)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
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
        gl.glUniform1f(self.loc["dotPass"], 1.0)
        gl.glUniform3f(self.loc["solid"], 1.0, 0.15, 0.1)
        gl.glBindVertexArray(self.vao_dot)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_dot)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, dot.nbytes, dot, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glViewport(prev[0], prev[1], prev[2], prev[3])

    def cleanup(self):
        if not self.active:
            return
        gl.glDeleteBuffers(1, [self.vbo_bg])
        gl.glDeleteBuffers(1, [self.vbo_dot])
        gl.glDeleteVertexArrays(1, [self.vao_bg])
        gl.glDeleteVertexArrays(1, [self.vao_dot])
        gl.glDeleteTextures(1, [self.tex])
        gl.glDeleteProgram(self.prog)