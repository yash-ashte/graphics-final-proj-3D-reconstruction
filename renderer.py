import os
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


TEXTURE_MAP = {
    "office": {
        "wall": "assets/Concrete.jpg",
        "floor": "assets/Office_Carpet.jpg",
    },
    "hallway": {
        "wall": "assets/Brick.jpg",
        "floor": "assets/Concrete.jpg",
    },
    "apartment": {
        "wall": "assets/Hardwood.jpg",
        "floor": "assets/Office_Carpet.jpg",
    },
}


VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragPos;
out vec3 fragNormal;
out vec2 fragUV;

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    fragPos = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(model))) * aNormal;
    fragUV = aUV;
    gl_Position = projection * view * worldPos;
}
"""


FRAGMENT_SHADER = """
#version 330 core
in vec3 fragPos;
in vec3 fragNormal;
in vec2 fragUV;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform sampler2D tex0;
uniform float specStrength;
uniform float shininess;

out vec4 FragColor;

void main() {
    vec3 ambient = 0.2 * lightColor;

    vec3 n = normalize(fragNormal);
    vec3 l = normalize(lightPos - fragPos);
    float diff = max(dot(n, l), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 v = normalize(viewPos - fragPos);
    vec3 r = reflect(-l, n);
    float spec = pow(max(dot(v, r), 0.0), shininess);
    vec3 specular = specStrength * spec * lightColor;

    vec3 tex = texture(tex0, fragUV).rgb;
    vec3 lighting = ambient + diffuse + specular;

    FragColor = vec4(lighting * tex * objectColor, 1.0);
}
"""


class Renderer:
    def __init__(self):
        glEnable(GL_DEPTH_TEST)

        self.program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

        self.model = np.eye(4)
        checker = self._make_checker()
        self.wall_tex = checker
        self.floor_tex = checker
        self.active_tex = self.wall_tex

    def _make_checker(self, size=64, tile=8):
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        img = np.zeros((size, size, 3), dtype=np.uint8)

        c0 = np.array([190, 190, 190], dtype=np.uint8)
        c1 = np.array([110, 110, 110], dtype=np.uint8)

        for y in range(size):
            for x in range(size):
                img[y, x] = c0 if ((x // tile + y // tile) % 2 == 0) else c1

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            size,
            size,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            img,
        )

        return tex_id

    def _load_texture(self, path):
        if not path or not os.path.exists(path):
            print(f"[warn] texture missing: {path}")
            return self._make_checker()

        img = Image.open(path).convert("RGB")
        data = np.array(img, dtype=np.uint8)
        h, w = data.shape[:2]

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            w,
            h,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            data,
        )

        return tex_id

    def set_scene_textures(self, label):
        cfg = TEXTURE_MAP.get(label)

        if not cfg:
            print(f"[warn] unknown label '{label}', keeping current textures")
            return
        glDeleteTextures(1, [self.wall_tex])
        glDeleteTextures(1, [self.floor_tex])

        self.wall_tex = self._load_texture(cfg["wall"])
        self.floor_tex = self._load_texture(cfg["floor"])

        self.active_tex = self.wall_tex

        print(f"[renderer] switched to '{label}' textures")

    def begin_frame(self):
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.active_tex)
        glUniform1i(glGetUniformLocation(self.program, "tex0"), 0)

    def set_camera(self, view, proj, cam_pos):
        glUniformMatrix4fv(
            glGetUniformLocation(self.program, "model"),
            1, GL_FALSE, self.model.T
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.program, "view"),
            1, GL_FALSE, view.T
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.program, "projection"),
            1, GL_FALSE, proj.T
        )

        glUniform3f(glGetUniformLocation(self.program, "viewPos"), *cam_pos)
        glUniform3f(glGetUniformLocation(self.program, "lightPos"), 3.0, 5.0, 3.0)
        glUniform3f(glGetUniformLocation(self.program, "lightColor"), 1.0, 1.0, 1.0)

    def set_material(self, color=(1.0, 1.0, 1.0), spec=0.3, shininess=16.0):
        glUniform3f(glGetUniformLocation(self.program, "objectColor"), *color)
        glUniform1f(glGetUniformLocation(self.program, "specStrength"), spec)
        glUniform1f(glGetUniformLocation(self.program, "shininess"), shininess)

    def use_wall(self):
        self.active_tex = self.wall_tex
        glBindTexture(GL_TEXTURE_2D, self.wall_tex)

    def use_floor(self):
        self.active_tex = self.floor_tex
        glBindTexture(GL_TEXTURE_2D, self.floor_tex)

    def cleanup(self):
        glDeleteProgram(self.program)
        glDeleteTextures(1, [self.wall_tex])
        glDeleteTextures(1, [self.floor_tex])