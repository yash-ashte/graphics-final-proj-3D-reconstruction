import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_REPEAT,
    GL_RGB,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_FALSE,
    glActiveTexture,
    glBindTexture,
    glClear,
    glClearColor,
    glDeleteProgram,
    glDeleteTextures,
    glEnable,
    glGenTextures,
    glGetUniformLocation,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
)
from OpenGL.GL.shaders import compileProgram, compileShader


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

    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specStrength * spec * lightColor;

    vec3 texColor = texture(tex0, fragUV).rgb;
    vec3 finalColor = (ambient + diffuse + specular) * texColor * objectColor;
    FragColor = vec4(finalColor, 1.0);
}
"""


class Renderer:
    def __init__(self):
        glEnable(GL_DEPTH_TEST)
        self.program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        self.texture = self._build_checker_texture()
        self.model = np.eye(4, dtype=np.float32)

    def _build_checker_texture(self):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        size = 64
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                if (x // 8 + y // 8) % 2 == 0:
                    img[y, x] = np.array([190, 190, 190], dtype=np.uint8)
                else:
                    img[y, x] = np.array([110, 110, 110], dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        return tex

    def begin_frame(self):
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.program, "tex0"), 0)

    def set_camera(self, view, projection, camera_pos):
        glUniformMatrix4fv(glGetUniformLocation(self.program, "model"), 1, GL_FALSE, self.model.T)
        glUniformMatrix4fv(glGetUniformLocation(self.program, "view"), 1, GL_FALSE, view.T)
        glUniformMatrix4fv(glGetUniformLocation(self.program, "projection"), 1, GL_FALSE, projection.T)
        glUniform3f(glGetUniformLocation(self.program, "viewPos"), camera_pos[0], camera_pos[1], camera_pos[2])
        glUniform3f(glGetUniformLocation(self.program, "lightPos"), 3.0, 5.0, 3.0)
        glUniform3f(glGetUniformLocation(self.program, "lightColor"), 1.0, 1.0, 1.0)

    def set_material(self, object_color=(1.0, 1.0, 1.0), spec_strength=0.3, shininess=16.0):
        glUniform3f(glGetUniformLocation(self.program, "objectColor"), *object_color)
        glUniform1f(glGetUniformLocation(self.program, "specStrength"), float(spec_strength))
        glUniform1f(glGetUniformLocation(self.program, "shininess"), float(shininess))

    def cleanup(self):
        glDeleteProgram(self.program)
        glDeleteTextures(1, [self.texture])
