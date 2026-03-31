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
