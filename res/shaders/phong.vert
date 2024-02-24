#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 2) out vec3 out_position;
layout(location = 3) out vec3 out_normal;

layout(location=30) uniform mat4x4 MVP;
layout(location=31) uniform mat4x4 MV;
layout(location=32) uniform mat3x3 normalMatrix;

void main()
{
    out_position = vec3(MV * vec4(position, 1.0));
    out_normal = normalMatrix * normal;

    gl_Position = MVP * vec4(position, 1.0f);
}
