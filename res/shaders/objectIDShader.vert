#version 430 core

layout(location = 0) in vec3 position;
layout(location = 3) in vec3 colour;

layout(location=16) uniform mat4x4 MVP;

layout(location = 1) out vec3 out_colour;

void main()
{
    out_colour = colour;
    gl_Position = MVP * vec4(position, 1.0f);
}
