#version 430 core

layout(location = 0) in vec3 position;
layout(location = 2) in vec3 texCoord;

layout(location = 1) out vec2 out_texCoord;

layout(location=16) uniform mat4x4 MVP;

void main()
{
	out_texCoord = texCoord.xy;

    gl_Position = MVP * vec4(position, 1.0f);
}
