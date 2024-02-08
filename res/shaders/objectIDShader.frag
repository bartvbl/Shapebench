#version 430 core

layout(location = 1) in vec3 colour;

out vec4 color;

void main()
{
	color = vec4(colour, 1.0);
}
