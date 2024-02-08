#version 430 core

layout(location = 1) in vec2 texCoords;

out vec4 color;

layout(binding = 0) uniform sampler2D texture0;

void main()
{
	vec3 value = texture(texture0, texCoords).rgb;
	color = vec4(value.rgb, 1.0);
}
