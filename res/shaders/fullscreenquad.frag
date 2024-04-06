#version 430 core

layout(location = 1) in vec2 texCoords;

out vec4 color;

layout(binding = 0) uniform sampler2D texture0;

layout(location = 25) uniform int mode;

void main()
{
	if(mode == 0) {
		vec3 value = texture(texture0, texCoords).rgb;
		color = vec4(value.rgb, 1.0);
	} else if(mode == 1) {
		float value = texture(texture0, texCoords).r * 0.5;
		color = vec4(value, value, value, 1.0);
	}

}
