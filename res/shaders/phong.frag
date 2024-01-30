#version 430 core

layout(location = 2) in vec3 vertexCoord;
layout(location = 3) in vec3 normal;

layout(location = 20) uniform vec4 materialDiffuseColour;

const vec3 eyePosition = vec3(0, 0, 0);
const float specularStrength = 0.5;

layout(location = 50) uniform vec3 lightPosition;

out vec4 color;

vec3 computeColourForLightSource(vec3 surfaceColour, vec3 lightSourcePosition, vec3 normalisedNormal, vec3 surfaceToEyeVector) {
	float distanceToLightSource = length(lightSourcePosition - vertexCoord);
	vec3 surfaceToLightVector = normalize(lightSourcePosition - vertexCoord);

	float attenuation = clamp(1.0 - distanceToLightSource*distanceToLightSource/(5 * 5), 0.0, 1.0);
	attenuation *= attenuation;
	attenuation = 1;

	float diffuse = max(dot(normalisedNormal, surfaceToLightVector) * attenuation, 0);
	vec3 diffuseColour = diffuse * surfaceColour;

	float specular = pow(max(dot(surfaceToEyeVector, reflect(-surfaceToLightVector, normalisedNormal)), 0), 4) * attenuation;
	vec3 specularColour = specular * vec3(0.3, 0.3, 0.3);

	return diffuseColour + specularColour;
}

void main()
{
	vec3 normalisedNormal = normalize(normal);

	vec3 surfaceToEyeVector = normalize(eyePosition - vertexCoord);

	vec3 finalColour = vec3(0, 0, 0);

	finalColour += computeColourForLightSource(materialDiffuseColour.xyz, lightPosition, normalisedNormal, surfaceToEyeVector);

	color = vec4(finalColour, materialDiffuseColour.a);
}
