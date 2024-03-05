#include "normalNoiseFilter.h"
#include "benchmarkCore/randomEngine.h"
#include "glm/glm.hpp"
#include "glm/ext/matrix_transform.hpp"

ShapeDescriptor::cpu::float3 computeDeviatedNormal(ShapeDescriptor::cpu::float3 inputNormal, float deviationAngleDegrees, float rotationAngleDegrees) {
    glm::vec3 originalNormal {inputNormal.x, inputNormal.y, inputNormal.z};
    glm::vec3 deviatedNormal {0, 0, 1};
    if(originalNormal.x == 0 && originalNormal.y == 0 && originalNormal.z < 0) {
        deviatedNormal = {0, 0, -1};
    }

    glm::mat3 deviationRotation = glm::rotate(glm::mat4(1.0), glm::radians(deviationAngleDegrees), glm::vec3(1, 0, 0));
    glm::mat3 directionRotation = glm::rotate(glm::mat4(1.0), glm::radians(rotationAngleDegrees), glm::vec3(0, 0, 1));
    deviatedNormal = (directionRotation * deviationRotation) * deviatedNormal;

    if(originalNormal.x == 0 && originalNormal.y == 0 && originalNormal.z > 0) {
        return normalize(ShapeDescriptor::cpu::float3(deviatedNormal.x, deviatedNormal.y, deviatedNormal.z));
    }

    // Align deviated normal with
    glm::vec3 orthogonalDirection = glm::cross(originalNormal, deviatedNormal);
    float angle = glm::acos(glm::dot(originalNormal, deviatedNormal));
    glm::mat3 rotationMatrix = glm::rotate(glm::mat4(1.0), angle, orthogonalDirection);
    deviatedNormal = rotationMatrix * originalNormal;

    return normalize(ShapeDescriptor::cpu::float3(deviatedNormal.x, deviatedNormal.y, deviatedNormal.z));
}

ShapeBench::NormalNoiseFilterOutput ShapeBench::applyNormalNoiseFilter(const nlohmann::json& config, FilteredMeshPair& filteredMesh, uint64_t randomSeed) {
    ShapeBench::NormalNoiseFilterOutput meta;

    float maxDeviation = config.at("filterSettings").at("normalVectorNoise").at("maxAngleDeviationDegrees");

    ShapeBench::randomEngine engine(randomSeed);
    std::uniform_real_distribution<float> deviationAngleDistribution(0, maxDeviation);
    std::uniform_real_distribution<float> rotationAngleDistribution(0, 360);

    // Displace reference point normals
    for(uint32_t i = 0; i < filteredMesh.mappedReferenceVertices.size(); i++) {
        ShapeDescriptor::cpu::float3& normal = filteredMesh.mappedReferenceVertices.at(i).normal;
        float deviationAngleDegrees = deviationAngleDistribution(engine);
        float rotationAngleDegrees = rotationAngleDistribution(engine);
        normal = computeDeviatedNormal(normal, deviationAngleDegrees, rotationAngleDegrees);
        meta.metadata["normal-noise-deviationAngle"].push_back(deviationAngleDegrees);
        meta.metadata["normal-noise-rotationAngle"].push_back(rotationAngleDegrees);
    }

    // Displace mesh normals
    for(uint32_t i = 0; i < filteredMesh.filteredSampleMesh.vertexCount; i++) {
        ShapeDescriptor::cpu::float3& normal = filteredMesh.filteredSampleMesh.normals[i];
        normal = computeDeviatedNormal(normal, deviationAngleDistribution(engine), rotationAngleDistribution(engine));
    }

    return meta;
}
