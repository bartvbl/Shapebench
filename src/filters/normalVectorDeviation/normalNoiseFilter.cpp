#include "normalNoiseFilter.h"
#include "benchmarkCore/randomEngine.h"
#include "glm/glm.hpp"
#include "glm/ext/matrix_transform.hpp"

ShapeDescriptor::cpu::float3 computeDeviatedNormal(ShapeDescriptor::cpu::float3 inputNormal, float deviationAngleDegrees, float rotationAngleDegrees) {
    glm::vec3 originalNormal {inputNormal.x, inputNormal.y, inputNormal.z};
    glm::vec3 xAxis {1, 0, 0};
    glm::vec3 zAxis {0, 0, 1};

    glm::mat3 deviationRotation = glm::rotate(glm::mat4(1.0), glm::radians(deviationAngleDegrees), xAxis);
    glm::mat3 directionRotation = glm::rotate(glm::mat4(1.0), glm::radians(rotationAngleDegrees), zAxis);
    glm::vec3 unalignedDeviatedNormal = (directionRotation * deviationRotation) * zAxis;

    // Avoids a zero division when computing the cross product
    // This can either happen when the original normal is the z-axis, or the negative z-axis
    if(originalNormal.x == 0 && originalNormal.y == 0) {
        if(originalNormal.z > 0) {
            return normalize(ShapeDescriptor::cpu::float3(unalignedDeviatedNormal.x, unalignedDeviatedNormal.y, unalignedDeviatedNormal.z));
        } else if(originalNormal.z < 0) {
            return normalize(ShapeDescriptor::cpu::float3(-unalignedDeviatedNormal.x, -unalignedDeviatedNormal.y, -unalignedDeviatedNormal.z));
        } else {
            throw std::runtime_error("Zero or invalid normal vector detected: " + std::to_string(originalNormal.x) + ", " + std::to_string(originalNormal.y) + ", " + std::to_string(originalNormal.z));
        }
    }

    // Align deviated normal with
    glm::vec3 orthogonalDirection = glm::cross(originalNormal, zAxis);
    float angle = glm::acos(glm::dot(originalNormal, zAxis));
    glm::mat3 rotationMatrix = glm::rotate(glm::mat4(1.0), -angle, orthogonalDirection);
    glm::vec3 deviatedNormal = rotationMatrix * unalignedDeviatedNormal;

    return normalize(ShapeDescriptor::cpu::float3(deviatedNormal.x, deviatedNormal.y, deviatedNormal.z));
}

ShapeBench::FilterOutput ShapeBench::NormalNoiseFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                                  const Dataset &dataset,
                                                  ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    ShapeBench::FilterOutput meta;

    float maxDeviation = config.at("filterSettings").at("normalVectorNoise").at("maxAngleDeviationDegrees");

    ShapeBench::randomEngine engine(randomSeed);
    std::uniform_real_distribution<float> deviationAngleDistribution(0, maxDeviation);
    std::uniform_real_distribution<float> rotationAngleDistribution(0, 360);

    // Displace reference point normals
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        ShapeDescriptor::cpu::float3& normal = scene.mappedReferenceVertices.at(i).normal;
        float deviationAngleDegrees = deviationAngleDistribution(engine);
        float rotationAngleDegrees = rotationAngleDistribution(engine);
        normal = computeDeviatedNormal(normal, deviationAngleDegrees, rotationAngleDegrees);
        nlohmann::json metadataEntry;
        metadataEntry["normal-noise-deviationAngle"] = deviationAngleDegrees;
        metadataEntry["normal-noise-rotationAngle"] = rotationAngleDegrees;
        meta.metadata.push_back(metadataEntry);
    }

    // Displace mesh normals
    for(uint32_t i = 0; i < scene.filteredSampleMesh.vertexCount; i++) {
        ShapeDescriptor::cpu::float3& normal = scene.filteredSampleMesh.normals[i];
        float deviationAngleDegrees = deviationAngleDistribution(engine);
        float rotationAngleDegrees = rotationAngleDistribution(engine);
        normal = computeDeviatedNormal(normal, deviationAngleDegrees, rotationAngleDegrees);
    }

    for(uint32_t i = 0; i < scene.filteredAdditiveNoise.vertexCount; i++) {
        ShapeDescriptor::cpu::float3& normal = scene.filteredAdditiveNoise.normals[i];
        float deviationAngleDegrees = deviationAngleDistribution(engine);
        float rotationAngleDegrees = rotationAngleDistribution(engine);
        normal = computeDeviatedNormal(normal, deviationAngleDegrees, rotationAngleDegrees);
    }

    // The mesh itself does not move, so we don't modify these values
    // They're included here for the sake of completion
    scene.sampleMeshTransformation *= glm::mat4(1.0);
    scene.sampleMeshNormalTransformation *= glm::mat3(1.0);
    for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
        scene.additiveNoiseInfo.at(i).transformation *= glm::mat4(1.0);
    }

    return meta;
}

void ShapeBench::NormalNoiseFilter::init(const nlohmann::json &config, bool invalidateCaches) {

}

void ShapeBench::NormalNoiseFilter::destroy() {

}

void ShapeBench::NormalNoiseFilter::saveCaches(const nlohmann::json& config) {

}



ShapeBench::FilterOutput
ShapeBench::NormalNoiseFilter::applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                           ShapeBench::FilteredMeshPair &scene, const ShapeBench::Dataset &dataset,
                                           ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    throw std::runtime_error("This filter is not meant to be applied on both meshes!");
    return {};
}

const std::string ShapeBench::NormalNoiseFilter::getFilterName() const {
    return "normal-noise";
}


