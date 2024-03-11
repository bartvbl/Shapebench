#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "supportRadiusNoise.h"
#include "benchmarkCore/randomEngine.h"

ShapeBench::SupportRadiusDeviationOutput ShapeBench::applySupportRadiusNoise(ShapeBench::FilteredMeshPair& scene, uint64_t randomSeed, const nlohmann::json& config) {
    float deviationLimit = config.at("filterSettings").at("supportRadiusDeviation").at("maxRadiusDeviation");

    ShapeBench::randomEngine generator(randomSeed);
    std::uniform_real_distribution<float> distribution(1 - deviationLimit, 1 + deviationLimit);

    float scaleFactor = distribution(generator);

    // Scale meshes
    for(uint32_t vertexIndex = 0; vertexIndex < scene.filteredSampleMesh.vertexCount; vertexIndex++) {
        scene.filteredSampleMesh.vertices[vertexIndex] *= scaleFactor;
    }
    for(uint32_t vertexIndex = 0; vertexIndex < scene.filteredAdditiveNoise.vertexCount; vertexIndex++) {
        scene.filteredAdditiveNoise.vertices[vertexIndex] *= scaleFactor;
    }

    // Update reference points
    ShapeBench::SupportRadiusDeviationOutput output;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        scene.mappedReferenceVertices.at(i).vertex *= scaleFactor;
        nlohmann::json entry;
        entry["support-radius-scale-factor"] = scaleFactor;
        output.metadata.push_back(entry);
    }

    return output;
}
