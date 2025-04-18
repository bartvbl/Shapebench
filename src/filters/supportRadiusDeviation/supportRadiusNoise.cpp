#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include <glm/ext/matrix_transform.hpp>
#include "supportRadiusNoise.h"
#include "benchmarkCore/randomEngine.h"

ShapeBench::FilterOutput
ShapeBench::SupportRadiusNoiseFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                            const Dataset &dataset,
                                            ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
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
    ShapeBench::FilterOutput output;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        scene.mappedReferenceVertices.at(i).vertex *= scaleFactor;
        nlohmann::json entry;
        entry["support-radius-scale-factor"] = scaleFactor;
        output.metadata.push_back(entry);
    }

    // The mesh itself does not move, so we don't modify these values
    // They're included here for the sake of completion
    scene.sampleMeshTransformation = glm::scale(scene.sampleMeshTransformation, glm::vec3(scaleFactor, scaleFactor, scaleFactor));
    scene.sampleMeshNormalTransformation *= glm::mat3(1.0);
    for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
        scene.additiveNoiseInfo.at(i).transformation *= glm::scale(scene.additiveNoiseInfo.at(i).transformation, glm::vec3(scaleFactor, scaleFactor, scaleFactor));
    }

    return output;
}

void ShapeBench::SupportRadiusNoiseFilter::saveCaches(const nlohmann::json& config) {

}

void ShapeBench::SupportRadiusNoiseFilter::destroy() {

}

void ShapeBench::SupportRadiusNoiseFilter::init(const nlohmann::json &config, bool invalidateCaches) {

}

ShapeBench::FilterOutput
ShapeBench::SupportRadiusNoiseFilter::applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                                  ShapeBench::FilteredMeshPair &scene,
                                                  const ShapeBench::Dataset &dataset,
                                                  ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    throw std::runtime_error("This filter is not meant to be applied on both meshes!");
    return {};
}

const std::string ShapeBench::SupportRadiusNoiseFilter::getFilterName() const {
    return "support-radius-deviation";
}

