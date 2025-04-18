#include "gaussianNoiseFilter.h"
#include "benchmarkCore/randomEngine.h"
#include "glm/glm.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "utils/filterUtils/gaussianNoise.h"

ShapeBench::FilterOutput ShapeBench::GaussianNoiseFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                                    const Dataset &dataset,
                                                    ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    ShapeBench::FilterOutput meta;

    float minStandardDeviation = config.at("filterSettings").at("gaussianNoise").at("minStandardDeviation");
    float maxStandardDeviation = config.at("filterSettings").at("gaussianNoise").at("maxStandardDeviation");

    ShapeBench::randomEngine engine(randomSeed);

    std::uniform_real_distribution<float> intensityDistribution(minStandardDeviation, maxStandardDeviation);
    float deviation = intensityDistribution(engine);

    ShapeBench::applyGaussianNoise(scene.filteredSampleMesh, engine(), deviation);
    ShapeBench::applyGaussianNoise(scene.filteredAdditiveNoise, engine(), deviation);

    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        if(!scene.mappedVertexIncluded.at(i)) {
            nlohmann::json metadataEntry;
            metadataEntry["gaussian-noise-max-deviation"] = deviation;
            meta.metadata.push_back(metadataEntry);
            continue;
        }
        ShapeDescriptor::cpu::float3 originalVertex = scene.mappedReferenceVertices.at(i).vertex;

        // Update vertex location to the displaced location
        scene.mappedReferenceVertices.at(i).vertex = scene.filteredSampleMesh.vertices[scene.mappedReferenceVertexIndices.at(i)];

        nlohmann::json metadataEntry;
        metadataEntry["gaussian-noise-max-deviation"] = deviation;
        metadataEntry["gaussian-noise-vertex-deviation"] = length(originalVertex - scene.mappedReferenceVertices.at(i).vertex);
        meta.metadata.push_back(metadataEntry);
    }

    // The mesh itself does not move, so we don't modify these values
    // They're included here for the sake of completion
    scene.sampleMeshTransformation *= glm::mat4(1.0);
    scene.sampleMeshNormalTransformation *= glm::mat3(1.0);
    for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
        scene.additiveNoiseInfo.at(i).transformation *= glm::mat4(1.0);
    }


    //std::cout << randomSeed << " -> " << deviation << std::endl;

    return meta;
}

void ShapeBench::GaussianNoiseFilter::init(const nlohmann::json &config, bool invalidateCaches) {

}

void ShapeBench::GaussianNoiseFilter::destroy() {

}

void ShapeBench::GaussianNoiseFilter::saveCaches(const nlohmann::json& config) {

}

ShapeBench::FilterOutput
ShapeBench::GaussianNoiseFilter::applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                             ShapeBench::FilteredMeshPair &scene, const ShapeBench::Dataset &dataset,
                                             ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    throw std::runtime_error("This filter is not meant to be applied on both meshes!");
    return {};
}

const std::string ShapeBench::GaussianNoiseFilter::getFilterName() const {
    return "gaussian-noise";
}
