#include "OcclusionFilter.h"
#include "benchmarkCore/randomEngine.h"
#include <random>


void ShapeBench::OcclusionFilter::init(const nlohmann::json &config, bool invalidateCaches) {
    uint32_t visibilityImageWidth = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(0);
    uint32_t visibilityImageHeight = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(1);
    sceneGenerator.init(visibilityImageWidth, visibilityImageHeight);
}

void ShapeBench::OcclusionFilter::destroy() {
    sceneGenerator.destroy();
}

void ShapeBench::OcclusionFilter::saveCaches(const nlohmann::json &config) {

}

ShapeBench::FilterOutput
ShapeBench::OcclusionFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                   const Dataset &dataset,
                                   ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    ShapeBench::randomEngine randomEngine(randomSeed);
    ShapeBench::FilterOutput output;
    OcclusionRendererSettings renderSettings;
    renderSettings.nearPlaneDistance = config.at("filterSettings").at("subtractiveNoise").at("nearPlaneDistance");
    renderSettings.farPlaneDistance = config.at("filterSettings").at("subtractiveNoise").at("farPlaneDistance");
    renderSettings.fovy = config.at("filterSettings").at("subtractiveNoise").at("fovYAngleRadians");
    renderSettings.objectDistanceFromCamera = config.at("filterSettings").at("subtractiveNoise").at("objectDistanceFromCamera");

    std::uniform_real_distribution<float> distribution(0, 1);
    renderSettings.yaw = float(distribution(randomEngine) * 2.0 * M_PI);
    renderSettings.pitch = float((distribution(randomEngine) - 0.5) * M_PI);
    renderSettings.roll = float(distribution(randomEngine) * 2.0 * M_PI);

    nlohmann::json entry;
    entry["subtractive-noise-pitch"] = renderSettings.pitch;
    entry["subtractive-noise-yaw"] = renderSettings.yaw;
    entry["subtractive-noise-roll"] = renderSettings.roll;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        output.metadata.push_back(entry);
    }

    sceneGenerator.computeOccludedMesh(renderSettings, scene);

    // This filter only removes triangles, so no need to transform the sample vertices

    return output;
}

ShapeBench::FilterOutput
ShapeBench::OcclusionFilter::applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                         ShapeBench::FilteredMeshPair &scene, const ShapeBench::Dataset &dataset,
                                         ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    throw std::runtime_error("This filter is not meant to be applied on both meshes!");
    return {};
}

const std::string ShapeBench::OcclusionFilter::getFilterName() const {
    return "subtractive-noise";
}

