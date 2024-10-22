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
                                   const ShapeBench::Dataset &dataset, ShapeBench::LocalDatasetCache* fileCache, uint64_t randomSeed) {
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

    // Not an entirely correct way to map all vertices, but this is the closest you can probably get
    // In any case, the portion that is not visible from the camera is straight up removed, so no orientation changes
    scene.sampleMeshTransformation *= glm::mat4(1.0);
    for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
        scene.additiveNoiseInfo.at(i).transformation *= glm::mat4(1.0);
    }

    return output;
}
