#include "MultiViewOcclusion.h"
#include "benchmarkCore/randomEngine.h"
#include <random>


void ShapeBench::MultiViewOcclusionFilter::init(const nlohmann::json &config, bool invalidateCaches) {
    uint32_t visibilityImageWidth = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(0);
    uint32_t visibilityImageHeight = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(1);
    sceneGenerator.init(visibilityImageWidth, visibilityImageHeight);
}

void ShapeBench::MultiViewOcclusionFilter::destroy() {
    sceneGenerator.destroy();
}

void ShapeBench::MultiViewOcclusionFilter::saveCaches(const nlohmann::json &config) {

}

ShapeBench::FilterOutput
ShapeBench::MultiViewOcclusionFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                   const Dataset &dataset,
                                   ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed,
                                   const nlohmann::json &filterOutputPreviousSequence) {
    ShapeBench::randomEngine randomEngine(randomSeed);
    ShapeBench::FilterOutput output;
    OcclusionRendererSettings renderSettings;
    renderSettings.nearPlaneDistance = config.at("filterSettings").at("multiViewOcclusion").at("nearPlaneDistance");
    renderSettings.farPlaneDistance = config.at("filterSettings").at("multiViewOcclusion").at("farPlaneDistance");
    renderSettings.fovy = config.at("filterSettings").at("multiViewOcclusion").at("fovYAngleRadians");
    renderSettings.objectDistanceFromCamera = config.at("filterSettings").at("multiViewOcclusion").at("objectDistanceFromCamera");

    bool isSecondFilterSequence = filterOutputPreviousSequence.contains("multi-view-occlusion-angle-between-objects");
    float maxAngleBetweenObjectsDegrees = config.at("filterSettings").at("multiViewOcclusion").at("maxAngleBetweenObjectsDegrees");

    std::uniform_real_distribution<float> distribution(0, 1);
    float angleBetweenObjects = ((distribution(randomEngine) * maxAngleBetweenObjectsDegrees) / 180.0f) * float(M_PI);
    renderSettings.yawDeviation = isSecondFilterSequence ? (angleBetweenObjects / 2.0f) : -(angleBetweenObjects / 2.0f);
    renderSettings.yaw = float(distribution(randomEngine) * 2.0 * M_PI);
    renderSettings.pitch = float((distribution(randomEngine) - 0.5) * M_PI);
    renderSettings.roll = float(distribution(randomEngine) * 2.0 * M_PI);

    nlohmann::json entry;
    entry["multi-view-occlusion-pitch"] = renderSettings.pitch;
    entry["multi-view-occlusion-yaw"] = renderSettings.yaw;
    entry["multi-view-occlusion-roll"] = renderSettings.roll;
    entry["multi-view-occlusion-angle-between-objects"] = renderSettings.yawDeviation;
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
