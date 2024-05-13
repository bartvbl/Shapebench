#pragma once

#include <filesystem>

namespace ShapeBench {
    struct AdditiveNoiseFilterSettings {
        std::filesystem::path compressedDatasetRootDir;
        uint32_t addedClutterObjectCount = 1;
        bool enableDebugRenderer = true;
        bool runSimulationUntilManualExit = false;
        float objectAttractionForceMagnitude = 5000.0f;
        float initialObjectSeparation = 2.1;
        uint32_t simulationFrameRate = 165;
        uint32_t simulationStepLimit = 2500;
        uint32_t tempAllocatorSizeBytes = 100 * 1024 * 1024;
        uint32_t maxConvexHulls = 64;
        uint32_t convexHullGenerationResolution = 400000;
        uint32_t convexHullGenerationRecursionDepth = 64;
        uint32_t convexHullGenerationMaxIntermediateHulls = 30000;
        uint32_t convexHullGenerationMaxVerticesPerHull = 256;
        float floorFriction = 0.5;
        float minRequiredObjectVolume = 0.000000001;
    };

    inline AdditiveNoiseFilterSettings readAdditiveNoiseFilterSettings(const nlohmann::json &config, const nlohmann::json &filterSettings) {
        AdditiveNoiseFilterSettings settings;
        settings.compressedDatasetRootDir = std::string(config.at("datasetSettings").at("compressedRootDir"));
        settings.addedClutterObjectCount = filterSettings.at("addedObjectCount");
        settings.enableDebugRenderer = filterSettings.at("enableDebugCamera");
        settings.runSimulationUntilManualExit = filterSettings.at("runSimulationUntilManualExit");
        settings.objectAttractionForceMagnitude = filterSettings.at("objectAttractionForceMagnitude");
        settings.initialObjectSeparation = filterSettings.at("initialObjectSeparation");
        settings.simulationFrameRate = filterSettings.at("simulationFramerate");
        settings.simulationStepLimit = filterSettings.at("simulationStepLimit");
        settings.maxConvexHulls = filterSettings.at("maxConvexHulls");
        settings.convexHullGenerationResolution = filterSettings.at("convexHullGenerationResolution");
        settings.convexHullGenerationRecursionDepth = filterSettings.at("convexHullGenerationRecursionDepth");
        settings.convexHullGenerationMaxIntermediateHulls = filterSettings.at("convexHullGenerationMaxIntermediateHulls");
        settings.convexHullGenerationMaxVerticesPerHull = filterSettings.at("convexHullGenerationMaxVerticesPerHull");
        settings.floorFriction = filterSettings.at("floorFriction");
        settings.minRequiredObjectVolume = filterSettings.at("minRequiredObjectVolume");
        return settings;
    }
}
