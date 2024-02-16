#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "benchmarkCore/Dataset.h"
#include "filters/FilteredMeshPair.h"
#include "AdditiveNoiseCache.h"

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
        uint32_t tempAllocatorSizeBytes = 10 * 1024 * 1024;
        uint32_t maxConvexHulls = 64;
        uint32_t convexHullGenerationResolution = 400000;
        uint32_t convexHullGenerationRecursionDepth = 64;
        uint32_t convexHullGenerationMaxVerticesPerHull = 256;
        float floorFriction = 0.5;
    };


    void initPhysics();
    std::vector<ShapeBench::Orientation> runPhysicsSimulation(ShapeBench::AdditiveNoiseFilterSettings settings,
                                                              ShapeBench::FilteredMeshPair& scene,
                                                              const ShapeBench::Dataset& dataset,
                                                              uint64_t randomSeed,
                                                              const std::vector<ShapeDescriptor::cpu::Mesh>& meshes);
    void runAdditiveNoiseFilter(AdditiveNoiseFilterSettings settings, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed, AdditiveNoiseCache& cache);

    inline AdditiveNoiseFilterSettings readAdditiveNoiseFilterSettings(const nlohmann::json &config, const nlohmann::json &filterSettings) {
        AdditiveNoiseFilterSettings settings;
        settings.compressedDatasetRootDir = std::string(config.at("compressedDatasetRootDir"));
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
        settings.convexHullGenerationMaxVerticesPerHull = filterSettings.at("convexHullGenerationMaxVerticesPerHull");
        settings.floorFriction = filterSettings.at("floorFriction");
        return settings;
    }

    inline void applyAdditiveNoiseFilter(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed, AdditiveNoiseCache& cache) {
        const nlohmann::json& filterSettings = config.at("filterSettings").at("additiveNoise");

        AdditiveNoiseFilterSettings settings = readAdditiveNoiseFilterSettings(config, filterSettings);
        runAdditiveNoiseFilter(settings, scene, dataset, randomSeed, cache);
    }
}
