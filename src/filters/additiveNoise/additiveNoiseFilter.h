#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "benchmarkCore/Dataset.h"
#include "filters/FilteredMeshPair.h"

namespace ShapeBench {
    struct AdditiveNoiseFilterSettings {
        std::filesystem::path compressedDatasetRootDir;
        uint32_t addedClutterObjectCount = 1;
        bool enableDebugRenderer = true;
        uint32_t simulationFrameRate = 165;
    };


    void initPhysics();
    void runAdditiveNoiseFilter(AdditiveNoiseFilterSettings settings, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed);
    inline void applyAdditiveNoiseFilter(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed) {
        AdditiveNoiseFilterSettings settings;
        settings.addedClutterObjectCount = config.at("filterSettings").at("additiveNoise").at("addedObjectCount");
        settings.compressedDatasetRootDir = std::string(config.at("compressedDatasetRootDir"));
        runAdditiveNoiseFilter(settings, scene, dataset, randomSeed);
    }
}
