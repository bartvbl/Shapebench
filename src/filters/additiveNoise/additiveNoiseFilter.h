#pragma once

class AdditiveNoiseCache;
#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "benchmarkCore/Dataset.h"
#include "filters/FilteredMeshPair.h"
#include "AdditiveNoiseCache.h"
#include "AdditiveNoiseFilterSettings.h"

namespace ShapeBench {
    void initPhysics();
    std::vector<ShapeBench::Orientation> runPhysicsSimulation(ShapeBench::AdditiveNoiseFilterSettings settings, const std::vector<ShapeDescriptor::cpu::Mesh>& meshes);
    void runAdditiveNoiseFilter(AdditiveNoiseFilterSettings settings, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed, AdditiveNoiseCache& cache);



    inline void applyAdditiveNoiseFilter(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed, AdditiveNoiseCache& cache) {
        const nlohmann::json& filterSettings = config.at("filterSettings").at("additiveNoise");

        AdditiveNoiseFilterSettings settings = readAdditiveNoiseFilterSettings(config, filterSettings);
        runAdditiveNoiseFilter(settings, scene, dataset, randomSeed, cache);
    }
}
