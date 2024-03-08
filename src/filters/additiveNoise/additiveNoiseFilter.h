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
    struct AdditiveNoiseOutput {
        nlohmann::json metadata;
    };

    void initPhysics();
    void destroyPhysics();
    std::vector<ShapeBench::Orientation> runPhysicsSimulation(ShapeBench::AdditiveNoiseFilterSettings settings, const std::vector<ShapeDescriptor::cpu::Mesh>& meshes);
    AdditiveNoiseOutput runAdditiveNoiseFilter(AdditiveNoiseFilterSettings settings, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed, AdditiveNoiseCache& cache);



    inline AdditiveNoiseOutput applyAdditiveNoiseFilter(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed, AdditiveNoiseCache& cache) {
        const nlohmann::json& filterSettings = config.at("filterSettings").at("additiveNoise");

        AdditiveNoiseFilterSettings settings = readAdditiveNoiseFilterSettings(config, filterSettings);
        return runAdditiveNoiseFilter(settings, scene, dataset, randomSeed, cache);
    }
}
