#pragma once

class AdditiveNoiseCache;
#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "dataset/Dataset.h"
#include "filters/FilteredMeshPair.h"
#include "AdditiveNoiseCache.h"
#include "AdditiveNoiseFilterSettings.h"
#include "filters/Filter.h"

namespace ShapeBench {

    class AdditiveNoiseFilter : public ShapeBench::Filter {
        AdditiveNoiseCache additiveNoiseCache;

    public:
        void init(const nlohmann::json& config) override;
        void destroy() override;
        void saveCaches(const nlohmann::json& config) override;

        FilterOutput apply(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed) override;
    };

    std::vector<ShapeBench::Orientation> runPhysicsSimulation(ShapeBench::AdditiveNoiseFilterSettings settings, const std::vector<ShapeDescriptor::cpu::Mesh>& meshes);
}
