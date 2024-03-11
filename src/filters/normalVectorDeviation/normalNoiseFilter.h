#pragma once

#include "filters/FilteredMeshPair.h"
#include "json.hpp"

namespace ShapeBench {
    struct NormalNoiseFilterOutput {
        nlohmann::json metadata;
    };

    NormalNoiseFilterOutput applyNormalNoiseFilter(const nlohmann::json& config, FilteredMeshPair& filteredMesh, uint64_t randomSeed);
}