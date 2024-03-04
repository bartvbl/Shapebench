#pragma once

#include "filters/FilteredMeshPair.h"
#include "json.hpp"

namespace ShapeBench {
    void applyNormalNoiseFilter(const nlohmann::json& config, FilteredMeshPair& filteredMesh, uint64_t randomSeed);
}