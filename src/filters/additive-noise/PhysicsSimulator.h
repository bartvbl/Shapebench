#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmark-core/ComputedConfig.h"
#include "benchmark-core/Dataset.h"
#include "filters/FiltereredMeshPair.h"

namespace ShapeBench {
    void initPhysics();
    void addClutterToScene(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed);
}
