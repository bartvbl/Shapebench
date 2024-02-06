#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "benchmarkCore/Dataset.h"
#include "filters/FiltereredMeshPair.h"

namespace ShapeBench {
    void initPhysics();
    void addClutterToScene(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed);
}
