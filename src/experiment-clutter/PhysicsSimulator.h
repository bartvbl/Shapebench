#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmark-core/ComputedConfig.h"
#include "benchmark-core/Dataset.h"

struct ClutteredScene {
    ShapeDescriptor::cpu::Mesh referenceMesh;
    ShapeDescriptor::cpu::Mesh clutterGeometry;
};

void initPhysics();
ClutteredScene createClutteredScene(const nlohmann::json& config, const ComputedConfig& computedConfig, const Dataset& dataset, uint64_t randomSeed);