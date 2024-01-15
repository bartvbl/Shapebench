#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "json.hpp"
#include "benchmark-core/ComputedConfig.h"
#include "benchmark-core/Dataset.h"



template<typename DescriptorMethod, typename DescriptorType>
void runClutterExperiment(const nlohmann::json& config, const ComputedConfig& computedConfig, const Dataset& dataset, uint64_t randomSeed) {
    ShapeDescriptor::cpu::Mesh occludedMesh = createOccludedScene(config, computedConfig, dataset, randomSeed);
}


