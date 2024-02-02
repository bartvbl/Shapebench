#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "json.hpp"
#include "benchmark-core/ComputedConfig.h"
#include "benchmark-core/Dataset.h"
#include "filters/subtractive-noise/OcclusionGenerator.h"


template<typename DescriptorMethod, typename DescriptorType>
void runSubtractiveNoiseExperiment(const nlohmann::json& config, const ComputedConfig& computedConfig, const Dataset& dataset, uint64_t randomSeed) {
    OccludedSceneGenerator occlusionGenerator(config, computedConfig);
    ShapeDescriptor::cpu::Mesh mesh;
    ShapeDescriptor::cpu::Mesh occludedMesh = occlusionGenerator.computeOccludedMesh(mesh, randomSeed);
}


