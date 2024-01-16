#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmark-core/ComputedConfig.h"
#include "benchmark-core/Dataset.h"

ShapeDescriptor::cpu::Mesh createOccludedScene(const nlohmann::json &json, const ComputedConfig &config, const Dataset &dataset, uint64_t seed);