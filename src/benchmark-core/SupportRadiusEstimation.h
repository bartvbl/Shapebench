#pragma once

#include <vector>
#include "Dataset.h"
#include "json.hpp"

namespace Shapebench {
    template<typename DescriptorMethod, typename DescriptorType, typename distanceFunction>
    float estimateSupportRadius(const nlohmann::json& config, const Dataset& dataset, uint64_t randomSeed);


}