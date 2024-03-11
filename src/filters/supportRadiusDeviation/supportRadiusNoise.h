#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"


namespace ShapeBench {
    struct SupportRadiusDeviationOutput {
        nlohmann::json metadata;
    };

    SupportRadiusDeviationOutput applySupportRadiusNoise(ShapeBench::FilteredMeshPair &scene, uint64_t randomSeed, const nlohmann::json& config);
}
