#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"


namespace ShapeBench {
    struct MeshSimplificationFilterOutput {
        nlohmann::json metadata;
    };

    MeshSimplificationFilterOutput simplifyMesh(ShapeBench::FilteredMeshPair &scene, const nlohmann::json& config, uint64_t randomSeed);
}
