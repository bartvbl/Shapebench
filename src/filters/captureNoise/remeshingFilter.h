#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"


namespace ShapeBench {
    struct RemeshingFilterOutput {
        nlohmann::json metadata;
    };

    RemeshingFilterOutput remesh(ShapeBench::FilteredMeshPair &scene);
}
