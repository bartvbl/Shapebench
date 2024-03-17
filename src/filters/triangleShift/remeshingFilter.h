#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"


namespace ShapeBench {
    namespace internal {

        void calculateAverageEdgeLength(const ShapeDescriptor::cpu::Mesh& mesh, double &averageEdgeLength, uint32_t &edgeIndex);
    }

    struct RemeshingFilterOutput {
        nlohmann::json metadata;
    };

    RemeshingFilterOutput remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config);
}
