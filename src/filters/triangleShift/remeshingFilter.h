#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"


namespace ShapeBench {
    namespace internal {

        ShapeDescriptor::cpu::Mesh remeshMesh(const ShapeDescriptor::cpu::Mesh& mesh, const nlohmann::basic_json<> &config);
    }

    struct RemeshingFilterOutput {
        nlohmann::json metadata;
    };

    RemeshingFilterOutput remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config);
}
