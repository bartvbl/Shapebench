#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"
#include "vector3.h"
#include "isotropichalfedgemesh.h"


namespace ShapeBench {
    namespace internal {
        void createRemeshingMesh(const ShapeDescriptor::cpu::Mesh& mesh, std::vector<Vector3>& output_vertices, std::vector<std::vector<size_t>>& output_indices);
        ShapeDescriptor::cpu::Mesh convertRemeshingToSDMesh(IsotropicHalfedgeMesh *halfedgeMesh);
        ShapeDescriptor::cpu::Mesh remeshMesh(const ShapeDescriptor::cpu::Mesh& mesh, const nlohmann::basic_json<> &config);
    }

    struct RemeshingFilterOutput {
        nlohmann::json metadata;
    };

    RemeshingFilterOutput remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config);
}
