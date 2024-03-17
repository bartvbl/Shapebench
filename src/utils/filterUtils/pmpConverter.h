#pragma once

#include "pmp/surface_mesh.h"

namespace ShapeBench {
    void convertSDMeshToPMP(const ShapeDescriptor::cpu::Mesh& mesh, pmp::SurfaceMesh& pmpMesh, uint32_t* removedCount = nullptr);
    ShapeDescriptor::cpu::Mesh convertPMPMeshToSD(const pmp::SurfaceMesh& mesh);
}