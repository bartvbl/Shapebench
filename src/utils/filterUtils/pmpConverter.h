#pragma once

#include "pmp/surface_mesh.h"

namespace ShapeBench {
    pmp::SurfaceMesh convertSDMeshToPMP(const ShapeDescriptor::cpu::Mesh& mesh);
    ShapeDescriptor::cpu::Mesh convertPMPMeshToSD(const pmp::SurfaceMesh& mesh);
}