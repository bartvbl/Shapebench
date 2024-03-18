#pragma once
#include <shapeDescriptor/shapeDescriptor.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel   CGALK;
typedef CGAL::Surface_mesh<CGALK::Point_3>                    CGALMesh;

namespace ShapeBench {
    CGALMesh convertSDMeshToCGAL(const ShapeDescriptor::cpu::Mesh& mesh, uint32_t* removedCount = nullptr);
    ShapeDescriptor::cpu::Mesh convertCGALMeshToSD(const CGALMesh& cgalMesh);
}