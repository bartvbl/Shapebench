#pragma once

#include <shapeDescriptor/containerTypes.h>

namespace ShapeBench {
    struct FilteredMeshPair {
        ShapeDescriptor::cpu::Mesh originalMesh;
        ShapeDescriptor::cpu::Mesh alteredMesh;
        ShapeDescriptor::OrientedPoint originalReferenceVertex;
        ShapeDescriptor::OrientedPoint mappedReferenceVertex;
    };
}