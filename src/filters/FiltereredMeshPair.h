#pragma once

#include <shapeDescriptor/containerTypes.h>

namespace Shapebench {
    struct FiltereredMeshPair {
        ShapeDescriptor::cpu::Mesh originalMesh;
        ShapeDescriptor::cpu::Mesh alteredMesh;
        ShapeDescriptor::OrientedPoint originalReferenceVertex;
        ShapeDescriptor::OrientedPoint mappedReferenceVertex;
    };
}