#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "TriangleMapping.h"

namespace Shapebench {
    struct FiltereredMeshPair {
        ShapeDescriptor::cpu::Mesh originalMesh;
        ShapeDescriptor::cpu::Mesh alteredMesh;
        Shapebench::TriangleMapping mapping;
    };
}