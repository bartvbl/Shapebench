#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "TriangleMapping.h"

namespace Shapebench {
    struct AlteredMesh {
        ShapeDescriptor::cpu::Mesh mesh;
        Shapebench::TriangleMapping mapping;
    };
}